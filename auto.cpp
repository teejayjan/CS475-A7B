#include <stdio.h>
#include <math.h>
#include <mpi.h>

// Which node is in charge? 
#define BOSS 0

// Files to read and write
#define BIGSIGNALFILEBIN (char*)"bigsignal.bin"
#define CSVPLOTFILE (char*)"plot.csv"

// Tag to "scatter"
#define TAG_SCATTER 'S'

// Tag to "gather"
#define TAG_GATHER 'G'

// How many elements are in the big signal
#define NUMELEMENTS (8*1024*1024)

// Do this many shifts
#define MAXSHIFTS 1024

// How many autocorrelation sums to plot (an excel limit)
#define MAXPLOT 256 

#define BINARY

// Print debug messages? 
#define DEBUG true

// Globals
float *BigSums;     // overall MAXSHIFTS autocorrelation array
float *BigSignal;   // overall NUMELEMENTS-big signal data
int    NumCpus;     // total # of cpus involved
float *PPSums;      // per-processor autocorrelation sums
float *PPSignal;    // per-processor local array to hold the sub-signal
int    PPSize;      // per-processor local array size

// Function prototype
void DoOneLocalAutocorrelation(int);

// module load slurm
// module load openmpi/3.1
// mpic++ -o auto auto.cpp
// mpiexec -np 4 -mca btl tcp,self auto
int main(int argc, char *argv[])
{
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &NumCpus);
    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    // Decide how much data to send each processor
    PPSize = NUMELEMENTS / NumCpus;

    // Local arrays
    PPSignal = new float [PPSize + MAXSHIFTS];  // per-processor local signal
    PPSums = new float [MAXSHIFTS];             // per-processor local sums of the products

    // Read the BigSignal Array
    if (me == BOSS)
    {
        BigSignal = new float [NUMELEMENTS + MAXSHIFTS];

    // #ifndef BINARY
        FILE *fp = fopen(BIGSIGNALFILEBIN, "rb");
        if (fp == NULL)
        {
            fprintf(stderr, "Cannot open data file\n");
            return -1;
        }

        fread(BigSignal, sizeof(float), NUMELEMENTS, fp);
    // #endif

        // Duplicate part of the array
        for (int i = 0; i < MAXSHIFTS; i++)
        {
            BigSignal[NUMELEMENTS + i] = BigSignal[i];
        }
    }

    // Create the array to hold all the sums
    if (me == BOSS)
    {
        BigSums = new float [MAXSHIFTS];
    }

    // Start the timer
    double time0 = MPI_Wtime();

    // Have the boss "send" to itself (not really send, just copy)
    if (me == BOSS)
    {
        for (int i = 0; i < PPSize + MAXSHIFTS; i++)
        {
            PPSignal[i] = BigSignal[BOSS*PPSize + i];
        }
    }

    // Have the boss send to everyone else
    if (me == BOSS)
    {
        for (int dst = 0; dst < NumCpus; dst++)
        {
            if (dst == BOSS)
            {
                continue;
            }

            MPI_Send(&BigSignal[dst*PPSize], PPSize, MPI_FLOAT, dst, TAG_SCATTER, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(PPSignal, PPSize, MPI_FLOAT, BOSS, TAG_SCATTER, MPI_COMM_WORLD, &status);
    }

    // Each processor does its own autocorrelation
    DoOneLocalAutocorrelation(me);

    // Each processor sends its sums back to the Boss
    if (me == BOSS)
    {
        for (int s = 0; s < MAXSHIFTS; s++)
        {
            BigSums[s] = PPSums[s]; // Start overall sums with the BOSS' sums
        }
    }
    else
    {
        MPI_Send(PPSums, MAXSHIFTS, MPI_FLOAT, BOSS, TAG_GATHER, MPI_COMM_WORLD);
    }

    // Receive the sums and add them into the overall sums
    if (me == BOSS)
    {
        float tmpSums[MAXSHIFTS];
        for (int src = 0; src < NumCpus; src++)
        {
            if (src == BOSS)
            {
                continue;
            }
            MPI_Recv(tmpSums, MAXSHIFTS, MPI_FLOAT, src, TAG_GATHER, MPI_COMM_WORLD, &status);
            // fprintf(stderr, "summing data from %d\n", src);
            for (int s = 0; s < MAXSHIFTS; s++)
                BigSums[s] += tmpSums[s];
        }
    }
    
    // Stop the timer
    double time1 = MPI_Wtime();

    // Print performance
    if (me == BOSS)
    {
        double seconds = time1 - time0;
        double performance = (double)MAXSHIFTS*(double)NUMELEMENTS/seconds/1000000.;        // mega-elements computed per second
        fprintf(stderr, "%3d processors, %10d elements, %9.2lf mega-autocorrelations computed per second\n", NumCpus, NUMELEMENTS, performance);
    }

    // Write the file to be plotted to look for sine wave
    if (me == BOSS)
    {
        FILE *fp = fopen(CSVPLOTFILE, "w");
        if (fp == NULL)
        {
            fprintf(stderr, "Cannot write to plot file\n");
        }
        else
        {
            for (int s = 1; s < MAXPLOT; s++)
            {
                fprintf(fp, "%6d , %10.2f\n", s, BigSums[s]);
            }
            fclose(fp);
        }
    }

    MPI_Finalize();
    return 0;
}

void DoOneLocalAutocorrelation(int me)
{
    MPI_Status status;
    if (DEBUG) fprintf(stderr, "Node %3d entered DoOneLocalAutocorrelation()\n", me);

    for (int s = 0; s < MAXSHIFTS; s++)
    {
        float sum = 0.;
        for (int i = 0; i < PPSize; i++)
        {
            sum += PPSignal[i] * PPSignal[i + s];
        }
        PPSums[s] = sum;
    }
}