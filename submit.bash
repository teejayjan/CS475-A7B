#!/bin/bash
#SBATCH -J MPIAutocorrelation
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o auto.out
#SBATCH -e auto.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jant@oregonstate.edu

mpic++ auto.cpp -o auto
mpiexec -mca btl self,tcp -np 4 auto