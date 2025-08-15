#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH -J ringComm
#SBATCH -o ringComm.out
#SBATCH -e ringComm.err
export PATH=/psi/home/studer_a1/data/condaforge/miniforge/bin:$PATH
source activate mpi
mpiexec -n 4 python ringComm.py

