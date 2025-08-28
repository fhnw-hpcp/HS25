#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH -J ringComm
#SBATCH -o ringComm.out
#SBATCH -e ringComm.err
module load anaconda
conda activate summer-school-hpc-2025
#No need to specify -n CLA, Slurm knows it
mpirun python RingComm.py

