#!/bin/bash
#SBATCH --job-name=no-srun
#SBATCH --cluster=merlin6
#SBATCH --partition=hourly
#SBATCH --time=00:00:30
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --output=use-srun/sbatch-%j.out
bash -c 'echo "$SLURM_PROCID: $(hostname) - no srun"'
srun bash -c 'echo "$SLURM_PROCID: $(hostname) - srun"'
