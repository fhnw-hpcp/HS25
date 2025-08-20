#!/bin/bash
#SBATCH --job-name=array-job
#SBATCH --cluster=merlin6
#SBATCH --partition=hourly
#SBATCH --array=1-10
#SBATCH --time=00:00:30
#SBATCH --output=array-job/out%A-%a
module load anaconda
module list
CMD="python3 array-job.py"
echo "$SLURM_JOB_ID: $CMD"
srun $CMD    # srun optional here
