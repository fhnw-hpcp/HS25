import os
import socket

if not 'SLURM_ARRAY_JOB_ID' in os.environ.keys():
    raise RuntimeError('this is not an slurm array job!')

job_id = os.environ['SLURM_ARRAY_JOB_ID']
task_id = os.environ['SLURM_ARRAY_TASK_ID']
cluster_id = os.environ['SLURM_CLUSTER_NAME']
partition_id = os.environ['SLURM_JOB_PARTITION']
host_name=socket.gethostname()

print(f'Executing array job {job_id} task {task_id} on cluster "{cluster_id}" partition "{partition_id}" host "{host_name}"')
