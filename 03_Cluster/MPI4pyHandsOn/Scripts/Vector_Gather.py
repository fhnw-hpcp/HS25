from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Each rank inits ist share
vectorLengthLocal = 4
vectorLocal = np.arange(vectorLengthLocal, dtype=np.float64)
print(f"Process {rank} created local vector: {vectorLocal}")

# Root process prepares to receive all data
if rank == 0:
    vector = np.empty(size*vectorLengthLocal, dtype=np.float64)
else:
    vector = None

# Gather all local arrays to root
comm.Gather(vectorLocal, vector, root=0)
# Root process prints the gathered array
if rank == 0:
    print(f"Root process ({rank}) gathered vector: {vector}")
