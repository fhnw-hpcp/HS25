from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

vectorLength = 4
# Root process creates the 1D array
if rank == 0:
    x = np.arange(vectorLength, dtype=np.float64)
    print(f"Root process ({rank}) created the array: {x}")
else:
    # Other processes initialize an empty array of the same size
    x = np.empty(vectorLength, dtype=np.float64)

#Broadcast the array from root to all processes
comm.Bcast(x, root=0)

#Print the received array on each process
print(f"Process {rank} received: {x}")
