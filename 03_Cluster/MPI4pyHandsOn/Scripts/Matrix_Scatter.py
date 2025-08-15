from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

matrixSize = 6 #Square matrix

if matrixSize % size != 0:
    print ("Warning, number of rows is not multiple of number of cores")

# Only rank 0 creates the full array
if rank == 0:
    A = np.arange(matrixSize**2, dtype=np.float64).reshape(matrixSize, matrixSize)
    print(f"Root process ({rank}) created the Matrix:\n{A}")
else:
    A = None #This declaration is mandatory

# Scatter the rows
# Calculate how many rows each process gets (above we made sure that no remainder)
MatrixRowsPerProcess = matrixSize//size
#Allocate memory for row-slices (also for process 0)
A_rows = np.empty((MatrixRowsPerProcess, matrixSize), dtype=np.float64)

# Scatter the rows (collective operation)
comm.Scatter(A, A_rows, root=0)

# Print the result on each process
print(f"Process {rank} received:\n{A_rows}")


