'''
Parallel (row-wise) Matrix times vector multiplication Ax = b
'''
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

matrixSize = 4 #Square matrix
matrixRowsPerRank = matrixSize//size
vectorLength = matrixSize #Mandatory for well defined multiplication

if matrixSize % size != 0:
    if rank == 0:
        print ("WARNING: Number of matrix rows is not multiple of number of cores")
        comm.Abort(1)

#Allocate and/or initialize/define data (rank dependent)
#--------------------------------------------------------
if rank == 0:
    A = np.arange(matrixSize**2, dtype=np.float64).reshape(matrixSize, matrixSize)
    x = np.arange(vectorLength, dtype=np.float64)
    b = np.empty(vectorLength, dtype=np.float64)
else:
    A = None
    x = np.empty(vectorLength, dtype=np.float64)
    b = None

# Scatter Matrix
# --------------
A_local = np.empty((matrixRowsPerRank, matrixSize), dtype=np.float64)
comm.Scatter(A, A_local, root=0)

# Broadcast vector
# ----------------
comm.Bcast(x, root=0)

# Do parallel computation
# ------------------------
b_local = np.dot(A_local, x)

# Gather local results
# --------------------
comm.Gather(b_local, b, root=0)

# Unit Test (Possible since process 0 holds entire A in memory)
# -------------------------------------------------------------
if rank == 0:
    print ("Parallel computation equals sequential :", np.allclose(np.dot(A, x), b))
