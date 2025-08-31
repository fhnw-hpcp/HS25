from mpi4py import MPI
import numpy as np
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure we are using 4 processes
assert size == 4, "This script is designed for exactly 4 MPI processes."

global_size = int(sys.argv[1])
global_array = None

if rank == 0:
    # Root process creates the array (random int values)
    global_array = np.random.randint(low=0, high=10*global_size, size=global_size)
    #print (global_array)
    start_time = time.time()

# Split the array into 4 chunks (one per process)
local_size = global_size // size
local_array = np.empty(local_size, dtype=np.int64)

# Scatter the array to all processes
comm.Scatter(global_array, local_array, root=0)

#if rank == 0:   start_time = time.time() #Computation only (no IPC)
# Compute the local maximum
local_max = np.max(local_array)
print(f"Process {rank}: Local max = {local_max}")

# Reduce all local maxima to find the global maximum
global_max = comm.reduce(local_max, op=MPI.MAX, root=0)

# Print the global maximum on the root process
if rank == 0:
    end_time = time.time()
    print(f"Time to find maximum: {end_time - start_time} sec")
    print(f"Global maximum: {global_max:}")
