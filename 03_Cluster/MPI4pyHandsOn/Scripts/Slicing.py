from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert size == 3

n = 10

assert n % 2 == 0

if rank == 0:
    doReallocate = False
    data = np.arange(n)
    if doReallocate:
        comm.Send(np.copy(data[::2]), dest=1, tag=13)
        comm.Send(np.copy(data[1::2]), dest=2, tag=13)
    else:
        comm.Send(data[::2], dest=1, tag=13)
        comm.Send(data[1::2], dest=2, tag=13)
    print ("0 sent to 1:", data[::2])
    print ("0 sent to 2:", data[1::2])
elif rank == 1:
    data = np.empty(n//2, dtype=np.int64)
    comm.Recv(data, source=0, tag=13)
    print ("1 recvd:", data)
else:
    data = np.empty(n//2, dtype=np.int64)
    comm.Recv(data, source=0, tag=13)
    print ("2 recvd:", data)
