from mpi4py import MPI
import numpy as np
from scipy.ndimage import laplace

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert size == 2

nr = 220
nc = 200
assert nr % 2 == 0
nh = nr//2

dt = 0.1
alpha = 1
bound = 1
numberOfIterations = 100

doExchangeHalo = True #Set to False to make UT fail

def update(u):
    return u + dt*alpha*laplace(u, mode='constant', cval=bound)

u_0 = np.arange(nr*nc, dtype = np.float64).reshape(nr,nc)

if rank == 0:
    u_upper = u_0[:nh +1, :]
else:
    u_lower = u_0[nh -1:, :]

for _ in range(numberOfIterations):
    if rank == 0:
        u_upper = update(u_upper)
        if doExchangeHalo:
            comm.Send(u_upper[-2,:], dest=1, tag=12)
            comm.Recv(u_upper[-1,:], source=1, tag=13)
    else:
        u_lower = update(u_lower)
        if doExchangeHalo:
            comm.Recv(u_lower[0,:], source=0, tag=12)
            comm.Send(u_lower[1,:], dest=0, tag=13)

if rank == 0:
    u_f_total = np.zeros_like(u_0)
    u_f_total[:nh, :] = u_upper[:-1, :]
    comm.Recv(u_f_total[nh:,:], source=1, tag=14)
else: #<- only since we know that we are running on 2 cores
    comm.Send(u_lower[1:,:], dest=0, tag=14)


if rank == 0:
    u = u_0
    for _ in range(numberOfIterations):
        u = update(u)
    u_f_total_seq = u
    print ("Parallel integration equals sequential :", np.allclose(u_f_total, u_f_total_seq))

