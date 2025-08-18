from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

imageSize = 3

if rank == 0:
    #We emulate a (small) detector image, typically rank 0 would load it from file
    detImage_2BeSent = np.arange(imageSize**2, dtype=np.int64).reshape(imageSize,imageSize)
    comm.Send(detImage_2BeSent, dest=1, tag=13)
    print ("0 sent:\n", detImage_2BeSent)
elif rank == 1:
    data = np.empty(imageSize**2, dtype=np.int64)
    comm.Recv(data, source=0, tag=13)
    detImage_Recv = data.reshape(imageSize,imageSize)
    print ("1 recvd:\n", detImage_Recv)
