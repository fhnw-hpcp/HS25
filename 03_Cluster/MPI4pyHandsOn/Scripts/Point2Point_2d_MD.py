from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dataTag = 13
metaDataTag = 14

if rank == 0:
    #We emulate a (small) detector image, typically rank 0 would load it from file
    #Hence only rank 0 knows its size (might be varying). Datatype is assumed fixed
    imageSize = 3 #this information would e.g be derived from hdf5 metadata from image directly
    #imageSize = 50 #Deadlocks, but this is implementation dependent
    detImage_2BeSent = np.arange(imageSize**2, dtype=np.int64).reshape(imageSize,imageSize)
    #Send data first
    comm.Send(detImage_2BeSent, dest=1, tag=dataTag)
    #comm.Ssend(detImage_2BeSent, dest=1, tag=dataTag) #Deadlocks
    print ("0 sent:\n", detImage_2BeSent)
    #Send Meta data (we could also send an int here, but for a non quadratic image would be an array)
    comm.Send(np.array(imageSize, dtype=np.int64), dest=1, tag=metaDataTag)

elif rank == 1:
    #Receive Metadata first (We must know how large the buffer is that we allocate
    imageSize_asArray = np.empty(1, dtype=np.int64)
    comm.Recv(imageSize_asArray, source=0, tag=metaDataTag)
    imageSize = imageSize_asArray[0]
    #Now that we know what to expect we can allocate receive buffer
    data = np.empty(imageSize**2, dtype=np.int64)
    comm.Recv(data, source=0, tag=dataTag)
    detImage_Recv = data.reshape(imageSize, imageSize)
    print ("1 recvd:\n", detImage_Recv)
