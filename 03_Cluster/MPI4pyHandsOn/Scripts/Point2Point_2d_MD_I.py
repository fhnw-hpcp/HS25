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
    detImage_2BeSent = np.arange(imageSize**2, dtype=np.int64).reshape(imageSize,imageSize)
    #Send data first (non-blocking)
    req_data = comm.Isend(detImage_2BeSent, dest=1, tag=dataTag)
    print ("0 sent:\n", detImage_2BeSent)
    #Send Meta data
    req_metadata = comm.Isend(np.array(imageSize, dtype=np.int64), dest=1, tag=metaDataTag)
    req_data.Wait()
    req_metadata.Wait()
elif rank == 1:
    #Receive Metadata first (We must know how large the buffer is that we allocate
    imageSize_asArray = np.empty(1, dtype=np.int64)
    req_metadata = comm.Irecv(imageSize_asArray, source=0, tag=metaDataTag)
    req_metadata.Wait() #make sure we have data here (comment out line and se what happens)
    imageSize = imageSize_asArray[0]
    #Now that we know what to expect we can allocate receive buffer
    data = np.empty(imageSize**2, dtype=np.int64)
    req_data = comm.Irecv(data, source=0, tag=dataTag)
    req_data.Wait()
    detImage_Recv = data.reshape(imageSize, imageSize)
    print ("1 recvd:\n", detImage_Recv)
