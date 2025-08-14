from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    # Non-blocking send
    data = np.asarray([1, 2, 3])
    req_send = comm.Isend(data, dest=1, tag=77)
    # Do some computation while the send is in progress
    result = 42**2
    # Wait for the send to complete
    req_send.Wait()
    print ("Sent:", data)
elif rank == 1:
    #Allocate memory for the receive buffer
    data = np.empty(3, np.int64)
    # Non-blocking receive
    req_recv = comm.Irecv(data, source=0, tag=77)
    # Do some computation while the receive is in progress
    result = 100/26
    # Wait for the receive to complete
    req_recv.Wait()
    print("Received:", data)
