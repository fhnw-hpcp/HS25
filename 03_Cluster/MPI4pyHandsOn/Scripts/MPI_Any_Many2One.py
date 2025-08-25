from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank < size -1:
    data = rank*np.ones(3, dtype=np.float64)
    # Send the data to rank 1 with tag 100
    comm.Send(data, dest=size-1, tag=np.random.randint(size))
    print(f"[Rank {rank}] Sent data: {data} to Rank {size-1}")
else: #rank size -1
    numberOfReceivedMessages = 0
    data = np.empty(3, dtype=np.float64)  # Same buffer for all receives
    while numberOfReceivedMessages < size -1:
        # Use ANY_SOURCE and ANY_TAG to receive from any rank/tag
        status = MPI.Status()
        comm.Recv(data, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        print(f"[Rank {rank}] Received data: {data} from rank {status.Get_source()} with tag {status.Get_tag()}")
        numberOfReceivedMessages += 1

