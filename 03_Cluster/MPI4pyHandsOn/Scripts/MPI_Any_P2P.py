from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    # Send the data to rank 1 with tag 10
    comm.Send(data, dest=1, tag=10)
    print(f"[Rank 0] Sent data: {data} to Rank 1")

elif rank == 1:
    # Prepare a NumPy array to receive the data
    data = np.empty(3, dtype=np.float64)  # Same size as sender's array
    # Use ANY_SOURCE and ANY_TAG to receive from any rank/tag
    status = MPI.Status()
    comm.Recv(data, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    print(f"[Rank 1] Received data: {data}")
    print(f"[Rank 1] Message received from rank {status.Get_source()} with tag {status.Get_tag()}")
else:
    pass
