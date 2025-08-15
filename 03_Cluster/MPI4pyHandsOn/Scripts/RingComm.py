'''
This python program sends data (a numpy array) in a ring topology using mpi4py.
The main purpose is to demonstrate the effect of different send/receive modes,
especially blocking versus non-blocking. We need at least two ranks, i.e.
$ mpiexec -n 2 python ringComm.py
'''

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendFunction = comm.Send #default send, blocking (returns after application buffer can be reused[*])
recvFunction = comm.Recv #default recv, blocking (returns after application buffer is filled)
#sendFunction = comm.Ssend #Synchronous send (returns after connection with reciever), deadlock [**]
#sendFunction = comm.Isend #Non-blocking send, recommended to avoid deadlock
#recvFunction = comm.Irecv #Non-blocking recv (not needed, if used, add a MPI.wait to avoid complaint,
#like: req = comm.Irecv(...), data2beRecv = req.wait() )


arraySize = 1

#Create numpy array with data unique to this rank/Process
data2beSent = np.ones(arraySize, dtype='i')*rank

#Allocate buffer for data to be received, should match with what is sent
data2beRecv = np.empty(arraySize, dtype='i')

#Define source and destination for this rank (nearest neighbours)
leftNeighbor = (rank -1 + size) % size
rightNeighbor = (rank + 1) % size

#Send
sendFunction([data2beSent, MPI.INT], dest=rightNeighbor, tag=0)
#Recv
recvFunction([data2beRecv, MPI.INT], source=leftNeighbor, tag=0)

#Send-And-Recv (Should always work, communication patterns hidden, so not suited for demonstration purpose)
#comm.Sendrecv(sendbuf=data2beSent, dest=rightNeighbor, sendtag=0,
#              recvbuf=data2beRecv, source=leftNeighbor,recvtag=0)

#Check if algorithm output matches our expectations
print(f"Process {rank} received data {data2beRecv} from process {leftNeighbor}")

'''
[*] Might deadlock for large numpy arrays (if MPI library decides that reallocating data
in library space is inefficient (i.e. copying data from application space to library space -> memory is doubled)
MPI might do a comm.Ssend under the hood, then the receiver must be ready at the time of sending.
[**]If you hear the fan kicking in: this indicates 'busy-waiting' (worst case scenario)
'''

