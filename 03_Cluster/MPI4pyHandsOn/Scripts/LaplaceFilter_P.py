from mpi4py import MPI
import numpy as np
from scipy.ndimage import laplace

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert size == 2

nr = 220
nc = 200
nh = nr//2
bitDepth = 8

np.random.seed(21) #Commen this line, unit test fails
detectorImage = np.random.randint(2**bitDepth, size=(nr, nc)).astype(np.float64)


if rank == 0:
    #Rank zero allocates filtered array of same size as initial size since it will receive lower half
    detectorImageFiltered = np.zeros_like(detectorImage)
    #Only calculate the upper half part but make sure that input contains a additional row
    #at the end of the array, this last row is skipped in the filerted (partial) result
    #In this case the cval only affects the outer border of the 2d array, not the inner
    detectorImageFiltered[:nh, :] = laplace(detectorImage[:nh +1, :], mode='constant', cval=0)[:-1, :]
    comm.Recv(detectorImageFiltered[nh:, :], source=1, tag=12)
    #Unit Test
    detectorImageFiltered_ref = laplace(detectorImage, mode='constant', cval=0)
    print ("Parallel computation equals sequential :", np.allclose(detectorImageFiltered_ref, detectorImageFiltered))
else: #<- only since we know that we are running on 2 cores
    #Only calculate the lower half part but make sure that input contains a additional row
    #at the start of the array, this first row is skipped in the filtered (partial) result
    #In this case the cval only affects the outer border of the 2d array, not the inner
    detectorImageFiltered = laplace(detectorImage[nh -1:, :], mode='constant', cval=0)[1:, :]
    comm.Send(detectorImageFiltered, dest=0, tag=12)

