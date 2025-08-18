from mpi4py import MPI
import numpy as np

from scipy.ndimage import laplace

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

numberOfRows = 220
numberOfColumns = 200
bitDepth = 8

if rank == 0:
    detectorImage = np.random.randint(2**bitDepth, size=(numberOfRows,numberOfColumns)).astype(np.float64)
else:
    detectorImage = None

numberOfRowsPerCore = numberOfRows//size

detectorImageSlice = np.empty((numberOfRowsPerCore, numberOfColumns), dtype=np.float64)

comm.Scatter(detectorImage, detectorImageSlice, root=0)

detectorImageSliceFiltered = laplace(detectorImageSlice, mode='constant', cval=0)

if rank == 0:
    detectorImageFiltered = np.empty((numberOfRows,numberOfColumns) , dtype=np.float64)
else:
    detectorImageFiltered = None

comm.Gather(detectorImageSliceFiltered, detectorImageFiltered, root=0)

if rank == 0:
    print ("Parallel computation equals sequential :", np.allclose(laplace(detectorImage, mode='constant', cval=0), detectorImageFiltered))


