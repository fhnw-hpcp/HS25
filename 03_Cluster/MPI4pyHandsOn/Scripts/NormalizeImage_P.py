from mpi4py import MPI
import numpy as np
import scipy.datasets
import matplotlib.pyplot as plt

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

detectorImageSliceNormalized = detectorImageSlice / 2**bitDepth

if rank == 0:
    detectorImageNormalized = np.empty((numberOfRows,numberOfColumns) , dtype=np.float64)
else:
    detectorImageNormalized = None

comm.Gather(detectorImageSliceNormalized, detectorImageNormalized, root=0)

if rank == 0:
    print ("Parallel computation equals sequential :", np.allclose(detectorImage/2**bitDepth, detectorImageNormalized))


