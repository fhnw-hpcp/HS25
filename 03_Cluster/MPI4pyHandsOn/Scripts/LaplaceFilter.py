import numpy as np
from scipy.ndimage import laplace

import matplotlib.pyplot as plt

bitDepth = 8
detectorImage = np.random.randint(2**bitDepth, size=(220, 200)).astype(np.float64)
filteredImage = laplace(detectorImage, mode='constant', cval=0)
plt.imshow(detectorImage); plt.colorbar()
plt.show()
plt.imshow(filteredImage); plt.colorbar()
plt.show()




