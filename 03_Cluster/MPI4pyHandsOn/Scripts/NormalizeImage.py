import numpy as np
import scipy.datasets
import matplotlib.pyplot as plt

bitDepth = 8
detectorImage = np.random.randint(2**bitDepth, size=(220, 200))
detectorImageNormalized = detectorImage.astype(np.float64) / 2**bitDepth
plt.imshow(detectorImage); plt.colorbar()
plt.show()
plt.imshow(detectorImageNormalized); plt.colorbar()
plt.show()
