import numpy as np
import skimage as ski
import matplotlib.pyplot as plt

img = np.load('train_tiles/tile_1.npy')  # Make sure it's `.npy`, not `.tif`
print("Shape of image:", img.shape)


plt.imshow(img[:, :, 0], cmap='gray')  # Show the first channel (or change to any valid channel index)
plt.title("Feature Channel 0")  # Optional title
plt.axis('off')  # Hide axes
plt.show()

