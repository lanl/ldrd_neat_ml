import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skimage

image = skimage.io.imread("/Users/treddy/LANL/LDRD_DR_NEAT_data/Images/DEXTRAN (10k) 2~14wt_ (with PEO 10K)/DEXTRAN 2wt_/DEX2wt_,PEO2wt_.tiff")
image = skimage.util.img_as_ubyte(image)
median = np.median(image)
fig, ax = plt.subplots(3, 2, figsize=(8, 8))
ax = ax.ravel()
ax[0].imshow(image, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("Original")
# replace anything darker than the median
# with the median (attempt to remove the
# "dark spots" in the background)
new_image = image.copy()
new_image[new_image < (median + 15)] = median
ax[1].imshow(new_image, cmap="gray", vmin=0, vmax=255)
ax[1].set_title("Processed: darker than\nmedian (+15) = median")
ax[2].hist(image.ravel(), bins=200)
ax[2].set_xlim(100, 150)
ax[2].set_ylabel("Frequency")
ax[2].set_title("Histogram of Original Greyscale Vals\n(median in red)")
ax[2].set_xlabel("Pixel Value")
ax[2].set_ylim(0, image.size)
ax[2].axvline(median, c="red")
ax[3].hist(new_image.ravel(), bins=200)
ax[3].set_xlim(100, 150)
ax[3].set_ylim(0, image.size)
ax[3].set_ylabel("Frequency")
ax[3].set_title("Histogram of Processed Greyscale Vals\n(median in red)")
ax[3].set_xlabel("Pixel Value")
median = np.median(new_image)
ax[3].axvline(median, c="red")
whitened_image = image.copy()
whitened_image[whitened_image > np.median(whitened_image)] = 255
ax[4].imshow(whitened_image,
             cmap="gray",
             vmin=0, vmax=255)
ax[4].set_title("Original with whitening above median")
whitened_processed_image = new_image.copy()
whitened_processed_image[whitened_processed_image > np.median(whitened_processed_image)] = 255
ax[5].imshow(whitened_processed_image,
             cmap="gray",
             vmin=0, vmax=255)
ax[5].set_title("Processed with whitening above median")
fig.tight_layout()
fig.savefig("debug.png", dpi=300)
