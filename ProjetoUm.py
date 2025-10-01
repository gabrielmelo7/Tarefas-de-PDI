# Importing necessary libraries
import numpy as np
import cv2
from resizing_methods import BilinearInterpolation, NearestNeighbour

# Reading and opening the image
IMG_BGR = cv2.imread("Fig0222(b)(cameraman).tif")
cv2.imshow('Cameraman', IMG_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()

# By default OpenCV reads all images in the BGR format, therefore it is necessary that we convert the image to grayscale
IMG = cv2.cvtColor(IMG_BGR, cv2.COLOR_BGR2GRAY)

# OpenCV also opens images in a numpy array format, so we can treat the image as a matrix
# Important informations of the image
print("For the original image:")
print("Image dimensions: ", IMG.shape)
print("Number of pixels in image: ", IMG.size)
print("Type of image: ", IMG.dtype)
print("Max value in the image: ", IMG.max())
print("Min value in the image: ", IMG.min())

# We see that we have a 256x256 image with 65536 pixels.
# The task at hand is to reduce the image in a factor of ten. This can be done with the nearest neighbour method 
# First, we calculate the new dimensions for the image simply by dividing the current ones by 10
new_height = int(np.ceil(IMG.shape[0] / 10))
new_width = int(np.ceil(IMG.shape[1] / 10))

# Now it is possible to downsize the image ten times
reduced_IMG = NearestNeighbour.nearest_neighbour_resize(IMG, new_height, new_width)
print(type(reduced_IMG))

cv2.imshow("Downsized image", reduced_IMG)
cv2.waitKey(0)
cv2.destroyAllWindows()

# The image has noticeable distortion. This is due to the drastic downsize in resolution
# We can take a look at its new information and compare them to the original ones
print("\nFor the reduced image:")
print("Image dimensions: ", reduced_IMG.shape)
print("Number of pixels in image: ", reduced_IMG.size)
print("Type of image: ", reduced_IMG.dtype)
print("Max value in the image: ", reduced_IMG.max())
print("Min value in the image: ", reduced_IMG.min())

# It is, now, necessary that we change the image back to its original resolution 
og_size_IMG = NearestNeighbour.nearest_neighbour_resize(reduced_IMG, 256, 256)
cv2.imshow("Image with the original size", og_size_IMG)
cv2.waitKey(0)
cv2.destroyAllWindows()

# We can see that even when we return to the original resolution the image is still very distorted,
#this happens because in the downsizing process the image lost a lot of its original information
#and when we go back to the original size we interpolate on this incomplete information.
# We can take a look at its new information and compare them to the original ones
print("\nFor the upsized image:")
print("Image dimensions: ", og_size_IMG.shape)
print("Number of pixels in image: ", og_size_IMG.size)
print("Type of image: ", og_size_IMG.dtype)
print("Max value in the image: ", og_size_IMG.max())
print("Min value in the image: ", og_size_IMG.min())

# To finish things off we write and save all of these images
cv2.imwrite("OriginalImage.png", IMG)
cv2.imwrite("DownsizedImage.png", reduced_IMG)
cv2.imwrite("UpsizedImage.png", og_size_IMG)


