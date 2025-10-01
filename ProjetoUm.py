# Importing necessary libraries
import numpy as np
import cv2

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
# The task at hand is to reduce the image in a factor of ten. This can be done with bilinear interpolation
def bilinear_resizing(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Resizes an image with bilinear interpolation according to a scale.

    Args:
        img (np.ndarray): The matrix of the image to be resized
        height(int): The height of the new matrix
        width(int): The width of the new matrix
    Returns:
        np.ndarray: The resized image matrix
    """

    # We begin by getting the old dimensions and calculating the scale
    old_h, old_w = img.shape

    # Now we generate a matrix with the new dimensions
    new_img = np.zeros((height, width))

    # And then we estimate the values for each of the pixels
    for line in range(height):
        for column in range(width):
            # First we map the coordinates back to the original matrix
            original_line = line * old_h / height 
            original_column = column * old_w / width 

            # Then calculate the four points to be used to interpolate
            line_floor = np.floor(original_line)
            line_ceil = min( old_h - 1, np.ceil(original_line))
            column_floor = np.floor(original_column)
            column_ceil = min( old_w - 1, np.ceil(original_column))

            # The four points can be
            # 1: One of the original points
            if(line_floor == line_ceil and column_floor == column_ceil):
                pixel = img[int(original_line), int(original_column)]

            # 2: On the same line
            elif(line_floor == line_ceil):
                pixel_floor = img[int(original_line), int(column_floor)]
                pixel_ceil = img[int(original_line), int(column_ceil)]

                pixel = pixel_floor * (column_ceil - original_column) + pixel_ceil * (original_column - column_floor)

            # 3: On the same column
            elif(column_floor == column_ceil):
                pixel_floor = img[int(line_floor), int(original_column)]
                pixel_ceil = img[int(line_ceil), int(original_column)]

                pixel = pixel_floor * (line_ceil - original_line) + pixel_ceil * (original_line - line_floor)

            # 4: The general interpolation case
            else:
                pixel_1 = img[int(line_floor), int(column_floor)]
                pixel_2 = img[int(line_ceil), int(column_floor)]
                pixel_3 = img[int(line_floor), int(column_ceil)]
                pixel_4 = img[int(line_ceil), int(column_ceil)]

                pixel_floor= pixel_1 * (line_ceil - original_line) + pixel_2 * (original_line - line_floor)
                pixel_ceil= pixel_3 * (line_ceil - original_line ) + pixel_4 * (original_line - line_floor)
                
                pixel = pixel_floor * (column_ceil - original_column) + pixel_ceil * (original_column - column_floor)
            
            new_img[line, column] = pixel

    return new_img.astype(np.uint8)
        

# First, we calculate the new dimensions for the image simply by dividing the current ones by 10
new_height = int(np.floor(IMG.shape[0] / 10))
new_width = int(np.floor(IMG.shape[1] / 10))

# Now it is possible to downsize the image ten times
reduced_IMG = bilinear_resizing(IMG, new_height, new_width)
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
og_size_IMG = bilinear_resizing(reduced_IMG, 256, 256)
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


