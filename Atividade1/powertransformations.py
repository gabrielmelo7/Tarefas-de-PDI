"""
Command-line arguments (sys.argv):

sys.argv[0] -> Script name (automatically provided by Python)
sys.argv[1] -> Path to the input image (string)
sys.argv[2] -> Base name for the output files (string)
sys.argv[3:] -> One or more gamma values to apply (float)

Example usage:
    python powertransformations.py "Fig0308(a)(fractured_spine) (1).tif" fracturedspine 0.6 0.4 0.3

This will generate the following output files:
    ./results/fracturedspine_g=0.6.png
    ./results/fracturedspine_g=0.4.png
    ./results/fracturedspine_g=0.3.png
"""

# Importing necessary libraries
import numpy as np
import cv2
import sys
import os

# Defining the power transformation function
def power_transform(img: np.ndarray, gamma: float, c: int) -> np.ndarray:
    """
    Adjusts the contrast of an image with a power transformation

    Args:
        img (np.ndarray): The matrix of the image to be adjusted
        gamma (float): The power to which the image will be adjusted
        c (int): A constant that will multiply the pixel value


    Returns: 
        np.ndarray: The adjusted image matrix
    """
    # First, the new image matrix is created
    new_img = np.zeros(img.shape)

    # The function to be applied is s = c(r^y)
    # It can be applied to every pixel in the following manner
    img_float = img.astype(np.float32) / 255.0

    new_img = c * np.power(img_float, gamma)

    new_img = np.clip(new_img * 255, 0, 255).astype(np.uint8)
    return new_img

# Reading and opening the image
img_path = sys.argv[1]
IMG_BGR = cv2.imread(img_path)
cv2.imshow('Fractured Spine', IMG_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()

# By default OpenCV reads images in the BGR format, therefore it is necessary to convert the image to grayscale
IMG_GRAY = cv2.cvtColor(IMG_BGR, cv2.COLOR_BGR2GRAY) 

# Defining the gamma values and the c value to transform the image
gamma_values = [float(x) for x in sys.argv[3:]] 
c_value = 1
adjusted_images = []

saving_name = sys.argv[2]
# Transforming the image to each gamma value
for gamma in gamma_values:
    new_img = power_transform(IMG_GRAY, gamma, c_value) 
    adjusted_images.append(new_img)

# Visualizing the results for each gamma value
for i in range(len(adjusted_images)):
    cv2.imshow(f'Fractured Spine with gamma = {gamma_values[i]}', adjusted_images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = os.path.join("./results", f"{saving_name}_g={gamma_values[i]}.png")
    cv2.imwrite(output_path, adjusted_images[i])

