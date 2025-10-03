import numpy as np

def nearest_neighbour_resize(img :np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Resizes an image using the nearest neighbour method.

        Args:
            img (np.ndarray): The matrix form of the new image
            new_height (int): The new height of the matrix
            new_width(int): The new width of the matrix
        Returns:
            np.ndarray: The resized image matrix
    """

    # We begin by getting the old dimensions of the matrix
    old_h, old_w = img.shape

    # Now, we generate a matrix with the new dimensions
    resized_img = np.zeros((new_height, new_width))

    # Now we can estimate the value for each of the pixels
    for line in range(new_height):
        for column in range(new_width):
            # First we map the current pixel coordenates to the original matrix
            original_line = line * old_h / new_height
            original_column = column * old_w / new_width

            # Then calculate the closest point to replicate 
            nearest_line = min(old_h-1, int(round(original_line)))
            nearest_column = min(old_w-1, int(round(original_column)))
            
            # Knowing now which of the points is closest we can copy it's value to the pixel
            pixel = img[int(nearest_line), int(nearest_column)]
            resized_img[line, column] = pixel

    return resized_img.astype(np.uint8)



