import numpy as np

def euclidian_distance(point_one: list, point_two: list) -> float:
    """
    Calculates the euclidian distance between two points

    Args:
        point_one (list): The first point of the calculation
        point_two (list): The second point of the calculation
    Returns:
        (float): The euclidian distance between the two points
    """

    distance = np.sqrt(((point_two[0]-point_one[0])**2) + (point_two[1]-point_one[1])**2)
    
    return distance


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
            original_column = column * old_h / new_height

            # Then calculate the four points to be used to interpolate
            line_floor = np.floor(original_line)
            line_ceil = min(old_h - 1, np.ceil(original_line))
            column_floor = np.floor(original_column)
            column_ceil = min(old_w - 1, np.ceil(original_column))
        
            # Now we must discover which of the points is closer to the point we want to calculate
            minimum = np.inf
            minimum_line = 0
            minimum_column = 0

            for l_val in [line_floor, line_ceil]:
                for c_val in [column_floor, column_ceil]:
                    distance = euclidian_distance([line, column], [l_val, c_val])
                    if distance < minimum: 
                        minimum_line = l_val; minimum_column = c_val

            # Knowing now which of the points is closest we can copy it's value to the pixel
            pixel = img[int(minimum_line), int(minimum_column)]
            resized_img[line, column] = pixel

    return resized_img.astype(np.uint8)



