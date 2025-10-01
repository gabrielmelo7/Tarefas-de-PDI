import numpy as np

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




