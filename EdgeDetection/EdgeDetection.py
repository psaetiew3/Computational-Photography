import cv2
import numpy as np
import scipy as sp

def imageGradientX(image):
    """ This function differentiates an image in the X direction.
    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The image gradient in the X direction.
    """    
    output_image = np.zeros(image.shape)

    for row in range(len(image)-1):
        for col in range(len(image[row])-1):
            output_image[row][col] = abs(np.float64(image[row+1][col]) - np.float64(image[row][col]))
    #cv2.imwrite("test_image_X.jpg", output_image)
    return output_image

def imageGradientY(image):
    """ This function differentiates an image in the Y direction.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The image gradient in the Y direction.
    """

    output_image = np.zeros(image.shape)

    for row in range(len(image)-1):
        for col in range(len(image[row])-1):
            output_image[row][col] = abs(np.float64(image[row][col+1]) - np.float64(image[row][col]))
    #cv2.imwrite("test_image_Y.jpg", output_image)
    return output_image

def computeGradient(image, kernel):
    """ This function applies an input 3x3 kernel to the image, and outputs the
    result.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        output (numpy.ndarray): The computed gradient for the input image.
    """
    
    output_image = np.zeros(image.shape)

    temp3 = np.float64(0)
    
    for row in range(len(image)-2):
        for col in range(len(image[row])-2):
            for rowK in range(0, 3):
                for colK in range(0,3):
                    temp3 = temp3 + (np.float64(image[row+rowK][col+colK]) * np.float64(kernel[rowK][colK]))
            output_image[row+1][col+1] = temp3
            temp3 = np.float64(0)
    #cv2.imwrite("test_image_Gradient.jpg", output_image)
    return output_image
