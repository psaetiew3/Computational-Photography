import numpy as np
import scipy as sp
import scipy.signal
import cv2

def generatingKernel(parameter):
  """ Return a 5x5 generating kernel based on an input parameter.

  Args:
    parameter (float): Range of value: [0, 1].

  Returns:
    numpy.ndarray: A 5x5 kernel.

  """
  kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                     0.25, 0.25 - parameter /2.0])
  return np.outer(kernel, kernel)

def reduce(image):
  """ Convolve the input image with a generating kernel of parameter of 0.4 and
  then reduce its width and height by two.

  Args:
    image (numpy.ndarray): a grayscale image of shape (r, c)

  Returns:
    output (numpy.ndarray): an image of shape (ceil(r/2), ceil(c/2))
      For instance, if the input is 5x7, the output will be 3x4.

  """
  ker = generatingKernel(.4)

  out = scipy.signal.convolve2d(image, ker, 'same')
  r = np.ceil(np.float(image.shape[0])/2)
  c = np.ceil(np.float(image.shape[1])/2)
  new = np.zeros((r, c))
  for row in range(len(new)):
    for col in range(len(new[row])):
      new[row][col] = out[row*2][col*2]
  return new

def expand(image):
  """ Expand the image to double the size and then convolve it with a generating
  kernel with a parameter of 0.4.
  
  Args:
    image (numpy.ndarray): a grayscale image of shape (r, c)

  Returns:
    output (numpy.ndarray): an image of shape (2*r, 2*c)
  """

  r = np.ceil(np.double(image.shape[0])*2)
  c = np.ceil(np.double(image.shape[1])*2)
  out = np.zeros((r, c))
  out[::2, ::2] = image
  ker = generatingKernel(.4)
  out = scipy.signal.convolve2d(out, ker, 'same') * 4   
  return out


def gaussPyramid(image, levels):
  """ Construct a pyramid from the image by reducing it by the number of levels
  passed in by the input.

  Args:
    image (numpy.ndarray): A grayscale image of dimension (r,c) and dtype float.
    levels (uint8): A positive integer that specifies the number of reductions
                    you should do.

  Returns:
    output (list): A list of arrays of dtype np.float. The first element of the
                   list (output[0]) is layer 0 of the pyramid (the image
                   itself). output[1] is layer 1 of the pyramid (image reduced
                   once), etc.

  """
  output = [image]
  
  for lv in range(levels):
    image = reduce(image)
    output.append(image)
  return output
  
def laplPyramid(gaussPyr):
  """ Construct a laplacian pyramid from the gaussian pyramid, of height levels.

  Args:
    gaussPyr (list): A Gaussian Pyramid. It is a list of numpy.ndarray items.

  Returns:
    output (list): A laplacian pyramid of the same size as gaussPyr.

           output[k] = gauss_pyr[k] - expand(gauss_pyr[k + 1])

           Note: The last element of output should be identical to the last 
           layer of the input pyramid since it cannot be subtracted anymore.

  For example, if my layer is of size 5x7, reducing and expanding will result
  in an image of size 6x8. In this case, crop the expanded layer to 5x7.
  """
  output = []

  for i in range(len(gaussPyr) - 1):    
    new = expand(gaussPyr[i+1])
    if gaussPyr[i].shape != new.shape:
      new = new[:gaussPyr[i].shape[0], :gaussPyr[i].shape[1]]
    output.append(gaussPyr[i] - new)
  output.append(gaussPyr[len(gaussPyr) - 1])
  return output

def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
  """ Blend the two laplacian pyramids by weighting them according to the
  gaussian mask.

  Args:
    laplPyrWhite (list): A laplacian pyramid of one image, as constructed by
                         your laplPyramid function.

    laplPyrBlack (list): A laplacian pyramid of another image, as constructed by
                         your laplPyramid function.

    gaussPyrMask (list): A gaussian pyramid of the mask. Each value is in the
                         range of [0, 1].

  The pyramids will have the same number of levels. Furthermore, each layer
  is guaranteed to have the same shape as previous levels.

  """ 

  blended_pyr = []

  for i in range(len(laplPyrWhite)):
    blended_pyr.append(gaussPyrMask[i] * laplPyrWhite[i] + (1 - gaussPyrMask[i]) * laplPyrBlack[i])
  return blended_pyr
  
def collapse(pyramid):
  """ Collapse an input pyramid.

  Args:
    pyramid (list): A list of numpy.ndarray images. You can assume the input is
                  taken from blend() or laplPyramid().

  Returns:
    output(numpy.ndarray): An image of the same shape as the base layer of the
                           pyramid and dtype float.
                           
  For example, expanding a layer of size 3x4 will result in an image of size
  6x8. If the next layer is of size 5x7, crop the expanded image to size 5x7.
  """

  out = pyramid[len(pyramid) - 1]
  for i in reversed(range(len(pyramid) - 1)):
    new = expand(out)
    if pyramid[i].shape != new.shape:
      new = new[:pyramid[i].shape[0], :pyramid[i].shape[1]] 
    out = new + pyramid[i]
  return out
