import numpy as np
import scipy as sp
import scipy.signal
import cv2

try:
  from cv2 import ORB as SIFT
except ImportError:
  try:
    from cv2 import SIFT
  except ImportError:
    try:
      SIFT = cv2.ORB_create
    except:
      raise AttributeError("Version of OpenCV(%s) does not have SIFT / ORB."
                      % cv2.__version__)

def findMatchesBetweenImages(image_1, image_2):
  """ Return the top 10 list of matches between two input images.

  This function detects and computes SIFT (or ORB) from the input images, and
  returns the best matches using the normalized Hamming Distance.

  Args:
    image_1 (numpy.ndarray): The first image (grayscale).
    image_2 (numpy.ndarray): The second image. (grayscale).

  Returns:
    image_1_kp (list): The image_1 keypoints, the elements are of type
                       cv2.KeyPoint.
    image_2_kp (list): The image_2 keypoints, the elements are of type 
                       cv2.KeyPoint.
    matches (list): A list of matches, length 10. Each item in the list is of
                    type cv2.DMatch.

  """
  # matches - type: list of cv2.DMath
  matches = None
  # image_1_kp - type: list of cv2.KeyPoint items.
  image_1_kp = None
  # image_1_desc - type: numpy.ndarray of numpy.uint8 values.
  image_1_desc = None
  # image_2_kp - type: list of cv2.KeyPoint items.
  image_2_kp = None
  # image_2_desc - type: numpy.ndarray of numpy.uint8 values.
  image_2_desc = None


  orb = cv2.ORB()
  image_1_kp = orb.detect(image_1, None)
  image_1_kp, image_1_desc = orb.compute(image_1, image_1_kp)
  
  image_2_kp = orb.detect(image_2, None)
  image_2_kp, image_2_desc = orb.compute(image_2, image_2_kp)

  bfMatch = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bfMatch.match(image_1_desc, image_2_desc)
  matches = sorted(matches, key = lambda x:x.distance)
  matches = matches[:10]
  image = drawMatches(image_1, image_1_kp, image_2, image_2_kp, matches)

  return image_1_kp, image_2_kp, matches


def drawMatches(image_1, image_1_keypoints, image_2, image_2_keypoints, matches):
  """ Draws the matches between the image_1 and image_2.

  Args:
    image_1 (numpy.ndarray): The first image (can be color or grayscale).
    image_1_keypoints (list): The image_1 keypoints, the elements are of type
                              cv2.KeyPoint.
    image_2 (numpy.ndarray): The image to search in (can be color or grayscale).
    image_2_keypoints (list): The image_2 keypoints, the elements are of type
                              cv2.KeyPoint.

  Returns:
    output (numpy.ndarray): An output image that draws lines from the input
                            image to the output image based on where the
                            matching features are.
  """
  # Compute number of channels.
  num_channels = 1
  if len(image_1.shape) == 3:
    num_channels = image_1.shape[2]
  # Separation between images.
  margin = 10
  # Create an array that will fit both images (with a margin of 10 to separate
  # the two images)
  joined_image = np.zeros((max(image_1.shape[0], image_2.shape[0]),
                           image_1.shape[1] + image_2.shape[1] + margin,
                           3))
  if num_channels == 1:
    for channel_idx in range(3):
      joined_image[:image_1.shape[0],
                   :image_1.shape[1],
                   channel_idx] = image_1
      joined_image[:image_2.shape[0],
                   image_1.shape[1] + margin:,
                   channel_idx] = image_2
  else:
    joined_image[:image_1.shape[0], :image_1.shape[1]] = image_1
    joined_image[:image_2.shape[0], image_1.shape[1] + margin:] = image_2

  for match in matches:
    image_1_point = (int(image_1_keypoints[match.queryIdx].pt[0]),
                     int(image_1_keypoints[match.queryIdx].pt[1]))
    image_2_point = (int(image_2_keypoints[match.trainIdx].pt[0] + \
                         image_1.shape[1] + margin),
                   int(image_2_keypoints[match.trainIdx].pt[1]))

    cv2.circle(joined_image, image_1_point, 5, (0, 0, 255), thickness = -1)
    cv2.circle(joined_image, image_2_point, 5, (0, 255, 0), thickness = -1)
    cv2.line(joined_image, image_1_point, image_2_point, (255, 0, 0), \
             thickness = 3)
  return joined_image

# image_1 = cv2.imread("images/source/sample/image_1.jpg")
# image_2 = cv2.imread("images/source/sample/image_2.jpg")
# kp1, kp2, matches = findMatchesBetweenImages(image_1, image_2)
# output = drawMatches(image_1, kp1, image_2, kp2, matches)
# cv2.imwrite("output.png", output)