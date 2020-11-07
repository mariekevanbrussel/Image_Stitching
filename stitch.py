import cv2
import numpy as np
from matplotlib import pyplot as plt
#Local import
from keypoint_matching import keypoint_matching
from RANSAC import ransac, map_img

def interpolate(img):
  for i in range(img.shape[0] - 1):
    for j in range(img.shape[1] - 1):
      if (np.array_equal(img[i, j], np.array([0, 0, 0]))):
        try:
          avg = np.mean([
              img[i - 1, j - 1, :], img[i + 1, j + 1, :], img[i + 1, j - 1, :],
              img[i - 1, j + 1, :]
          ])
          img[i, j] = avg
        except:
          pass
  return img

def stitch(left, right, ratio=0.6):

  # Detect keypoints in images and match,
  matches, k1, k2, _, _ = keypoint_matching(right, left, ratio=ratio)

  # Use RANSAC to estimate the transformation matrix from the matches.
  x, _ = ransac(k1, k2, P=70, N=100, verbose=False)

  # Create 3x3 homogenous matrix from the parameters.
  H = np.array([[x[0], x[1], x[4]], [x[2], x[3], x[5]], [0, 0, 1]])

  # Calculate transformed coordinates of the corner of right.
  c1 = np.dot(H, [0, 0, 1])
  c2 = np.dot(H, [right.shape[0] - 1, 0, 1])
  c3 = np.dot(H, [right.shape[0] - 1, right.shape[1] - 1, 1])
  c4 = np.dot(H, [0, right.shape[1] - 2, 1])

  # Determing the size of the final output image.
  output_h = max(left.shape[0], right.shape[0])
  output_w = max(int(c1[0]), int(c2[0]), int(c3[0]), int(c4[0]))

  # Transform right image by applying best RANSAC match.
  tright = map_img(right, x)

  # Interpolate using nearest neighbor to handle holes.
  tright = interpolate(tright)

  # Create final black image.
  result = np.zeros((output_h, output_w, 3), dtype=left.dtype)

  # Combile the left and transformed right image.
  result[0:tright.shape[0], 0:tright.shape[1]] = tright
  result[0:left.shape[0], 0:left.shape[1]] = left

  return result


if __name__ == '__main__':

  left = cv2.imread('images/tram_left.jpg')
  right = cv2.imread('images/tram_right.jpg')

  result = stitch(left, right, ratio=0.6)

  plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
  plt.show()