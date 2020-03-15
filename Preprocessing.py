import cv2
import imutils
import numpy as np
import os
from networkx.drawing.tests.test_pylab import plt

# image writing method
def write_image(title,img,ctype):
  plt.figure(figsize=(10, 10))
  if ctype=='bgr':
    b,g,r = cv2.split(img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    cv2.imwrite('./Images/SkullStriping/'+title+'.jpg', img)
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    cv2.imwrite('./Images/SkullStriping/'+title+'.jpg', img)
  elif ctype=='gray':
    cv2.imwrite('./Images/SkullStriping/'+title+'.jpg', img)
  elif ctype=='rgb':
    cv2.imwrite('./Images/SkullStriping/'+title+'.jpg', img)
  else:
    raise Exception("Unknown colour type")

# Resizing method
def resizing(img):
    W=500
    # Image resizing
    height, width = img.shape
    imgScale = W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv2.resize(img, (int(newX), int(newY)))
    return img
# skull striping
def skull_striping(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # writing binary image by Otsu
    write_image('img_otsu', thresh, 'gray')
    ret, markers = cv2.connectedComponents(thresh)
    # blending
    colormask = np.zeros(img.shape, dtype=np.uint8)
    colormask[thresh != 0] = np.array((0, 0, 255))
    blended = cv2.addWeighted(img, 0.7, colormask, 0.1, 0)
    # writing blended image
    write_image('img_blended', blended, 'bgr')

    # Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]
    # Get label of largest component by area
    largest_component = np.argmax(marker_area) + 1  # Add 1 since we dropped zero above
    # Get pixels which correspond to the brain
    brain_mask = markers == largest_component

    brain_out = img.copy()
    # In a copy of the original image, clear those pixels that don't correspond to the brain
    brain_out[brain_mask == False] = (0, 0, 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    # writing watershed
    write_image('img_watershed', img, 'rgb')

    return brain_out




