import cv2
import imutils
import numpy as np
import os
from networkx.drawing.tests.test_pylab import plt


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


def resizing(img):
    W=500
    # Image resizing
    height, width = img.shape
    imgScale = W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv2.resize(img, (int(newX), int(newY)))
    return img

def skull_striping(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binary using Otsu's method
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    write_image('img_otsu', thresh, 'gray')
    colormask = np.zeros(img.shape, dtype=np.uint8)
    colormask[thresh != 0] = np.array((0, 0, 255))
    blended = cv2.addWeighted(img, 0.7, colormask, 0.1, 0)
    write_image('img_blended', blended, 'bgr')
    ret, markers = cv2.connectedComponents(thresh)

    # Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]
    # Get label of largest component by area
    largest_component = np.argmax(marker_area) + 1  # Add 1 since we dropped zero above
    # Get pixels which correspond to the brain
    brain_mask = markers == largest_component

    brain_out = img.copy()
    # In a copy of the original image, clear those pixels that don't correspond to the brain
    brain_out[brain_mask == False] = (0, 0, 0)
    write_image('img_connected_components', brain_out, 'rgb')

    brain_mask = np.uint8(brain_mask)
    kernel = np.ones((8, 8), np.uint8)
    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    write_image('img_closing', closing, 'gray')

    return brain_out




