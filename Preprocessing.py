import cv2
import imutils
import numpy as np


def resizing(img):
    W=500
    # Image resizing
    height, width = img.shape
    imgScale = W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv2.resize(img, (int(newX), int(newY)))
    return img

def applying_otsu(img):
    # Threshold the image to binary using Otsu's method
    t_val_otsu, img_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # writing the image to images/SkullStriping folder
    cv2.imwrite('./Images/SkullStriping/img_otsu.jpg', img_otsu)
    return img_otsu
def conected_component(img):

    ret, markers = cv2.connectedComponents(img)
    # Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]
    # Get label of largest component by area
    largest_component = np.argmax(marker_area) + 1  # Add 1 since we dropped zero above
    # Get pixels which correspond to the brain
    brain_mask = markers == largest_component

    brain_out = img.copy()
    # In a copy of the original image, clear those pixels that don't correspond to the brain
    brain_out[brain_mask == False] = (0)

    brain_mask = np.uint8(brain_mask)
    kernel = np.ones((8, 8), np.uint8)
    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    return brain_out
