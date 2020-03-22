import cv2
import imutils
import numpy as np


img = cv2.imread('./Images/CNN_Output/img_segment_by_cnn.jpg', cv2.IMREAD_GRAYSCALE)
img_original = cv2.imread('./Images/Input/img_original.jpg', cv2.IMREAD_GRAYSCALE)
img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2RGB)
# threshold value
thresh = 200

# binary threshold
thresh_value, img_binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
cnts = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# look over the contours to find max and draw the suspicious area by green
c1_max = max(cnts, key=cv2.contourArea)

# find the bounding rectangles
rect = cv2.boundingRect(c1_max)
x, y, w, h = rect

# draw the rectangle
cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 255, 0), 2)
# put text with it
cv2.putText(img_original, 'Tumor Region', (x+w+10, y+h), 0,0.3, (0, 255, 0))





cv2.imshow('a',img_original)
cv2.waitKey(0)
