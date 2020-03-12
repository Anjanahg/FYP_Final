import cv2
import imutils
import numpy as np

def main():
	from keras_segmentation.predict import predict

	# Read original image from Input folder
	img_original = cv2.imread('./Images/Input/img_original.jpg')

	# Prediction method
	predict(
		# path to trained model
		checkpoints_path="./Model_CNN_Skull_Remove/model",
		# input image path from images/original
		inp="./Images/Input/img_original.jpg",
		# output image path from  ./Images/CNN_Output
		out_fname="./Images/CNN_Skull_Remove_Output/img_skull_removed_by_cnn_mask.jpg"
	)

	try:
		# threshold value
		thresh = 200
		# read image
		img = cv2.imread('./Images/CNN_Skull_Remove_Output/img_skull_removed_by_cnn_mask.jpg', cv2.IMREAD_GRAYSCALE)
		# binary threshold
		thresh_value, img_binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
		cnts = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		# look over the contours to find max and draw the suspicious area by green
		c1_max = max(cnts, key=cv2.contourArea)
        # convert to array
		pts = np.array(c1_max, dtype=np.int32)
		# find the bounding rectangles
		rect = cv2.boundingRect(pts)
		x, y, w, h = rect
		cropped = img_original[y:y + h, x:x + w].copy()

		pts = pts - pts.min(axis=0)

		mask = np.zeros(cropped.shape[:2], np.uint8)
		cv2.drawContours(mask, [pts], -1, 255, -1, cv2.LINE_AA)
		# do bit-op
		out = cv2.bitwise_and(cropped, cropped, mask=mask)
		# writing tumor marked image into ./Images/CNN_Output/img_tumor_marked.jpg
		cv2.imwrite('./Images/CNN_Skull_Remove_Output/img_skull_removed_by_cnn.jpg', out)
	except:
		print("An exception occurred")


