import cv2
import imutils
import numpy as np

from Classification import get_rf_result
from Cnn_Yes_No import get_yes_no


def main():
	from keras_segmentation.predict import predict

	# Read original image from Input folder
	img_original = cv2.imread('./Images/Input/img_original.jpg')

	# Prediction method
	predict(
		# path to trained model
		checkpoints_path="./Model_CNN_Segmentation/model",
		# input image path from images/skull striping
		inp="./Images/Input/img_original.jpg",
		# output image path from  ./Images/CNN_Output
		out_fname="./Images/CNN_Output/img_segment_by_cnn.jpg"
	)

	try:
		# Read segmented image from output folder
		# check weather yes image or no
		result, features = get_rf_result()
		# if no use tha blank mask
		if result == 0:
			img = cv2.imread('./Images/CNN_Output/blank_mask.jpg', cv2.IMREAD_GRAYSCALE)
		else:
			img = cv2.imread('./Images/CNN_Output/img_segment_by_cnn.jpg', cv2.IMREAD_GRAYSCALE)
		# given threshold to get tumor mask
		thresh = 200

		thresh_value, img_binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

		# contour for suspicious point
		cnts = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		# look over the contours to find max and draw the suspicious area by green
		c1_max = max(cnts, key=cv2.contourArea)
		cv2.drawContours(img_original, [c1_max], -1, (0, 255, 0), 2)
		# writing tumor marked image into ./Images/CNN_Output/img_tumor_marked.jpg
		cv2.imwrite('./Images/CNN_Output/img_tumor_marked.jpg', img_original)
	except:
		cv2.imwrite('./Images/CNN_Output/img_tumor_marked.jpg', img_original)
		print("An exception occurred")




