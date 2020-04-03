import cv2
import imutils
import numpy as np


def main():
    image_skull_removed = cv2.imread('./Images/CNN_Skull_Remove_Output/img_skull_removed_by_cnn.jpg', cv2.IMREAD_GRAYSCALE)

    # adding Gaussian blur
    img_blr = cv2.GaussianBlur(image_skull_removed, (5,5), 0)
    # enhancement
    img_blr = cv2.convertScaleAbs(img_blr, alpha=1.3, beta=.8)

    # declaring array
    arr = []
    # making fix array of 255
    for i in range(256):
        arr.append(0)

    rows, cols = image_skull_removed.shape
    total_pixel = rows * cols
    # calculate in intensities and assign to the array
    for j in range(cols - 1):
        for i in range(rows - 1):
            arr[image_skull_removed[i][j]] = arr[image_skull_removed[i][j]] + 1





    def cal_thresh_sus1():
        # Calculating threshold
        sum = 0

        for i in range(256):
            index = 255 - i
            sum += arr[index]
            if sum > (total_pixel / 5):
                break
        return index


    def cal_thresh_sus2():
        sum = 0

        for i in range(256):
            index = 255 - i
            sum += arr[index]
            if sum > (total_pixel / 10):
                break
        return index



    # finding contours
    def find_contour(img_original, img):
        img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2RGB)

        # finding suspicious area
        thresh_1 = cal_thresh_sus1()
        thresh_value, img_binary_sus1 = cv2.threshold(img, thresh_1, 255, cv2.THRESH_BINARY)


        # finding suspicious  point
        thresh_2 = cal_thresh_sus2()
        thresh_value, img_binary_sus2 = cv2.threshold(img, thresh_2, 255, cv2.THRESH_BINARY)


        # contour for suspicious area
        cnts_for_sus_1 = cv2.findContours(img_binary_sus1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_for_sus_1 = imutils.grab_contours(cnts_for_sus_1)

        # contour for suspicious point
        cnts_for_sus_2 = cv2.findContours(img_binary_sus2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_for_sus_2 = imutils.grab_contours(cnts_for_sus_2)

        # look over the contours to find max and draw the suspicious area by green
        c1_max = max(cnts_for_sus_1, key=cv2.contourArea)
        cv2.drawContours(img_original, [c1_max], -1, (0, 255, 0), 2)
        # writing image to folder
        cv2.imwrite('./Images/Knowledgebase_Output/kb_img_tumor_marked_1.jpg', img_original)
        # look over the contours to find max and draw the suspicious area by red
        c2_max = max(cnts_for_sus_2, key=cv2.contourArea)
        cv2.drawContours(img_original, [c2_max], -1, (0, 0, 255), 2)
        # writing image to folder
        cv2.imwrite('./Images/Knowledgebase_Output/kb_img_tumor_marked_2.jpg', img_original)




    def partial_cropeed_thresholding():
        img = cv2.imread('./Images/CNN_Output/img_segment_by_cnn.jpg', cv2.IMREAD_GRAYSCALE)
        img_original = cv2.imread('./Images/Input/img_original.jpg', cv2.IMREAD_GRAYSCALE)
        # img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2RGB)
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
        #cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # put text with it
        #cv2.putText(img_original, 'Tumor Region', (x + w + 10, y + h), 0, 0.3, (0, 255, 0))

        # Image cropping and masking
        white_bg = np.ones_like(img_original)

        # expanding rectangle
        x, y, w, h = x-20, y-20, w+20, h+20

        crop_img = img_original[y:y + h, x:x + w]
        white_bg[y:y + h, x:x + w] = crop_img

        # calling contour method
        find_contour(img_original, white_bg)

    partial_cropeed_thresholding()
