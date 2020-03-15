import cv2
import imutils

def main():
    image_skull_removed = cv2.imread('./Images/CNN_Skull_Remove_Output/img_skull_removed_by_cnn.jpg', cv2.IMREAD_GRAYSCALE)

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
            if sum > (total_pixel / 10):
                break
        return index


    def cal_thresh_sus2():
        sum = 0

        for i in range(256):
            index = 255 - i
            sum += arr[index]
            if sum > (total_pixel / 80):
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
        cv2.drawContours(img_original, [c1_max], -1, (0, 255, 0), 1)
        # writing image to folder
        cv2.imwrite('./Images/Knowledgebase_Output/kb_img_tumor_marked_1.jpg', img_original)
        # look over the contours to find max and draw the suspicious area by red
        c2_max = max(cnts_for_sus_2, key=cv2.contourArea)
        cv2.drawContours(img_original, [c2_max], -1, (0, 0, 255), 1)
        # writing image to folder
        cv2.imwrite('./Images/Knowledgebase_Output/kb_img_tumor_marked_2.jpg', img_original)

    find_contour(image_skull_removed, image_skull_removed)

