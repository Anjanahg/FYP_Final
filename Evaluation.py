
import os
import cv2
from xlwt import Workbook

ddd = Workbook()
sheet1 = ddd.add_sheet('Sheet 1')
sheet1.write(0, 0, 'Image Name')
sheet1.write(0, 1, 'errorrate')
sheet1.write(0, 2, 'accuracy')
sheet1.write(0, 3, 'recall')
sheet1.write(0, 4, 'specificity')
sheet1.write(0, 5, 'falsepositiverate')
sheet1.write(0, 6, 'precision')
sheet1.write(0, 7, 'f0_5')
sheet1.write(0, 8, 'f1')
sheet1.write(0, 9, 'f2')

folder_path = "C:/Users/DELL/Desktop/Evaluation/Original"
image_path = os.listdir(folder_path)

for n, image in enumerate(image_path):
    print(image_path[n].split('.')[0])
    imge_name = image_path[n].split('.')[0]

    label = cv2.imread(folder_path + '/' + imge_name + '.png', cv2.IMREAD_GRAYSCALE)

    model_out = cv2.imread('C:/Users/DELL/Desktop/Evaluation/Detected/' + imge_name + '.jpg', cv2.IMREAD_GRAYSCALE)
    thresh = 200
    thresh_value, model_out = cv2.threshold(model_out, thresh, 255, cv2.THRESH_BINARY)

    height = label.shape[0]
    width = label.shape[1]

    # plt.imshow(label)
    # plt.show()

    bb = 0
    bw = 0
    wb = 0
    ww = 0

    for j in range(height):
        for i in range(width):
            labelval = label[j, i]
            modelval = model_out[j, i]

            if ((labelval == 0) & (modelval == 0)):
                bb += 1

            if ((labelval == 0) & (modelval == 255)):
                bw += 1

            if ((labelval == 1) & (modelval == 255)):
                ww += 1
            if ((labelval == 1) & (modelval == 0)):
                wb += 1

    print("bb   :", bb)
    print("bw   :", bw)
    print("ww   :", ww)
    print("wb   :", wb)

    all = bb + bw + ww + wb

    errorrate = (bw + wb)/all
    errorrate = round(errorrate, 2)
    accuracy = (ww + bb)/all
    accuracy = round(accuracy, 2)
    recall = ww/(ww + wb)
    recall = round(recall, 2)
    specificity = bb/(bb + bw)
    specificity = round(specificity, 2)
    falsepositiverate = 1 - specificity
    falsepositiverate = round(falsepositiverate, 2)
    precision = ww/(ww + bw)
    precision = round(precision, 2)
    f0_5 = (1.25 * precision * recall)/(0.25 * precision + recall)
    f0_5 = round(f0_5, 2)
    f1 = (2 * precision * recall)/(precision + recall)
    f1 = round(f1, 2)
    f2 = (5 * precision * recall)/(4 * precision + recall)
    f2 = round(f2, 2)
    print("errorrate   :", errorrate)
    print("accuracy   :", accuracy)
    print("recall   :", recall)
    print("specificity   :", specificity)
    print("falsepositiverate   :", falsepositiverate)
    print("precision   :", precision)
    print("f0_5  :", f0_5)
    print("f1   :", f1)
    print("f2   :", f2)


    sheet1.write(n+1, 0, str(imge_name))
    sheet1.write(n+1, 1, str(errorrate))
    sheet1.write(n+1, 2, str(accuracy))
    sheet1.write(n+1, 3, str(recall))
    sheet1.write(n+1, 4, str(specificity))
    sheet1.write(n+1, 5, str(falsepositiverate))
    sheet1.write(n+1, 6, str(precision))
    sheet1.write(n+1, 7, str(f0_5))
    sheet1.write(n+1, 8, str(f1))
    sheet1.write(n+1, 9, str(f2))

ddd.save('./EvaluationResult.xls')
