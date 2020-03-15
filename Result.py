import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
import Preprocessing as pp
import Cnn_Tumor as ct
import Methods as m
import KnowledgeBase as kb
import Classification as clsf
import Cnn_Skull_Remove as csr

def main(filepath):
   #window attributes
    ResultApp = tk.Toplevel()
    ResultApp.title("Tumor Identification process")
    ResultApp.geometry('1480x700')
    counter = tk.IntVar()

# methods
    def buttonClick():
        img = cv2.imread(filepath)
        # writing the original image to input folder
        cv2.imwrite('./Images/Input/img_original.jpg', img)

        
        # img_resized = pp.resizing(img)

        # calling the skull striping method from Pre-processing.py
        pp.skull_striping(img)
        csr.main()


        # Initializing a counter
        counter.set(counter.get() + 1)

        # display otsu image--------------------------------------------------------------------------------------------
        if (counter.get() == 1):
            ResultApp.image_1 = ImageTk.PhotoImage(Image.open('./Images/SkullStriping/img_otsu.jpg').resize((200, 258)))
            label_2 = tk.Label(ResultApp, image=ResultApp.image_1)
            label_2.place(x=220, y=3)
            tk.Label(ResultApp, text="Threshold", fg="Blue", font="Arial 10 bold").place(x=285, y=265)

        # display Blended image-----------------------------------------------------------------------------------------
        if (counter.get() == 2):
            ResultApp.image_2 = ImageTk.PhotoImage(Image.open('./Images/SkullStriping/img_blended.jpg').resize((200, 258)))
            label_2 = tk.Label(ResultApp, image=ResultApp.image_2)
            label_2.place(x=430, y=3)
            tk.Label(ResultApp, text="Blended", fg="Blue", font="Arial 10 bold").place(x=495, y=265)

        # Display Skull removed-----------------------------------------------------------------------------------------
        if (counter.get() == 3):
            ResultApp.image_3 = ImageTk.PhotoImage(Image.open('./Images/SkullStriping/img_watershed.jpg').resize((200, 258)))
            label_3 = tk.Label(ResultApp, image=ResultApp.image_3)
            label_3.place(x=640, y=3)
            tk.Label(ResultApp, text=" Watershed", fg="Blue", font="Arial 10 bold").place(x=680, y=265)

        # Display Closing-----------------------------------------------------------------------------------------------
        if (counter.get() == 4):
            ResultApp.image_4 = ImageTk.PhotoImage(
                Image.open('./Images/CNN_Skull_Remove_Output/img_skull_removed_by_cnn.jpg').resize((200, 258)))
            label_4 = tk.Label(ResultApp, image= ResultApp.image_4)
            label_4.place(x=850, y=3)
            tk.Label(ResultApp, text="Skull Stripped", fg="Blue", font="Arial 10 bold").place(x=900, y=265)

        # Display Segmented image by CNN--------------------------------------------------------------------------------
        if (counter.get() == 5):
            ct.main()
            # get results and value array from classification.py
            result, featureSet = clsf.get_rf_result()
            if result == 0:
                ResultApp.image_5 = ImageTk.PhotoImage(
                    Image.open('./Images/CNN_Output/blank_mask.jpg').resize((200, 258)))
            else:
                ResultApp.image_5 = ImageTk.PhotoImage(
                    Image.open('./Images/CNN_Output/img_segment_by_cnn.jpg').resize((200, 258)))

            label_5 = tk.Label(ResultApp, image=ResultApp.image_5)
            label_5.place(x=1060, y=3)
            tk.Label(ResultApp, text="Segmented image", fg="Blue", font="Arial 10 bold").place(x=1095, y=265)

        # Display Segmented image---------------------------------------------------------------------------------------
        if (counter.get() == 6):
            ResultApp.image_6 = ImageTk.PhotoImage(
                Image.open('./Images/CNN_Output/img_tumor_marked.jpg').resize((200, 258)))
            label_6 = tk.Label(ResultApp, image=ResultApp.image_6)
            label_6.place(x=1270, y=3)

            ct.main()
            # get results and value array from classification.py
            result, featureSet = clsf.get_rf_result()
            if result == 0:
                tk.Label(ResultApp, text="No Tumor Detected", fg="Green", font="Arial 10 bold").place(x=1310, y=265)
            else:
                tk.Label(ResultApp, text="Tumor identified", fg="Red", font="Arial 10 bold").place(x=1310, y=265)



        # Features section---------------------------------------------------------------------------------------
        if (counter.get() == 7):
            # get results and value array from classification.py
            result, featureSet = clsf.get_rf_result()
            # mark the state according to the result
            if result[0]==1:
                state = "Yes"
            else:
                state = "No"
            tk.Label(ResultApp, text="--------Image Feature Set-------", fg="firebrick4", font="Verdana 10 bold").place(
                x=10, y=330)
            tk.Label(ResultApp, text="Mean :  " + str(featureSet[0][0]), font="Verdana 10 bold").place(
                x=10, y=360)
            tk.Label(ResultApp, text="Entropy :  " + str(featureSet[0][1]), font="Verdana 10 bold").place(
                x=10, y=390)
            tk.Label(ResultApp, text="Kurtosis :  " + str(featureSet[0][2]),font="Verdana 10 bold").place(
                x=10, y=420)
            tk.Label(ResultApp, text="Standard Dev :  " + str(featureSet[0][3]), font="Verdana 10 bold").place(
                x=10, y=450)
            tk.Label(ResultApp, text="Skewness :  " + str(featureSet[0][4]), font="Verdana 10 bold").place(
                x=10, y=480)
            tk.Label(ResultApp, text="Contrast :  " + str(featureSet[0][5]), font="Verdana 10 bold").place(
                x=10, y=510)
            tk.Label(ResultApp, text="Homogeneity :  " + str(featureSet[0][6]), font="Verdana 10 bold").place(
                x=10, y=540)
            tk.Label(ResultApp, text="Co-Relation :  " + str(featureSet[0][7]), font="Verdana 10 bold").place(
                x=10, y=570)
            tk.Label(ResultApp, text="Energy :  " + str(featureSet[0][8]), font="Verdana 10 bold").place(
                x=10, y=600)
            tk.Label(ResultApp, text="Dissimilarity :  " + str(featureSet[0][9]), font="Verdana 10 bold").place(
                x=10, y=630)
            tk.Label(ResultApp, text="State :  " + state, fg="red", font="Verdana 10 bold").place(
                x=10, y=660)

        # Display tumor marked image---------------------------------------------------------------------------------------
        if (counter.get() == 8):
            # get results and value array from classification.py
            result, featureSet = clsf.get_rf_result()
            if result == 0:
                ResultApp.image_7 = ImageTk.PhotoImage(
                    Image.open('./Images/CNN_Skull_Remove_Output/img_skull_removed_by_cnn.jpg').resize((200, 258)))
                label_7 = tk.Label(ResultApp, image=ResultApp.image_7)
                label_7.place(x=1060, y=300)
                tk.Label(ResultApp, text="No Tumor region", fg="Green", font="Arial 10 bold").place(x=1105, y=570)
            else:
                # calling knowledge base
                kb.main()
                ResultApp.image_7 = ImageTk.PhotoImage(
                    Image.open('./Images/Knowledgebase_Output/kb_img_tumor_marked_1.jpg').resize((200, 258)))
                label_7 = tk.Label(ResultApp, image=ResultApp.image_7)
                label_7.place(x=1060, y=300)
                tk.Label(ResultApp, text="Tumor region", fg="Red", font="Arial 10 bold").place(x=1105, y=570)



        # Display tumor marked image 2--------------------------------------------------------------------------------------
        if (counter.get() == 9):
            # get results and value array from classification.py
            result, featureSet = clsf.get_rf_result()
            if result == 0:
                ResultApp.image_8 = ImageTk.PhotoImage(
                    Image.open('./Images/CNN_Skull_Remove_Output/img_skull_removed_by_cnn.jpg').resize((200, 258)))
                label_8 = tk.Label(ResultApp, image=ResultApp.image_8)
                label_8.place(x=1270, y=300)
                tk.Label(ResultApp, text="No Critical Area", fg="Green", font="Arial 10 bold").place(x=1320, y=570)
            else:
                ResultApp.image_8 = ImageTk.PhotoImage(
                    Image.open('./Images/Knowledgebase_Output/kb_img_tumor_marked_2.jpg').resize((200, 258)))
                label_8 = tk.Label(ResultApp, image=ResultApp.image_8)
                label_8.place(x=1270, y=300)
                tk.Label(ResultApp, text="Critical Area", fg="Red", font="Arial 10 bold").place(x=1320, y=570)





    # Next Button
    button_next = tk.Button(ResultApp, text="Click Here", padx=60, pady=5, command=buttonClick)
    button_next.place(x=680, y=660)

    # OriginalImage ----------------------------------------------------------------------------------------------------
    ResultApp.image = ImageTk.PhotoImage(Image.open(filepath).resize((200, 258)))
    label = tk.Label(ResultApp, image=ResultApp.image)
    label.place(x=10, y=3)
    tk.Label(ResultApp, text="Input Image", fg="Blue", font="Arial 10 bold").place(x=70, y=265)

    #


    ResultApp.mainloop()

