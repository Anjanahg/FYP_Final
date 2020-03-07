import tkinter as tk
from PIL import Image, ImageTk
import time
import cv2
import Preprocessing as pp
import Cnn_Tumor as ct
def main(filepath):
   #window attributes
    ResultApp = tk.Toplevel()
    ResultApp.title("Tumor Identification process")
    ResultApp.geometry('1480x700')
    counter = tk.IntVar()

# methods
    def buttonClick():
        img = cv2.imread(filepath)
        #img_resized = pp.resizing(img)
        img_skull_removed = pp.skull_striping(img)



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
            ResultApp.image_3 = ImageTk.PhotoImage(Image.open('./Images/SkullStriping/img_skull_removed.jpg').resize((200, 258)))
            label_3 = tk.Label(ResultApp, image=ResultApp.image_3)
            label_3.place(x=640, y=3)
            tk.Label(ResultApp, text=" Skull Removed", fg="Blue", font="Arial 10 bold").place(x=680, y=265)

        # Display Closing-----------------------------------------------------------------------------------------------
        if (counter.get() == 4):
            ResultApp.image_4 = ImageTk.PhotoImage(
                Image.open('./Images/SkullStriping/img_closing.jpg').resize((200, 258)))
            label_4 = tk.Label(ResultApp, image= ResultApp.image_4)
            label_4.place(x=850, y=3)
            tk.Label(ResultApp, text="Closing", fg="Blue", font="Arial 10 bold").place(x=900, y=265)

        # Display Segmented image by CNN--------------------------------------------------------------------------------
        if (counter.get() == 5):
            ct.main()
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
            tk.Label(ResultApp, text="Tumor identified", fg="Red", font="Arial 10 bold").place(x=1310, y=265)


    # Next Button
    button_next = tk.Button(ResultApp, text=">>", padx=50, pady=5, command=buttonClick)
    button_next.place(x=740, y=580)

    # OriginalImage ----------------------------------------------------------------------------------------------------
    ResultApp.image = ImageTk.PhotoImage(Image.open(filepath).resize((200, 258)))
    label = tk.Label(ResultApp, image=ResultApp.image)
    label.place(x=10, y=3)
    tk.Label(ResultApp, text="Input Image", fg="Blue", font="Arial 10 bold").place(x=70, y=265)

    ResultApp.mainloop()

