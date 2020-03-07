import tkinter as tk
from PIL import Image, ImageTk
import time
import cv2
import Preprocessing as pp

def main(filepath):
   #window attributes
    ResultApp = tk.Toplevel()
    ResultApp.title("Tumor Identification process")
    ResultApp.geometry('1530x750')
    counter = tk.IntVar()

# methods
    def buttonClick():
        img = cv2.imread(filepath)
        #img_resized = pp.resizing(img)
        img_skull_removed = pp.skull_striping(img)



        # Initializing a counter
        counter.set(counter.get() + 1)
        # Read image from the path

        if (counter.get() == 1):
            ResultApp.image_1 = ImageTk.PhotoImage(Image.open('./Images/SkullStriping/img_otsu.jpg').resize((200, 258)))
            label_2 = tk.Label(ResultApp, image=ResultApp.image_1)
            label_2.place(x=210, y=3)
            tk.Label(ResultApp, text="Otsu", fg="Blue", font="Verdana 10 bold italic").place(x=250, y=265)

        if (counter.get() == 2):
            ResultApp.image_2 = ImageTk.PhotoImage(Image.open('./Images/SkullStriping/img_blended.jpg').resize((200, 258)))
            label_2 = tk.Label(ResultApp, image=ResultApp.image_2)
            label_2.place(x=420, y=3)
            tk.Label(ResultApp, text="Blended", fg="Blue", font="Verdana 10 bold italic").place(x=450, y=265)


        if (counter.get() == 3):
            ResultApp.image_3 = ImageTk.PhotoImage(Image.open('./Images/SkullStriping/img_connected_components.jpg').resize((200, 258)))
            label_3 = tk.Label(ResultApp, image=ResultApp.image_3)
            label_3.place(x=630, y=3)
            tk.Label(ResultApp, text="Connected components", fg="Blue", font="Verdana 10 bold italic").place(x=650, y=265)


        if (counter.get() == 4):
            ResultApp.image_4 = ImageTk.PhotoImage(
                Image.open('./Images/SkullStriping/img_closing.jpg').resize((200, 258)))
            label_4 = tk.Label(ResultApp, image= ResultApp.image_4)
            label_4.place(x=840, y=3)
            tk.Label(ResultApp, text="Closing", fg="Blue", font="Verdana 10 bold italic").place(x=850, y=265)

    # Next Button
    button_next = tk.Button(ResultApp, text=">>", padx=10, pady=5, command=buttonClick)
    button_next.place(x=700, y=500)

    # OriginalImage -----------------------
    ResultApp.image = ImageTk.PhotoImage(Image.open(filepath).resize((200, 258)))
    label = tk.Label(ResultApp, image=ResultApp.image)
    label.place(x=3, y=3)
    tk.Label(ResultApp, text="Input Image", fg="Blue", font="Verdana 10 bold italic").place(x=50, y=265)

    ResultApp.mainloop()

