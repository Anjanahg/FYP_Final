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
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img_resized = pp.resizing(img)
        pp.applying_otsu(img_resized)
        # Initializing a counter
        counter.set(counter.get() + 1)
        # Read image from the path

        if (counter.get() == 1):
            ResultApp.image_1 = ImageTk.PhotoImage(Image.open('./Images/SkullStriping/img_otsu.jpg').resize((200, 258)))
            label_1 = tk.Label(ResultApp, image=ResultApp.image_1)
            label_1.place(x=210, y=3)

    # Next Button
    button_next = tk.Button(ResultApp, text=">>", padx=10, pady=5, command=buttonClick)
    button_next.place(x=700, y=500)

    # OriginalImage -----------------------
    ResultApp.image = ImageTk.PhotoImage(Image.open(filepath).resize((200, 258)))
    label = tk.Label(ResultApp, image=ResultApp.image)
    label.place(x=3, y=3)
    tk.Label(ResultApp, text="Input Image", fg="VioletRed4", font="Verdana 10 bold italic").place(x=50, y=265)

    ResultApp.mainloop()

