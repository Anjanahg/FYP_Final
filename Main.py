import tkinter as tk
from tkinter import filedialog, messagebox

import Result
import Cnn_Tumor

# UI attributes
root = tk.Tk()
root.title("Brain Tumor Identifier")
root.geometry('500x150')

# variables
mri_image = tk.StringVar()

# methods
def open_brain_image():
    filename = filedialog.askopenfilename(initialdir="/", filetypes=[('Png Files', '*.png'), ('Jpeg Files', '*.jpg'),('Dicom Files', '*.DCM')])
    mri_image.set(filename)

# process starting method
def start_finding():
    Result.main(mri_image.get())

# title
topic_1 = tk.Label(root, text="File Location", fg="blue", font="Verdana 10 bold")
topic_1.grid(row=0, column=3, padx=(5, 10), pady=(10, 1))

entry_1 = tk.Entry(root, textvariable=mri_image)
entry_1.grid(row=1, column=3, padx=(5, 10), pady=(20, 20))

# file open button
openfile_1 = tk.Button(root, text="Open File", padx=40, pady=5, command=open_brain_image)
openfile_1.grid(row=1, column=2, padx=(5, 5))

# Start Calculation button
start_button = tk.Button(root, text="Start Process", padx=40, pady=5, command=start_finding)
start_button.grid(row=2, column=2, padx=(5, 5))

root.mainloop()
