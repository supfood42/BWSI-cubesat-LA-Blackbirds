# Python image processing script
#imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

### SAMPLE IMAGE ###
#https://science.nasa.gov/image-detail/amf-pia13642/

#configs
root = Tk()
root.withdraw()  # Hide the main window
image_path = askopenfilename(title="Select an image file", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
root.destroy()

#display image
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise ValueError("Could not load the image.")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title("NASA Mars Surface")
plt.axis("off")
plt.show()

#
