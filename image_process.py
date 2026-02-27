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

#Show RGB image
"""
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title("NASA Mars Surface")
plt.axis("off")
plt.show()
"""
#

#convert grayscale
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#TODO: import Orbital coords, target coords, camera orientation 

#TODO: Transform image to top-down view (homography)

#Add contrast and brightness
alpha = 1.5  # contrast
beta = 20     # brightness
adjusted = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)

#CLAHE enhancement
"""
plt.figure(figsize=(4, 3))
plt.hist(img_gray.ravel(), bins=40, color="gray")
plt.title("Grayscale histogram")
plt.xlabel("Pixel value")
plt.ylabel("Count")
plt.show()
"""
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(img_gray)

for title, img in [
    ("Original", img_gray),
    ("CLAHE Enhanced", clahe_img),
]:
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()
