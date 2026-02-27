# Python image processing script
#imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

### SAMPLE IMAGE ###
#https://science.nasa.gov/image-detail/amf-pia13642/

#configs
image_path = ''

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
