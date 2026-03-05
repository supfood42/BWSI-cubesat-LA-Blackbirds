# Python image processing script
#imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tkinter import Tk
from tkinter.filedialog import askopenfilename

### SAMPLE IMAGE ###
#https://science.nasa.gov/image-detail/amf-pia13642/

#configs

picture_width = 640
picture_height = 480

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
'''
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()
'''
# Convert to grayscale
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


'''
#Regular pipeline for contrast enhance (disabled)
#CONFIGS
contrast = 1.5
brightness = 20
#enhance contrast and brightness
img_enhanced = cv2.convertScaleAbs(img_gray, alpha=contrast, beta=brightness)
'''

#CLAHE enhancement
#CONFIGS
clipLimit = 2.0
clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
img_clahe = clahe.apply(img_gray)

#Noise filtering 
img_gaussian = cv2.GaussianBlur(img_clahe, (5, 5), 0)
#median = cv2.medianBlur(img_clahe, 5)
#bilateral = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)

#Sharpening
img_clahe_sharp = cv2.addWeighted(img_gaussian, 1.5, img_clahe, -0.5, 0)

# --- Canny Edge Detection with Sliders ---
# --- Canny Edge Detection with Side-by-Side Display and Sliders ---
def canny_update(val):
    low = int(slider_low.val)
    high = int(slider_high.val)
    edges = cv2.Canny(img_clahe_sharp, low, high)
    # Overlay green edges on original image
    overlay = img_rgb.copy()
    green = [0, 255, 0]
    mask = edges > 0
    overlay[mask] = green
    ax_canny.imshow(overlay)
    fig.canvas.draw_idle()


init_low = 50
init_high = 150
edges = cv2.Canny(img_clahe_sharp, init_low, init_high)

fig, (ax_orig, ax_canny) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.25, bottom=0.25)

ax_orig.imshow(img_rgb)
ax_orig.set_title("Original Image")
ax_orig.axis("off")

# Overlay green edges on original image for initial display
overlay_init = img_rgb.copy()
green = [0, 255, 0]
mask_init = edges > 0
overlay_init[mask_init] = green
ax_canny.imshow(overlay_init)
ax_canny.set_title("Canny Edges Overlay")
ax_canny.axis("off")

axcolor = 'lightgoldenrodyellow'
ax_low = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_high = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

slider_low = Slider(ax_low, 'Low', 0, 255, valinit=init_low, valstep=1)
slider_high = Slider(ax_high, 'High', 0, 255, valinit=init_high, valstep=1)

slider_low.on_changed(canny_update)
slider_high.on_changed(canny_update)

plt.show()

#TODO: read sat data from log
target_lat = 0.0    #read
target_lon = 0.0    #read
target_width = 2000 #read (meters)
#Satellite position (ME coords)
sat_lat = 0.0       #read
sat_lon = 0.0       #read
sat_alt = 0.0*1000  #read (km to meters)
sat_time = '2026-03-03T18:00:00 UTC' #read
#Satellite attitude (quaternion)
sat_attitude = [1.0, 0.0, 0.0, 0.0] #read (qw, qx, qy, qz)

#TODO: center image, transform image to top-down

#TODO:export covered area to for repeat analysis
