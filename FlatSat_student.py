"""
The Python code you will write for this module should read
acceleration data from the IMU. When a reading comes in that surpasses
an acceleration threshold (indicating a shake), your Pi should pause,
trigger the camera to take a picture, then save the image with a
descriptive filename. You may use GitHub to upload your images automatically,
but for this activity it is not required.

The provided functions are only for reference, you do not need to use them. 
You will need to complete the take_photo() function and configure the VARIABLES section
"""

#AUTHOR: 
#DATE:

#import libraries
import time
import board
import os
import math
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS
from adafruit_lis3mdl import LIS3MDL
from git import Repo
from picamera2 import Picamera2

# VARIABLES
THRESHOLD = 3  # allowed deviation from gravity; tune up/down as needed
REPO_PATH = "/home/supfood/BWSI-cubesat-LA-Blackbirds"  # local clone on the Pi
FOLDER_PATH = "Pictures"  # no leading slash

#imu and camera initialization
i2c = board.I2C()
accel_gyro = LSM6DS(i2c)
mag = LIS3MDL(i2c)
picam2 = Picamera2()


def git_push():
    """
    Stages, commits, and pushes new images to your GitHub repo.
    """
    repo = Repo(REPO_PATH)
    pictures_path = os.path.join(REPO_PATH, FOLDER_PATH)
    try:
        repo.git.add(pictures_path)
        repo.index.commit('New Photo')
    except Exception as e:
        # Likely nothing to commit (e.g., identical files)
        print(f"Nothing to commit or commit failed: {e}")
        return
    try:
        origin = repo.remote('origin')
        origin.pull()
        origin.push()
        print('pushed changes')
    except Exception as e:
        print(f"Couldn’t upload to git: {e}")


def img_gen(name):
    """
    This function is complete. Generates a new image name.

    Parameters:
        name (str): your name ex. MasonM
    """
    t = time.strftime("_%H%M%S")
    imgname = (f'{REPO_PATH}/{FOLDER_PATH}/{name}{t}.jpg')
    return imgname

def take_photo():
    """
    Takes a photo when the FlatSat is shaken.
    Saves images to both the GitHub repo and the local Pictures folder.
    """
    name = "Aaron"  # First Name, Last Initial
    local_pictures_dir = "/home/supfood/Pictures"

    # Ensure local Pictures directory exists
    if not os.path.exists(local_pictures_dir):
        os.makedirs(local_pictures_dir)

    while True:
        accelx, accely, accelz = accel_gyro.acceleration
        magnitude = math.sqrt(accelx**2 + accely**2 + accelz**2)
        delta = abs(magnitude - 9.81)  # subtract gravity

        if delta >= THRESHOLD:
            print(f"Shake detected (delta={delta:.2f}). Capturing photo in 1 sec...")
            time.sleep(1)  # Pause

            # Generate filenames
            t = time.strftime("_%H%M%S")
            repo_img_path = img_gen(name)
            local_img_path = f"{local_pictures_dir}/{name}{t}.jpg"

            picam2.start()
            picam2.capture_file(repo_img_path)
            picam2.capture_file(local_img_path)
            picam2.stop()

            print(f"Photos saved to repo and {local_img_path}")

            git_push()

        time.sleep(2.0)  # cooldown before checking again


def main():
    take_photo()


if __name__ == '__main__':
    main()