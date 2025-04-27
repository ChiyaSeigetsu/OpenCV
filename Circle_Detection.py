import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# Open file dialog to select an image
root = tk.Tk()
root.withdraw()  # Hide the root window
file_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

if file_path:
    # Read the selected image
    planets = cv2.imread(file_path)
    gray_img = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray_img, 5)

    # Detect circles - improve parameters
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,             # Increased dp for better resolution
        minDist=80,         # Distance between circle centers (adjust based on planet gaps)
        param1=100,         # Canny high threshold
        param2=50,          # Stricter threshold for center detection
        minRadius=30,       # Minimum radius (planet size)
        maxRadius=100       # Maximum radius (planet size)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Save the detected image
        save_path = os.path.splitext(file_path)[0] + "_circles_detected_fixed.jpg"
        cv2.imwrite(save_path, planets)
        print(f"Circle detection completed and saved to: {save_path}")
    else:
        print("No circles were detected.")
else:
    print("No file was selected.")