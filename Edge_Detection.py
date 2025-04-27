import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def detect_edges(image_path):
    """
    Detects edges in an image using Canny, Laplacian, and Sobel filters.
    Saves the resulting edge-detected images to files.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        None.
    """
    # 1. Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as color image
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    # Get the original image's name without extension for saving later
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 2. Pre-processing: Reduce noise (important for edge detection)
    blurred = cv2.GaussianBlur(img, (7, 7), 0)  # Increased Gaussian kernel size to 7x7 for more blurring
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # 4. Edge Detection
    # a) Sobel Edge Detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in X direction
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in Y direction
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Combine X and Y gradients
    sobel_combined = np.uint8(sobel_combined)
    cv2.imwrite(f"{base_name}_sobel.jpg", sobel_combined)

    # b) Laplacian Edge Detection
    laplacian = cv2.Laplacian(gray, cv2.CV_8U)
    # Increase brightness of Laplacian output
    laplacian_bright = np.clip(laplacian * 2.0, 0, 255).astype(np.uint8)  # Multiply by 2 and clip
    cv2.imwrite(f"{base_name}_laplacian.jpg", laplacian_bright)

    # c) Canny Edge Detection
    canny = cv2.Canny(gray, 20, 40)  # Lower and upper thresholds

    # Apply morphological operations to clean up Canny edges
    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel for morphological operations
    canny_cleaned = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=1)  # Closing operation
    cv2.imwrite(f"{base_name}_canny.jpg", canny_cleaned)



def main():
    # 1. Create a Tkinter root window (it will be hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # 2. Show the file dialog and get the selected file path
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]  # Filter for image files
    )

    # 3. Check if a file was selected
    if file_path:
        detect_edges(file_path)  # Process the selected image
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()
