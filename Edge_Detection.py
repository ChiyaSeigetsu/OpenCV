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
        None.  Displays the original image and the edge-detected images.
    """
    # 1. Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as color image
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    # Get the original image's name without extension for saving later
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 2. Pre-processing: Reduce noise (important for edge detection)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)  # 5x5 Gaussian blur

    # 3. Convert to grayscale (edge detection is often done on grayscale images)
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
    cv2.imwrite(f"{base_name}_laplacian.jpg", laplacian)

    # c) Canny Edge Detection
    canny = cv2.Canny(gray, 20, 50)  # Lower and upper thresholds
    cv2.imwrite(f"{base_name}_canny.jpg", canny)

    # 5. Display the results
    cv2.imshow('Original Image', img)
    cv2.imshow('Sobel Edges', sobel_combined)
    cv2.imshow('Laplacian Edges', laplacian)
    cv2.imshow('Canny Edges', canny)
    cv2.waitKey(0)  # Wait for a key press to close the windows
    cv2.destroyAllWindows()

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
