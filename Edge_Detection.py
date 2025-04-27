import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog

def ask_for_thresholds():
    """
    Prompts the user for Canny edge detection thresholds (lower and upper).
    Returns:
        tuple: lower and upper threshold values.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Ask for the lower threshold
    lower_threshold = simpledialog.askinteger("Input", "Enter lower threshold (default 10):", minvalue=1, maxvalue=255, initialvalue=10)

    # Ask for the upper threshold
    upper_threshold = simpledialog.askinteger("Input", "Enter upper threshold (default 40):", minvalue=1, maxvalue=255, initialvalue=40)

    # Return the thresholds, defaulting to 10 and 40 if no input is given
    return lower_threshold if lower_threshold else 10, upper_threshold if upper_threshold else 40

def detect_edges_and_contours(image_path, lower_threshold=10, upper_threshold=40):
    """
    Detects edges and contours in an image using Canny, Laplacian, and Sobel filters.
    Saves the resulting edge-detected and contour images to files.

    Args:
        image_path (str): The path to the input image file.
        lower_threshold (int): Lower threshold for Canny edge detection.
        upper_threshold (int): Upper threshold for Canny edge detection.

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

    # 2. Pre-processing: Reduce noise (important for edge and contour detection)
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
    cv2.imwrite(f"{base_name}_laplacian.jpg", laplacian)

    # c) Canny Edge Detection with user-defined thresholds
    canny = cv2.Canny(gray, lower_threshold, upper_threshold)  # Lower and upper thresholds
    cv2.imwrite(f"{base_name}_canny.jpg", canny)

    # 5. Contour Detection
    # Find contours in the cleaned Canny output
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image (for visualization)
    contour_img = img.copy()  # Create a copy of original image to draw contours on
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)  # Draw all contours in green
    cv2.imwrite(f"{base_name}_contours.jpg", contour_img)  # save contour image

    # 6. Crack Fixing (Simplified)
    fixed_img = fix_cracks(img, contours)
    cv2.imwrite(f"{base_name}_fixed.jpg", fixed_img)

def fix_cracks(image, contours):
    """
    Attempts to fix cracks in the image by dilating the regions around the detected contours.
    This is a simplified approach and may not work for all types of cracks.

    Args:
        image (numpy.ndarray): The original input image.
        contours (list): List of contours detected in the image.

    Returns:
        numpy.ndarray: The image with the cracks "fixed".
    """
    # Create a mask from the contours
    mask = np.zeros_like(image[:,:,0], dtype=np.uint8)  # Single-channel mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)  # Fill the contours in the mask

    # Dilate the mask to fill in crack gaps
    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)  # Adjust iterations as needed

    # Use the dilated mask to inpaint the original image
    fixed_image = cv2.inpaint(image, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA) # Adjust inpaintRadius as needed

    return fixed_image

def main():
    # 1. Create a Tkinter root window (it will be hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # 2. Show the file dialog and get the selected file path
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.*")]  # Filter for image files
    )

    # 3. Get custom thresholds from the user
    lower_threshold, upper_threshold = ask_for_thresholds()

    # 4. Check if a file was selected
    if file_path:
        detect_edges_and_contours(file_path, lower_threshold, upper_threshold)  # Process the selected image with custom thresholds
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()
