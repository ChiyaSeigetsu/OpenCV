import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
import os

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

def detect_and_fix_cracks(image_path, lower_threshold=10, upper_threshold=40):
    """
    Detects lines (cracks) in an image using the Probabilistic Hough Transform and attempts to fix them.
    Saves the original image with detected and fixed cracks to a file.

    Args:
        image_path (str): The path to the input image file.
        lower_threshold (int): Lower threshold for Canny edge detection.
        upper_threshold (int): Upper threshold for Canny edge detection.

    Returns:
        None.  Saves the image to a file.
    """
    # 1. Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    # Get the original image's name without extension for saving
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    height, width = img.shape[:2]

    # 2. Pre-processing: Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Edge Detection: Use Canny Edge Detector
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

    # 4. Morphological Closing (to connect broken edges)
    kernel = np.ones((5, 5), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5. Line Detection: Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges_closed, 1, np.pi / 180, threshold=60,
                            minLineLength=40, maxLineGap=60)

    # 6. Draw the detected lines on the original image
    crack_img = img.copy()
    if lines is not None:
        print(f"Detected {len(lines)} lines.")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(crack_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        print("No lines detected.")
    cv2.imwrite(f"{base_name}_cracks.jpg", crack_img)  # Save the image with detected cracks

    # 7. Crack Fixing (Simplified)
    fixed_img = fix_cracks(img, lines, height, width)

    # 8. Save the image with detected and "fixed" cracks
    cv2.imwrite(f"{base_name}_cracks_fixed.jpg", fixed_img)

def fix_cracks(image, lines, height, width):
    """
    Attempts to fix cracks in the image by dilating the regions around the detected lines.
    This is a simplified approach and may not work for all types of cracks.

    Args:
        image (numpy.ndarray): The original input image.
        lines (list): List of lines detected in the image.
        height (int): height of the image
        width (int): width of the image

    Returns:
        numpy.ndarray: The image with the cracks "fixed".
    """
    # Create a mask
    mask = np.zeros((height, width), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)  # Draw lines on mask

    # Dilate the mask to fill in crack gaps
    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)  # Adjust iterations as needed

    # Use the dilated mask to inpaint the original image
    fixed_image = cv2.inpaint(image, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)  # Adjust parameters as needed

    return fixed_image

def main():
    # 1. Create a Tkinter root window (it will be hidden)
    root = tk.Tk()
    root.withdraw()

    # 2. Show the file dialog and get the selected file path
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.*")]
    )

    # 3. Get custom thresholds from the user
    lower_threshold, upper_threshold = ask_for_thresholds()

    # 4. Check if a file was selected
    if file_path:
        detect_and_fix_cracks(file_path, lower_threshold, upper_threshold)  # Process the selected image with custom thresholds
        print(f"Image processed and saved: {os.path.splitext(os.path.basename(file_path))[0]}_cracks.jpg and {os.path.splitext(os.path.basename(file_path))[0]}_cracks_fixed.jpg")
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()
