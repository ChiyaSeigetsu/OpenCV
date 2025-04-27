import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def non_max_suppression_circle(circles, threshold_dist=50):
    filtered = []
    for c in circles:
        keep = True
        for f in filtered:
            dist = np.linalg.norm(np.array(c[:2]) - np.array(f[:2]))
            if dist < threshold_dist:
                if c[2] < f[2]:  # Keep the larger one
                    keep = False
                    break
                else:
                    filtered.remove(f)
        if keep:
            filtered.append(c)
    return filtered

# --- Open file dialog ---
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

if file_path:
    image = cv2.imread(file_path)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    # --- Detect circles first ---
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=100,
        param2=50,
        minRadius=30,
        maxRadius=100
    )

    final_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        final_circles = non_max_suppression_circle(circles)

        for (x, y, r) in final_circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
            cv2.putText(output, "Circle", (x - 40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --- Then detect other shapes ---
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 50000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            vertices = len(approx)

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)

            # Check if already detected as a circle
            is_circle = False
            for (cx, cy, cr) in final_circles:
                if np.linalg.norm(np.array(center) - np.array((cx, cy))) < 0.6 * cr:
                    is_circle = True
                    break
            if is_circle:
                continue

            if vertices == 3:
                shape = "Triangle"
            elif vertices == 4:
                shape = "Rectangle"
            elif vertices == 5:
                shape = "Pentagon"
            elif vertices == 6:
                shape = "Hexagon"
            elif vertices == 7:
                shape = "Heptagon"
            elif vertices == 8:
                shape = "Octagon"
            elif vertices == 9:
                shape = "Nonagon"
            else:
                shape = "Polygon"

            cv2.drawContours(output, [approx], -1, (0, 255, 255), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output, shape, (cX - 40, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # --- Save output ---
    save_path = os.path.splitext(file_path)[0] + "_shapes_detected.jpg"
    cv2.imwrite(save_path, output)
    print(f"Detection completed and saved to: {save_path}")

else:
    print("No file selected.")
