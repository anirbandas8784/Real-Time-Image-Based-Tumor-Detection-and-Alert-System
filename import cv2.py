import cv2
import numpy as np
import os
import csv
from playsound import playsound  # Make sure it's installed: pip install playsound

# ----------------------------
# CONFIGURATION
# ----------------------------
USE_WEBCAM = False  # Set to True if you want to use webcam
IMAGE_FOLDER = "images"  # Folder containing test images
ALERT_SOUND = "alert.mp3"  # Path to a short alert sound (e.g., beep, alarm)

# Create output CSV file
csv_file = open("tumor_data.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Image Name", "Tumor Area", "X", "Y", "Width", "Height"])

# Keep track of whether sound was played
sound_played = False

# Load images from folder
image_files = []
if not USE_WEBCAM:
    folder_path = r"D:\photos clicked by rokey"  # Raw string to avoid issues with backslashes
    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(os.path.join(folder_path, file))

# ----------------------------
# Frame processing function
# ----------------------------
def process_frame(frame, frame_name="Frame"):
    global sound_played

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    edged = cv2.Canny(closed, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tumor_mask = np.zeros_like(gray)
    tumor_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if area > 1000 and w > 20 and h > 20:
            tumor_detected = True
            cv2.drawContours(tumor_mask, [contour], -1, 255, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Area: {int(area)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Write to CSV
            csv_writer.writerow([frame_name, int(area), x, y, w, h])

    # Red overlay
    colored_mask = cv2.cvtColor(tumor_mask, cv2.COLOR_GRAY2BGR)
    overlay = frame.copy()
    overlay[tumor_mask == 255] = (0, 0, 255)
    alpha = 0.4
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    if tumor_detected:
        cv2.putText(frame, "Tumor Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Play sound once
        if not sound_played and os.path.exists(ALERT_SOUND):
            playsound(ALERT_SOUND)
            sound_played = True
    else:
        cv2.putText(frame, "No Tumor Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame

# ----------------------------
# Run detection
# ----------------------------
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame = process_frame(frame)
        cv2.imshow("Tumor Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
else:
    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to load {img_path}")
            continue

        filename = os.path.basename(img_path)
        processed = process_frame(frame, filename)
        cv2.imshow("Tumor Detection", processed)
        print(f"Processed: {filename}")

        key = cv2.waitKey(0)
        if key == ord("q"):
            break

cv2.destroyAllWindows()
csv_file.close()
