# Real-Time-Image-Based-Tumor-Detection-and-Alert-System

This project implements a real-time tumor detection system using computer vision and automated auditory alerts. Live video is captured from a webcam, and then image processing techniques—such as grayscale conversion, Gaussian blur, thresholding, contour extraction, and morphological operations—are applied to identify suspicious tumor-like regions. When a potential abnormality is detected, the system visually highlights the area, logs detailed measurements including timestamps, and instantly provides voice alerts via a text-to-speech engine. This enables prompt notification suitable for preliminary clinical screening and accessibility use-cases.

Features
Real-time detection from webcam video
Visual highlighting of detected regions
Automated text-to-speech alerts for immediate notification
Logging of detection events with timestamps and measurements
Interactive interface for manual frame capture and real-time feedback


Technologies Used
Python 3.x
OpenCV
NumPy
pyttsx3 (text-to-speech)
Runs on Windows 11, tested with Intel i5 CPU and 8GB RAM
