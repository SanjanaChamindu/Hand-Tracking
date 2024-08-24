# Hand Tracking and Finger Counting

This project uses computer vision and machine learning to track hand movements and count the number of fingers shown in real-time using a webcam. The project leverages OpenCV for image processing, MediaPipe for hand tracking, and scikit-learn for machine learning.


## Project Description

The project consists of two main parts:
1. **Hand Tracking**: Using MediaPipe to detect and track hand landmarks in real-time.
2. **Finger Counting**: Using a machine learning model to count the number of fingers shown based on the hand landmarks.

## Installation

To run this project, you need to have Python installed. You can install the required packages using pip:

```bash
pip install opencv-python mediapipe scikit-learn joblib numpy
