#!/usr/bin/env python3
"""Real-time mask detector using a trained Keras model and OpenCV Haar cascades.

Usage examples:
  # Webcam (default):
  python3 detect_mask_camera.py

  # Specific camera index:
  python3 detect_mask_camera.py --camera 1

  # Video file:
  python3 detect_mask_camera.py --video path/to/video.mp4

Dependencies:
  pip install tensorflow opencv-python

The script expects a `mask_detector.h5` model in the current directory by default.
"""

import argparse
import time
import cv2
import numpy as np
import os
import keras
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.models import load_model
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def parse_args():
    parser = argparse.ArgumentParser(description="Run face mask detection from webcam or video file.")
    parser.add_argument("--model", type=str, default="mask_detector.h5",
                        help="Path to trained Keras model file (HDF5).")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video file. If omitted, webcam is used.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Integer index of webcam to use (default 0).")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Minimum probability to accept a prediction.")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not show the video window (run headless).")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Make sure `mask_detector.h5` is in the current directory or pass --model path")
        return

    print(f"Loading model from: {args.model}")
    model = load_model(args.model)

    # Use OpenCV's Haar Cascade (bundled with opencv-python)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Define class labels in the same order used during training
    LABELS = ["with_mask", "without_mask"]

    # Open video stream
    if args.video:
        print(f"Opening video file: {args.video}")
        vs = cv2.VideoCapture(args.video)
    else:
        print(f"Opening webcam (index {args.camera})")
        vs = cv2.VideoCapture(args.camera)

    time.sleep(1.0)

    while True:
        ret, frame = vs.read()
        if frame is None or not ret:
            break

        # Resize frame for consistent processing speed (optional)
        frame = cv2.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # Extract face ROI and preprocess for MobileNetV2
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_resized = face_resized.astype("float32")
            face_pre = preprocess_input(face_resized)
            face_blob = np.expand_dims(face_pre, axis=0)

            preds = model.predict(face_blob)
            preds = preds[0]
            class_idx = int(np.argmax(preds))
            label = LABELS[class_idx] if class_idx < len(LABELS) else str(class_idx)
            prob = preds[class_idx]

            # Choose display label and box color
            display_label = "Mask" if label == "with_mask" else "No Mask"
            color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)

            text = f"{display_label}: {prob * 100:.2f}%"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        if not args.no_display:
            cv2.imshow("Mask Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
