from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import pickle

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		# filters out weak detections using confidence threshold
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# prediction begins only if face is detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

prototxtPath = "./face_detector/deploy.prototxt"
weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.keras")

# Load label mapping so we do not hardcode class names/order
with open("label_binarizer.pickle", "rb") as f:
    lb = pickle.load(f)

vs = VideoStream(src=0).start()


# loop over the frames from the video stream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=800)

	# prediction for mask or no mask
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box

		# Get the index of the highest probability
		class_idx = int(np.argmax(pred))
		internal_label = lb.classes_[class_idx]   # e.g., "with_mask", "without_mask", "improper_mask"
		prob = float(pred[class_idx])

		# Map internal labels to display text + colors
		if internal_label == "with_mask":
			display_label = "Mask"
			color = (0, 255, 0)
		elif internal_label == "without_mask":
			display_label = "No Mask"
			color = (0, 0, 255)
		else:
			display_label = "Improper Mask"
			color = (0, 255, 255)  # yellow-ish

		label = "{}: {:.2f}%".format(display_label, prob * 100)


		# display output frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()