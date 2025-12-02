import keras 
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import requests

recent_frames = []
MAX_FRAMES = 30	
frame_count = 0 

def load_latest_model():
	model_files = [f for f in os.listdir("models") if f.startswith("mask_detector_v") and f.endswith(".keras")]

	if not model_files:
		print("No model files found.")
		return load_model("mask_detector.keras")  # Load default model
	
	model_files.sort()
	latest = model_files[-1]
	print(f"Loading latest model: {latest}")
	return load_model(latest)
	
def evaluate_model(model, samples): #get ave score of model on recent frames
	scores = []
	for (img, old_label, old_conf) in samples:

		rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		resized = cv2.resize(rgb, (224, 224))
		arr = preprocess_input(resized.astype("float32"))
		arr = np.expand_dims(arr, axis=0)

		pred = model.predict(arr)[0]
		confidence = float(max(pred))

		scores.append(confidence)

	return sum(scores) / len(scores) if scores else 0.0

def download_cloud_model(): #get latest model from cloud server
    url = "https://mask-detection-system.onrender.com/model/latest"

    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            print("[INFO] No model downloaded.")
            return None

        # Extract filename from Content-Disposition header
        header = r.headers.get("Content-Disposition", "")
        if "filename=" in header:
            filename = header.split("filename=")[1].strip('"')
        else:
            filename = "mask_detector_unknown.keras"

        save_path = os.path.join("models", filename)
        with open(save_path, "wb") as f:
            f.write(r.content)

        print(f"[INFO] Downloaded cloud model as {filename}")
        return filename

    except Exception as e:
        print("Download failed:", e)
        return None


def load_latest_model(): #get curr model
    model_files = [
        f for f in os.listdir("models")
        if f.startswith("mask_detector_v") and f.endswith(".keras")
    ]

    if not model_files:
        print("[WARN] No local versioned models. Using default mask_detector.keras.")
        return load_model("mask_detector.keras")

    model_files.sort()
    latest = model_files[-1]

    print(f"[INFO] Loading latest local model: {latest}")
    return load_model(os.path.join("models", latest))


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
#maskNet = load_model("mask_detector.keras") 
maskNet = load_latest_model()

vs = VideoStream(src=0).start()

last_label = None

# loop over the frames from the video stream
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=800)

	# prediction for mask or no mask
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# rgb value for frames and text
		label = "Mask" if mask > withoutMask else "No Mask"
		confidence = float(max(mask, withoutMask))

		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label_text = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display output frame
		cv2.putText(frame, label_text, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		#get recent frames
		recent_frames.append((frame.copy(), label, confidence))
		if len(recent_frames) > MAX_FRAMES:
			recent_frames.pop(0)
			
		if label != last_label and confidence > 0.80:
			print(f"[INFO] Face state changed: {last_label} -> {label}")
			last_label = label

			#send the data from detection to cloud server
			try:
				_, img_encoded = cv2.imencode('.jpg', frame)

				files = {"image": ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

				data = {
					"predicted": label,
					"correct": "unknown", 
					"confidence": str(confidence)
				}

				res = requests.post(
					"https://mask-detection-system.onrender.com/upload",
					files=files,
					data=data,
					timeout=5
				)

				print("Uploaded:", res.json())
			except Exception as e:
				print("Failed to upload:", e)

	frame_count += 1
	if frame_count % 500 == 0:
		new_file = download_cloud_model()
		if new_file:
			temp_path = os.path.join("models", new_file)

			try:
				temp_model = load_model(temp_path)
				new_score = evaluate_model(temp_model, recent_frames)
				cur_score = evaluate_model(maskNet, recent_frames)

				print(f"[INFO] new={new_score:.4f} current={cur_score:.4f}")

				if new_score >= cur_score:
					print("[INFO] Accepted new model:", new_file)
					maskNet = temp_model
				else:
					print("[INFO] Rejected new model:", new_file)
					os.remove(temp_path)

			except Exception as e:
				print("Model evaluation failed:", e)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()