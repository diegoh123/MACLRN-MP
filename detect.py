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
import pickle
import json


recent_frames = []
MAX_FRAMES = 30	
frame_count = 0 
MODEL_DIR = "tier2_cloud/cloud_storage/models"

def load_model_metadata(model_path):
    """Load metadata JSON for a model given its full path"""
    metadata_path = model_path.replace(".keras", "_metadata.json")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load metadata: {e}")
            return None
    
    print(f"[WARN] No metadata found at {metadata_path}")
    return None

def load_best_model():
    """Load the BEST model based on validation accuracy."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Find model files
    model_files = [
        f for f in os.listdir(MODEL_DIR)
        if f.startswith("mask_detector_v") and f.endswith(".keras")
    ]

    if not model_files:
        print("[WARN] No versioned models found. Loading fallback mask_detector.keras")

        if os.path.exists("mask_detector.keras"):
            print("[INFO] Loaded fallback model: mask_detector.keras")
            return load_model("mask_detector.keras"), None
        else:
            raise FileNotFoundError("No model found: mask_detector.keras missing!")

    # Search for best accuracy
    best_model_file = None
    best_metadata = None
    best_acc = -1.0

    for f in model_files:
        full_path = os.path.join(MODEL_DIR, f)

        metadata = load_model_metadata(full_path)
        if metadata:
            val_acc = metadata.get("val_accuracy", -1)
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_file = f
                best_metadata = metadata

    # NO METADATA FOUND – load latest version ONLY AS FALLBACK
    if best_model_file is None:
        best_model_file = sorted(model_files)[-1]
        full_path = os.path.join(MODEL_DIR, best_model_file)

        print(f"[WARN] No metadata found for any model. Using latest version: {best_model_file}")
        model = load_model(full_path)

        # Print clean summary
        print(f"[INFO] Loaded MODEL FILE: {best_model_file}")
        print("[INFO] No metadata available for performance stats.\n")

        return model, None

    # LOAD BEST MODEL
    full_path = os.path.join(MODEL_DIR, best_model_file)
    model = load_model(full_path)
    print(f"[INFO] BEST MODEL LOADED File: {best_model_file}")

    return model, best_metadata


def download_cloud_model():
    """Download latest model from cloud server"""
    #keras_url = "http://localhost:5000/model/latest/keras"
    #json_url = "http://localhost:5000/model/latest/json"
    keras_url = "https://mask-detection-system.onrender.com/model/latest/keras"
    json_url = "https://mask-detection-system.onrender.com/model/latest/json"

    try:
        r = requests.get(keras_url, timeout=120)
        if r.status_code != 200:
            print("[INFO] No keras model downloaded.")
            return None

        # Extract filename
        header = r.headers.get("Content-Disposition", "")
        if "filename=" in header:
            filename = header.split("filename=")[1].strip('"')
        else:
            filename = "mask_detector_unknown.keras"

        # Ignore invalid / non-model file
        if not filename.endswith(".keras"):
            print(f"[WARN] Server sent NON-MODEL file ({filename}). Ignored.")
            return None

        # Save model
        save_path = os.path.join(MODEL_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(r.content)

        print(f"[INFO] Downloaded cloud MODEL: {filename}")

        r_meta = requests.get(json_url, timeout=30)
        if r_meta.status_code == 200:
            metadata_filename = filename.replace(".keras", "_metadata.json")
            metadata_path = os.path.join(MODEL_DIR, metadata_filename)

            with open(metadata_path, "wb") as f:
                f.write(r_meta.content)

            print(f"[INFO] Downloaded metadata: {metadata_filename}")
        else:
            print("[WARN] Metadata not available from server.")


        return filename

    except Exception as e:
        print("Download failed:", e)
        return None


def detect_and_predict_mask(frame, faceNet, maskNet):
    """Detect faces and predict mask status"""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
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

# Load face detector
prototxtPath = "./face_detector/deploy.prototxt"
weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load mask detection model
maskNet, current_metadata = load_best_model()

# Load label binarizer to get class names
try:
    with open("label_binarizer.pickle", "rb") as f:
        lb = pickle.load(f)
    print(f"[INFO] Loaded classes: {lb.classes_}")
except:
    print("[ERROR] Could not load label_binarizer.pickle")
    exit(1)

# Start video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

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
        
        # Get the index of the highest probability
        class_idx = int(np.argmax(pred))
        internal_label = lb.classes_[class_idx]   # e.g., "with_mask", "without_mask", "improper_mask"
        confidence = float(pred[class_idx])
        
        # Extract clean face crop BEFORE drawing anything
        face_crop = frame[startY:endY, startX:endX].copy()
        
        # Map internal labels to display text + colors
        if internal_label == "with_mask":
            display_label = "Mask"
            color = (0, 255, 0)  # Green
        elif internal_label == "without_mask":
            display_label = "No Mask"
            color = (0, 0, 255)  # Red
        else:  # improper_mask
            display_label = "Improper Mask"
            color = (0, 255, 255)  # Orange
        
        label_text = "{}: {:.2f}%".format(display_label, confidence * 100)
        
        # display output frame
        cv2.putText(frame, label_text, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # Store recent frames for potential evaluation
        recent_frames.append((frame.copy(), internal_label, confidence))
        if len(recent_frames) > MAX_FRAMES:
            recent_frames.pop(0)
        
        # Upload to cloud when label changes and confidence is high
        if internal_label != last_label and confidence > 0.80:
            print(f"[INFO] Face state changed: {last_label} -> {internal_label}")
            last_label = internal_label
            
            # Send the data from detection to cloud server
            try:
                # Resize the CLEAN face crop (without boxes/text)
                face_resized = cv2.resize(face_crop, (224, 224))
                _, img_encoded = cv2.imencode('.jpg', face_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                
                files = {"image": ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
                
                data = {
                    "predicted": internal_label,
                    "correct": "unknown", 
                    "confidence": str(confidence)
                }
                
                res = requests.post(
                    #"http://localhost:5000/upload",
                    "https://mask-detection-system.onrender.com/upload",
                    files=files,
                    data=data,
                    timeout=120
                )
                
                print("Uploaded:", res.json())
            except Exception as e:
                print("Failed to upload:", e)
    
    # Check for new model from cloud every 1500 frames
    frame_count += 1
    if frame_count % 1500 == 0:
        new_file = download_cloud_model()
        if new_file:
            if not new_file.endswith(".keras"):
                print(f"[WARN] Ignoring non-model file: {new_file}")
                continue
            
            temp_path = os.path.join(MODEL_DIR, new_file)
            
            try:
                temp_model = load_model(temp_path)
                new_metadata = load_model_metadata(temp_path)
                
                if new_metadata and current_metadata:
                    # Compare using validation accuracy
                    new_val_acc = new_metadata.get("val_accuracy", 0)
                    cur_val_acc = current_metadata.get("val_accuracy", 0)
                    
                    print(f"[INFO] NEW model v{new_metadata['version']} - Val Acc: {new_val_acc:.4f}")
                    print(f"[INFO] CURRENT model v{current_metadata['version']} - Val Acc: {cur_val_acc:.4f}")
                    
                    if new_val_acc > cur_val_acc:
                        print(f"ACCEPTED new model (better accuracy: {new_val_acc:.4f} > {cur_val_acc:.4f})")
                        maskNet = temp_model
                        current_metadata = new_metadata
                    else:
                        print(f"REJECTED new model (worse accuracy: {new_val_acc:.4f} <= {cur_val_acc:.4f})")
                        print("[INFO] Keeping current model. New model stored but not used.")
						# DO NOTHING — keep the file, don't load it

                elif new_metadata:
                    # New model has metadata, current doesn't - accept it
                    print(f"ACCEPTED new model (has metadata, current doesn't)")
                    maskNet = temp_model
                    current_metadata = new_metadata
                
                else:
                    # No metadata available - fallback warning
                    print("[WARN] No metadata for comparison. Keeping current model.")
                    os.remove(temp_path)
            
            except Exception as e:
                print("Model evaluation failed:", e)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()