import os
from flask import Flask, request, jsonify, send_file
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

app = Flask(__name__)

UPLOAD_FOLDER = "tier2_cloud/cloud_storage/uploads/"
MODEL_FOLDER = "tier2_cloud/cloud_storage/models/"

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

#connect to db

client = MongoClient(os.getenv("MONGODB_URI"))
db = client['maclrn_db']
uploads = db['uploads']

try:
    client.admin.command('ping')
    print("Connected to MongoDB successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

@app.route("/ping")
def ping():
    return {"status": "success", "message": "pong!"}

def save_async(image, save_path, record):
    image.save(save_path)
    uploads.insert_one(record)

#send new data + metadata to db
@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image part in the request"}), 400
    
    image = request.files["image"]
    predicted = request.form.get("predicted")
    correct = request.form.get("correct")
    confidence = request.form.get("confidence")

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(save_path)

    uploads.insert_one({
        "filename": filename,                                                       #picture
        "predicted": predicted,                                                     #label predicted by model
        "correct": correct,                                                         #if correct/no
        "confidence": float(confidence) if confidence else None,                #model confidence
        "timestamp": datetime.now()                                              #upload time
    })
    return {"status": "success", "message": "File uploaded successfully", "filename": filename}


#get model in cloud (keras)
@app.route("/model/latest/keras")
def model_latest_keras():
    # Get all KERAS model files
    keras_files = sorted([
        f for f in os.listdir(MODEL_FOLDER)
        if f.endswith(".keras")
    ])

    if not keras_files:
        return {"error": "No .keras models found"}, 404

    latest = keras_files[-1]
    full_path = os.path.join(MODEL_FOLDER, latest)

    return send_file(full_path, as_attachment=True)

#get model in cloud (json)
@app.route("/model/latest/json")
def model_latest_json():
    # Get all metadata JSON files
    json_files = sorted([
        f for f in os.listdir(MODEL_FOLDER)
        if f.endswith("_metadata.json")
    ])

    if not json_files:
        return {"error": "No metadata found"}, 404

    latest = json_files[-1]
    full_path = os.path.join(MODEL_FOLDER, latest)

    return send_file(full_path, as_attachment=True, mimetype="application/json")

#for human labeling
@app.route("/label", methods=["POST"])
def label_image():
    filename = request.form.get("filename")
    correct_label = request.form.get("correct")

    if not filename or not correct_label:
        return {"status": "error", "message": "Missing fields"}, 400

    uploads.update_one(
        {"filename": filename},
        {"$set": {"correct": correct_label}}
    )

    return {"status": "success", "message": "Label updated successfully"}

#get all unlabeled images
@app.route("/unlabeled")
def unlabeled():
    docs = list(uploads.find({"correct": "unknown"}))
    for d in docs:
        d["_id"] = str(d["_id"])
    return jsonify(docs)

#get pic for html page
@app.route("/uploads/<filename>")
def serve_uploaded_image(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return {"error": "file not found"}, 404
    return send_file(path)

#go to labeler page
@app.route("/labeler")
def labeler():
    return send_file("labeler.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)