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
db = client['mask_detection_db']
uploads = db['uploads']

try:
    client.admin.command('ping')
    print("Connected to MongoDB successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

@app.route("/ping")
def ping():
    return {"status": "success", "message": "pong!"}

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

#update model in cloud -> fix this so that if latest model flops get previously better one
@app.route("/model/latest")
def model_latest():
    models = sorted(os.listdir(MODEL_FOLDER))
    if not models:
        return {"error": "no models found"}, 404
    
    latest = models[-1]
    return send_file(os.path.join(MODEL_FOLDER, latest), as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)