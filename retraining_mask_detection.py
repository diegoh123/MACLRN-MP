import os
import json
import random
import pickle
import numpy as np
from pymongo import MongoClient
from imutils import paths
from sklearn.utils import shuffle
from sklearn import preprocessing as sk_pre, model_selection as sk_model, metrics as sk_metrics
from tensorflow import keras
from tensorflow.keras import applications, preprocessing, layers, models, utils
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

#connect to db
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)

db = client["maclrn_db"]
uploads_col = db["uploads"]

DATASET_DIR = "./dataset"
UPLOAD_DIR = "tier2_cloud/cloud_storage/uploads"
MODEL_DIR = "tier2_cloud/cloud_storage/models"

CLASSES = ["with_mask", "without_mask", "improper_mask"]

EPOCHS = 10
BS = 32
INIT_LR = 1e-4


#get from dataset
def load_kaggle_dataset():
    print("[INFO] Loading Kaggle dataset...")

    images, labels = [], []

    for cls in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(class_dir):
            continue

        for path in paths.list_images(class_dir):
            img = preprocessing.image.load_img(path, target_size=(224, 224))
            img = img.convert("RGB")
            img = preprocessing.image.img_to_array(img)
            img = applications.mobilenet_v2.preprocess_input(img)

            images.append(img)
            labels.append(cls)

    print(f"[INFO] Kaggle dataset loaded: {len(images)} samples")
    return images, labels


#get from db
def load_realworld_labeled():
    print("[INFO] Loading REAL-WORLD labeled images...")

    images, labels = [], []

    docs = uploads_col.find({
        "correct": {"$in": CLASSES}  # ignore 'unknown'
    })

    count = 0

    for doc in docs:
        filename = doc.get("filename")
        label = doc.get("correct")

        if not filename or not label:
            continue

        path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(path):
            continue

        img = preprocessing.image.load_img(path, target_size=(224, 224))
        img = img.convert("RGB")
        img = preprocessing.image.img_to_array(img)
        img = applications.mobilenet_v2.preprocess_input(img)

        images.append(img)
        labels.append(label)
        count += 1

    print(f"[INFO] Loaded {count} REAL-WORLD labeled samples")
    return images, labels


#balance datasets
def balance_datasets(kaggle_imgs, kaggle_labels, real_imgs, real_labels):

    if len(real_imgs) == 0:
        print("[WARN] No real-world data -> using full Kaggle dataset.")
        return kaggle_imgs, kaggle_labels

    size = min(len(kaggle_imgs), len(real_imgs))

    kag_idx = random.sample(range(len(kaggle_imgs)), size)
    real_idx = random.sample(range(len(real_imgs)), size)

    kaggle_imgs_bal = [kaggle_imgs[i] for i in kag_idx]
    kaggle_labels_bal = [kaggle_labels[i] for i in kag_idx]

    real_imgs_bal = [real_imgs[i] for i in real_idx]
    real_labels_bal = [real_labels[i] for i in real_idx]

    print(f"[INFO] BALANCED MODE -> {size} Kaggle + {size} Real samples")

    # combine
    images = kaggle_imgs_bal + real_imgs_bal
    labels = kaggle_labels_bal + real_labels_bal

    # shuffle
    images, labels = shuffle(images, labels, random_state=42)

    return images, labels


#retraining
def train_balanced():

    kaggle_imgs, kaggle_labels = load_kaggle_dataset()
    real_imgs, real_labels = load_realworld_labeled()

    all_imgs, all_labels = balance_datasets(
        kaggle_imgs, kaggle_labels, real_imgs, real_labels
    )

    all_imgs = np.array(all_imgs)
    all_labels = np.array(all_labels)

    lb = sk_pre.LabelBinarizer()
    lb.fit(all_labels)

    with open("label_binarizer.pickle", "wb") as f:
        pickle.dump(lb, f)
        
    (trainX, testX, trainY, testY) = sk_model.train_test_split(
        all_imgs, all_labels, test_size=0.20, random_state=42, stratify=all_labels
    )

    trainY = lb.transform(trainY)
    testY = lb.transform(testY)

    aug = preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    #make model
    base = applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    x = layers.AveragePooling2D(pool_size=(7, 7))(base.output)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(lb.classes_), activation="softmax")(x)

    model = models.Model(inputs=base.input, outputs=outputs)

    for L in base.layers:
        L.trainable = False

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(INIT_LR),
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, monitor="val_loss", restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    print("[INFO] Training model...")
    history = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Get final metrics
    final_val_loss = history.history["val_loss"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    final_train_loss = history.history["loss"][-1]
    final_train_acc = history.history["accuracy"][-1]

    print("\n" + "="*50)
    print("[INFO] Training Complete!")
    print(f"[INFO] Final Training Accuracy: {final_train_acc:.4f}")
    print(f"[INFO] Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"[INFO] Final Training Loss: {final_train_loss:.4f}")
    print(f"[INFO] Final Validation Loss: {final_val_loss:.4f}")
    print("="*50 + "\n")

    # Evaluate the model on test set
    print("[INFO] Evaluating model on test set...")
    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)
    print("\n" + sk_metrics.classification_report(
    testY.argmax(axis=1), predIdxs, target_names=lb.classes_
    ))
    

    # Save model with versioning
    os.makedirs(MODEL_DIR, exist_ok=True)
    existing = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("mask_detector_v") and f.endswith(".keras")])

    version = 1 if not existing else int(existing[-1].split("_v")[1].split(".")[0]) + 1

    filename = f"mask_detector_v{version}.keras"
    save_path = os.path.join(MODEL_DIR, filename)

    model.save(save_path)
    print(f"[INFO] Saved NEW MODEL -> {save_path}")

    # Save metadata with metrics
    metadata = {
        "version": version,
        "filename": filename,
        "val_accuracy": float(final_val_acc),
        "val_loss": float(final_val_loss),
        "train_accuracy": float(final_train_acc),
        "train_loss": float(final_train_loss),
        "epochs_trained": len(history.history["loss"]),
        "kaggle_samples": len(kaggle_imgs),
        "real_samples": len(real_imgs),
        "total_samples": len(all_imgs)
    }

    metadata_path = os.path.join(MODEL_DIR, f"mask_detector_v{version}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[INFO] Saved metadata -> {metadata_path}")

    # Plot training history
    N = len(history.history["loss"])
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
    plt.title(f"Training Loss and Accuracy - Model v{version}")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    
    plot_path = os.path.join(MODEL_DIR, f"mask_detector_v{version}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"[INFO] Saved training plot -> {plot_path}")

    return metadata


if __name__ == "__main__":
    train_balanced()