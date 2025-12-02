import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, preprocessing, layers, models, utils
from sklearn import preprocessing as sk_pre, model_selection as sk_model, metrics as sk_metrics
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pickle

INIT_LR = 1e-4
EPOCHS = 10
BS = 32

DATASET_DIR = "./dataset"
MODEL_DIR = "tier2_cloud/cloud_storage/models"
CLASSES = ["with_mask", "without_mask", "improper_mask"]

print("[INFO] Loading and preprocessing dataset...")

# Load and preprocess dataset by transforming images into arrays and appending labels
data = []
labels = []

for class_name in CLASSES:
    class_dir = os.path.join(DATASET_DIR, class_name)
    
    if not os.path.isdir(class_dir):
        print(f"[WARN] Skipping missing directory: {class_dir}")
        continue
    
    image_paths = list(paths.list_images(class_dir))
    print(f"[INFO] Loading {len(image_paths)} images from {class_name}")

    for image_path in image_paths:
        image = preprocessing.image.load_img(image_path)
        image = image.convert("RGB")     # fix palette/transparency images
        image = image.resize((224, 224))  # MobileNetV2 size
        image = preprocessing.image.img_to_array(image)
        image = applications.mobilenet_v2.preprocess_input(image)

        data.append(image)
        labels.append(class_name)

print(f"[INFO] Total samples loaded: {len(data)}")

# One-hot encode the labels and save the label binarizer for inference
lb = sk_pre.LabelBinarizer()
labels = lb.fit_transform(labels)

# Save the label binarizer for inference
with open('label_binarizer.pickle', 'wb') as f:
    pickle.dump(lb, f)
print("[INFO] Saved label_binarizer.pickle")

# For 2 classes, LabelBinarizer returns shape (N, 1) so we expand.
# For 3+ classes, it already returns one-hot of shape (N, num_classes).
if len(lb.classes_) == 2:
    labels = utils.to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

print(f"[INFO] Classes: {lb.classes_}")
print(f"[INFO] Data shape: {data.shape}")
print(f"[INFO] Labels shape: {labels.shape}")

# Split the dataset and construct training and testing sets
(trainX, testX, trainY, testY) = sk_model.train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

print(f"[INFO] Training samples: {len(trainX)}")
print(f"[INFO] Testing samples: {len(testX)}")

aug = preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Load the MobileNetV2 network and build model
print("[INFO] Building model...")
baseModel = applications.MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

headModel = baseModel.output
headModel = layers.AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = layers.Flatten(name="flatten")(headModel)
headModel = layers.Dense(128, activation="relu")(headModel)
headModel = layers.Dropout(0.5)(headModel)
headModel = layers.Dense(len(lb.classes_), activation="softmax")(headModel)
model = models.Model(inputs=baseModel.input, outputs=headModel)

# Freeze the base model layers (as to not update during first training process)
for layer in baseModel.layers:
    layer.trainable = False

optimizer = keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Callbacks: early stop, reduce LR on plateau
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

print("[INFO] Training model...")
history = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
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

# Evaluate the model using predictions
print("[INFO] Evaluating model on test set...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print("\n" + sk_metrics.classification_report(
    testY.argmax(axis=1), predIdxs, target_names=lb.classes_
))

# Save the trained model with versioning
os.makedirs(MODEL_DIR, exist_ok=True)

# Check for existing versions
existing = sorted([
    f for f in os.listdir(MODEL_DIR) 
    if f.startswith("mask_detector_v") and f.endswith(".keras")
])

version = 1 if not existing else int(existing[-1].split("_v")[1].split(".")[0]) + 1

filename = f"mask_detector_v{version}.keras"
save_path = os.path.join(MODEL_DIR, filename)

model.save(save_path)
print(f"[INFO] Saved model -> {save_path}")

# Also save as the default fallback model
model.save("mask_detector.keras")
print(f"[INFO] Saved fallback model -> mask_detector.keras")

# Save metadata with metrics
metadata = {
    "version": version,
    "filename": filename,
    "val_accuracy": float(final_val_acc),
    "val_loss": float(final_val_loss),
    "train_accuracy": float(final_train_acc),
    "train_loss": float(final_train_loss),
    "epochs_trained": len(history.history["loss"]),
    "total_samples": len(data),
    "train_samples": len(trainX),
    "test_samples": len(testX),
    "classes": lb.classes_.tolist(),
    "model_type": "initial_training"
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

# Save plot with version number
plot_path = os.path.join(MODEL_DIR, f"mask_detector_v{version}_plot.png")
plt.savefig(plot_path)
print(f"[INFO] Saved training plot -> {plot_path}")

# Also save as default plot
plt.savefig("training_plot.png")
print(f"[INFO] Saved default plot -> training_plot.png")

plt.close()

print("\n[✓] Training pipeline complete!")