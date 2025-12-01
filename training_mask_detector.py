import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, preprocessing, layers, models, utils
from sklearn import preprocessing as sk_pre, model_selection as sk_model, metrics as sk_metrics
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

INIT_LR = 1e-4
EPOCHS = 10
BS = 32

# Load and preprocess dataset by transforming images into arrays and appending labels
dir = "./dataset"
classes = ["with_mask", "without_mask"]
data = []
labels = []

for class_name in classes:
    class_dir = os.path.join(dir, class_name)
    image_paths = list(paths.list_images(class_dir))

    for image_path in image_paths:
        image = preprocessing.image.load_img(image_path)
        image = image.convert("RGB")     # fix palette/transparency images
        image = image.resize((224,224))  # MobileNetV2 size
        image = preprocessing.image.img_to_array(image)
        image = applications.mobilenet_v2.preprocess_input(image)

        data.append(image)
        labels.append(class_name)


# One-hot encode the labels and save the label binarizer for inference
lb = sk_pre.LabelBinarizer()
labels = lb.fit_transform(labels)
with open('label_binarizer.pickle', 'wb') as f:
    pickle.dump(lb, f)
labels = utils.to_categorical(labels)
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split the dataset and construct training and testing sets
(trainX, testX, trainY, testY) = sk_model.train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
aug = preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Load the MobileNetV2 network and models
baseModel = applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
headModel = baseModel.output
headModel = layers.AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = layers.Flatten(name="flatten")(headModel)
headModel = layers.Dense(128, activation="relu")(headModel)
headModel = layers.Dropout(0.5)(headModel)
headModel = layers.Dense(2, activation="softmax")(headModel)
model = models.Model(inputs=baseModel.input, outputs=headModel)

# Freeze the base model layers (as to not update during first training process)
for layer in baseModel.layers:
    layer.trainable = False

optimizer = keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Callbacks: early stop, save best model, reduce LR on plateau
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_mask_detector.keras', monitor='val_loss', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

head = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Evaluate the model using predictions
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(sk_metrics.classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Save the trained model
# Save final model (also best model saved by ModelCheckpoint)
model.save("mask_detector.keras")

# Plot training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), head.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), head.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), head.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), head.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("training_plot.png")

