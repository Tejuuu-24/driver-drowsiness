import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# -------------------------------------------------
# Base directory (location of this file)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------
# Paths
# -------------------------------------------------
TRAIN_PATH = os.path.join(BASE_DIR, "..", "dataset", "train")
TEST_PATH  = os.path.join(BASE_DIR, "..", "dataset", "test")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

# Create results directory if not exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------
# Image generators
# -------------------------------------------------
train_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)

test_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_data.flow_from_directory(
    TRAIN_PATH,
    target_size=(24, 24),
    color_mode="grayscale",
    batch_size=32,
    class_mode="binary"
)

test_generator = test_data.flow_from_directory(
    TEST_PATH,
    target_size=(24, 24),
    color_mode="grayscale",
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# -------------------------------------------------
# CNN Model
# -------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.Input(shape=(24, 24, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------------------------
# Training
# -------------------------------------------------
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# -------------------------------------------------
# Save Model (recommended format)
# -------------------------------------------------
model.save(os.path.join(BASE_DIR, "eye_model.h5"))
print("Model Saved!")

# -------------------------------------------------
# Accuracy Graph
# -------------------------------------------------
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.legend(["Train", "Validation"])
plt.savefig(os.path.join(RESULTS_DIR, "accuracy.png"))
plt.close()

# -------------------------------------------------
# Loss Graph
# -------------------------------------------------
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.legend(["Train", "Validation"])
plt.savefig(os.path.join(RESULTS_DIR, "loss.png"))
plt.close()

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------
predictions = model.predict(test_generator)
predictions = (predictions > 0.5).astype(int).ravel()

cm = confusion_matrix(test_generator.classes, predictions)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Open", "Closed"],
    yticklabels=["Open", "Closed"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()