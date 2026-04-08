# ─────────────────────────────────────────────────────────────────────────────
# CNN Image Classification — ResNet50 Transfer Learning
# Based on ic_cnn_fun_Resnet50.ipynb, corrected for Apple M1 Silicon
#
# Fixes vs notebook:
#   • Windows paths removed → dataset auto-downloads via tf.keras.utils
#   • weights="imagenet" replaces weights=None + manual .load_weights()
#   • tf.keras.utils.image_dataset_from_directory (modern API)
#   • Model saved as .keras (native format, not legacy .h5)
#   • M1 Metal GPU detection added
#   • Evaluation: classification report + confusion matrix added
#   • Prediction plot: preprocessing undone so images display correctly
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 0: Imports & M1 Metal Check ─────────────────────────────────────────
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("Metal GPU:", gpus if gpus else "Not found — running on CPU")
print("  → pip install tensorflow-metal  for M1 GPU acceleration")

# ── Step 1: Paths & Hyperparams ───────────────────────────────────────────────
# Set DATA_ROOT to your local flower_photos folder, or leave None to auto-download
DATA_ROOT  = None   # e.g. "/Users/you/data/flower_photos"
IMG_HEIGHT = 180
IMG_WIDTH  = 180
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

# Auto-download TF Flowers dataset (~218 MB, cached after first run)
if DATA_ROOT is None:
    import pathlib
    url       = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir  = tf.keras.utils.get_file("flower_photos", origin=url, untar=True)
    DATA_ROOT = str(pathlib.Path(data_dir) / "flower_photos")
    print(f"Dataset: {DATA_ROOT}")

# ── Step 2: Load Raw (Unbatched) Datasets ─────────────────────────────────────
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_ROOT,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=None,        # unbatched — batching happens in prepare()
    label_mode="int"
)
raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_ROOT,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=None,
    label_mode="int"
)

NUM_CLASSES = len(raw_train_ds.class_names)
class_names = raw_train_ds.class_names
print("Classes:", class_names, "| Num classes:", NUM_CLASSES)

# ── Step 3: Augmentation & Preprocessing ─────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
], name="data_augmentation")

def preprocess(image, label, augment=False):
    image = tf.cast(image, tf.float32)
    if augment:
        image = data_augmentation(image)
    image = resnet_preprocess(image)    # ResNet BGR mean-subtraction
    return image, label

# ── Step 4: Build tf.data Pipelines ──────────────────────────────────────────
def prepare(ds, shuffle=False, augment=False):
    if shuffle:
        ds = ds.shuffle(1000, seed=42)
    ds = ds.map(lambda x, y: preprocess(x, y, augment), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = prepare(raw_train_ds, shuffle=True, augment=True)
val_ds   = prepare(raw_val_ds)

# ── Step 5: Build Model ───────────────────────────────────────────────────────
image_inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="Image_input")

# ResNet50 backbone — weights="imagenet" downloads automatically (no manual .h5 needed)
base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=image_inputs)
base_model.trainable = False    # Freeze backbone, train head only

# Feature extraction head
image_net = layers.GlobalAveragePooling2D(name="gap")(base_model.output)
image_net = layers.Dense(128, activation="relu")(image_net)
image_net = layers.Dropout(0.5, name="features")(image_net)

image_features = layers.Dense(64, activation="relu")(image_net)
image_features = layers.LayerNormalization(axis=-1)(image_features)

image_output = layers.Dense(NUM_CLASSES, activation="softmax", name="image_out")(image_features)

final_model = Model(inputs=[image_inputs], outputs=[image_output], name="resnet50_flowers")
final_model.summary()

# ── Step 6: Compile & Callbacks ───────────────────────────────────────────────
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

ckpt_path = os.path.join(SAVE_DIR, "ic_cnn_fun_Resnet50.keras")   # .keras not .h5
callbacks = [
    ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
]

# ── Step 7: Train ─────────────────────────────────────────────────────────────
print("\n── Training (backbone frozen) ──")
history = final_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    callbacks=callbacks
)

model = final_model

# ── Step 8: Evaluate ──────────────────────────────────────────────────────────
print("\n── Evaluation ──")
val_loss, val_acc = model.evaluate(val_ds, verbose=1)
print(f"\nVal Loss     : {val_loss:.4f}")
print(f"Val Accuracy : {val_acc:.4f}")

# Collect all predictions for classification report + confusion matrix
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)
axes[1].plot(history.history["loss"],         label="Train")
axes[1].plot(history.history["val_loss"],     label="Val")
axes[1].set_title("Loss");     axes[1].legend(); axes[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
plt.show()

# ── Step 9: Visualize Predictions ────────────────────────────────────────────
for images, labels in val_ds.take(1):
    preds = model.predict(images, verbose=0)
    pred_labels = tf.argmax(preds, axis=1).numpy()

    # Undo ResNet preprocessing (add back means, BGR→RGB) so images display correctly
    display = images.numpy() + [103.939, 116.779, 123.68]
    display = display[..., ::-1]
    display = np.clip(display, 0, 255).astype("uint8")

    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(display[i])
        true_name = class_names[labels[i]]
        pred_name = class_names[pred_labels[i]]
        color = "green" if true_name == pred_name else "red"
        plt.title(f"T:{true_name}\nP:{pred_name}", color=color, fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "sample_predictions.png"), dpi=150)
    plt.show()
    break

print(f"\nModel saved : {ckpt_path}")
print(f"Plots saved : {SAVE_DIR}")
