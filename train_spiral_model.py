# train_spiral_model.py
import os, json, itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----- CONFIG -----
DATA_DIR = "hand_pd_dataset/spiral"    # <-- change if your path is different
IMG_SIZE = 128
BATCH = 32
EPOCHS = 12
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "spiral_pd_model.keras")  # new Keras format
LABELMAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ----- DATA -----
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,                 # 80/20 split from the same folder
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    shear_range=0.05
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    class_mode='binary',
    subset='training',
    batch_size=BATCH,
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    class_mode='binary',
    subset='validation',
    batch_size=BATCH,
    shuffle=False
)

# Save class index mapping (e.g., {"control": 0, "patient": 1})
label_map = {k: int(v) for k, v in train_gen.class_indices.items()}
with open(LABELMAP_PATH, "w") as f:
    json.dump(label_map, f, indent=2)
print("Label map:", label_map)

# ----- MODEL -----
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.35),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks: Early stop + best checkpoint
ckpt = callbacks.ModelCheckpoint(
    MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1
)
es = callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# ----- TRAIN -----
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[ckpt, es]
)

# Save final model (best already saved via checkpoint)
model.save(MODEL_PATH)
print(f"✅ Saved model to {MODEL_PATH}")

# ----- EVALUATE -----
val_gen.reset()
y_true = val_gen.classes
y_prob = model.predict(val_gen).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=[k for k,_ in sorted(label_map.items(), key=lambda x:x[1])]))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(4,3))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(label_map))
plt.xticks(tick_marks, [k for k,_ in sorted(label_map.items(), key=lambda x:x[1])], rotation=45)
plt.yticks(tick_marks, [k for k,_ in sorted(label_map.items(), key=lambda x:x[1])])
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.xlabel('Predicted'); plt.ylabel('True')
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), bbox_inches='tight')
print("📈 Saved confusion matrix to models/confusion_matrix.png")
