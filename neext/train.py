# === Imports ===
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, BatchNormalization,
                                     Bidirectional, GRU, Dense, Dropout,
                                     Multiply, GlobalAveragePooling1D, Attention)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

# === Path Setup ===
X_PATH = "/content/drive/MyDrive/Research/ModelJULY24/Data/X_final.npy"
y_PATH = "/content/drive/MyDrive/Research/ModelJULY24/Data/y_final.npy"
MODEL_SAVE_PATH = "/content/drive/MyDrive/Research/ModelJULY24/Model/final_model_attention.keras"
SCALER_SAVE_PATH = "/content/drive/MyDrive/Research/ModelJULY24/Scaler/scaler.save"
PLOT_SAVE_PATH = "/content/drive/MyDrive/Research/ModelJULY24/Scaler/training_plot.png"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)

# === Load Dataset ===
X = np.load(X_PATH)
y = np.load(y_PATH)

# === Scale Data ===
X_reshaped = X.reshape(-1, X.shape[-1])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(-1, 10, 237)
joblib.dump(scaler, SCALER_SAVE_PATH)

# === Split ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# === Convert Labels to Categorical ===
y_train_cat = to_categorical(y_train, num_classes=4)
y_val_cat = to_categorical(y_val, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

# === Class Weights ===
class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights_array))

# === Callbacks ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# === Model Architecture ===
input_layer = Input(shape=(10, 237))

x = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.0005))(input_layer)
x = MaxPooling1D(pool_size=2)(x)
x = BatchNormalization()(x)

x = Bidirectional(GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.0005)))(x)
x = Dropout(0.3)(x)
x = Bidirectional(GRU(32, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.0005)))(x)
x = Dropout(0.3)(x)

attn = Attention()([x, x])
x = Multiply()([x, attn])
x = GlobalAveragePooling1D()(x)

x = Dense(128, activation='relu', kernel_regularizer=l2(0.0005))(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.0005))(x)
x = Dropout(0.3)(x)

output = Dense(4, activation='softmax')(x)

# === Compile Model ===
model = Model(inputs=input_layer, outputs=output)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
model.summary()

# === Train Model ===
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint, lr_scheduler],
    class_weight=class_weights,
    verbose=1
)

# === Evaluate ===
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✅ Final Test Accuracy: {test_acc * 100:.2f}% | Loss: {test_loss:.4f}")

# === Plot ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH)
plt.show()