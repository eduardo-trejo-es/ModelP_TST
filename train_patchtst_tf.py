import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pickle  # Para guardar el scaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from patchtst_tf_model import PatchTST

# --------- CONFIGURACIÓN ---------
csv_path = "OIL_CRUDE/Id90/DataSet_lastPoppingColums.csv"
model_save_path = "Models/best_patchtst_tf.keras"
scaler_save_path = "Models/scaler_inputs.pkl"
Seq_len = 80
Patch_len = 25
batch_size = 128
epochs = 50

# --------- DEVICE SETUP ---------
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# --------- CARGA DATA ---------
df = pd.read_csv(csv_path)
all_columns = df.columns.tolist()
input_cols = all_columns[1:]  # Ignorar la columna Date
target_col = 'Close'

# --------- NORMALIZAR INPUTS ---------
scaler = MinMaxScaler()
df[input_cols] = scaler.fit_transform(df[input_cols])

# Guardar scaler
os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
with open(scaler_save_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to: {scaler_save_path}")

# --------- PREPARAR DATA X, y ---------
X, y = [], []
for i in range(len(df) - Seq_len - 1):
    features = df[input_cols].iloc[i:i+Seq_len].values
    target = df[target_col].iloc[i+Seq_len]
    X.append(features)
    y.append(target)

X = np.array(X)
y = np.array(y)

# Train/Validation split
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size)

# --------- CREAR MODELO ---------
model = PatchTST(seq_len=Seq_len, patch_len=Patch_len, input_dim=len(input_cols))

# Build model (forward pass de prueba)
dummy_input = tf.random.normal((1, Seq_len, len(input_cols)))
model(dummy_input)

# --------- COMPILAR MODELO ---------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
)
model.summary()

# --------- CALLBACKS ---------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

# --------- ENTRENAR ---------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks
)

# --------- GRAFICAR Y GUARDAR HISTÓRICO ---------
os.makedirs("Plots/", exist_ok=True)

plt.figure(figsize=(12,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.savefig("Plots/training_loss_patchtst.png")
plt.show()
