import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from patchtst_tf_model import PatchTST, PatchEmbedding, TransformerEncoder

# --------- CONFIGURACIÓN EXPERIMENTO ---------
patch_len = 25
embed_dim = 512
n_layers = 5
dropout_rate = 0.3
batch_size = 128
epochs = 30

# Paths
csv_path = "OIL_CRUDE/Id90/DataSet_lastPoppingColums.csv"
model_save_dir = "Models/"
scaler_save_path = "Models/scaler_inputs.pkl"
experiments_log = "Models/experiments_patchtst_tf.csv"

# Device setup
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# --------- CARGA DATA ---------
df = pd.read_csv(csv_path)
all_columns = df.columns.tolist()
input_cols = all_columns[1:]  # Ignorar Date column
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
Seq_len = 80
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

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# --------- CREAR MODELO ---------
model = PatchTST(seq_len=Seq_len, patch_len=patch_len, input_dim=len(input_cols), embed_dim=embed_dim, n_layers=n_layers, dropout_rate=dropout_rate)

dummy_input = tf.random.normal((1, Seq_len, len(input_cols)))
model(dummy_input)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
)
model.summary()

# Model save path
model_save_path = os.path.join(model_save_dir, f"best_patchtst_patch{patch_len}_embed{embed_dim}.keras")

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

# --------- EVALUACIÓN DIRECCIÓN ---------
# Predicciones validación
val_preds = model.predict(X_val)

# Inverse transform
close_idx = input_cols.index('Close')
close_min = scaler.data_min_[close_idx]
close_max = scaler.data_max_[close_idx]

val_preds_real = val_preds.flatten() * (close_max - close_min) + close_min
val_reals_real = y_val * (close_max - close_min) + close_min

# Direcciones
pred_dirs = np.sign(val_preds_real[1:] - val_preds_real[:-1])
real_dirs = np.sign(val_reals_real[1:] - val_reals_real[:-1])

# Aciertos
correct_dirs = (pred_dirs == real_dirs)
direction_accuracy = correct_dirs.mean()

# Métricas
mae = np.mean(np.abs(val_reals_real - val_preds_real))
rmse = np.sqrt(np.mean((val_reals_real - val_preds_real)**2))

print(f"\n--- Evaluación ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Precisión Dirección: {direction_accuracy*100:.2f}%")

# --------- GUARDAR RESULTADO EXPERIMENTO ---------
os.makedirs(model_save_dir, exist_ok=True)

exp_data = pd.DataFrame([{
    "Patch_len": patch_len,
    "Embed_dim": embed_dim,
    "N_layers": n_layers,
    "Dropout": dropout_rate,
    "Epochs": epochs,
    "Acc_Direccion(%)": round(direction_accuracy*100, 2),
    "MAE": round(mae, 6),
    "RMSE": round(rmse, 6)
}])

if os.path.exists(experiments_log):
    old = pd.read_csv(experiments_log)
    exp_data = pd.concat([old, exp_data], ignore_index=True)

exp_data.to_csv(experiments_log, index=False)
print(f"Resultados guardados en {experiments_log}")

# --------- PLOT TRAINING LOSS ---------
os.makedirs("Plots/", exist_ok=True)

plt.figure(figsize=(12,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.savefig(f"Plots/training_loss_patchtst_patch{patch_len}_embed{embed_dim}.png")
plt.show()
