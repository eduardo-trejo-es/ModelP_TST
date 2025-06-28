import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from patchtst_tf_model import PatchTST, PatchEmbedding, TransformerEncoder
from datetime import datetime

# --------- CONFIGURACIÓN EXPERIMENTO ---------
patch_len = 15
embed_dim = 512
n_layers = 4
dropout_rate = 0.0
batch_size = 64
epochs = 30

exp_num=9

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

# --------- NORMALIZAR INPUTS Y TARGET ---------
input_scaler = MinMaxScaler()
df[input_cols] = input_scaler.fit_transform(df[input_cols])

target_scaler = MinMaxScaler()
df[[target_col]] = target_scaler.fit_transform(df[[target_col]])

# Guardar scalers
os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
with open("Models/scaler_inputs.pkl", 'wb') as f:
    pickle.dump(input_scaler, f)
with open("Models/scaler_target.pkl", 'wb') as f:
    pickle.dump(target_scaler, f)
print("Scalers saved to: Models/scaler_inputs.pkl and scaler_target.pkl")

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
    loss='mae',
    metrics=['mae']
)
model.summary()


fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

os.makedirs(model_save_dir, exist_ok=True)

"""if os.path.exists(experiments_log):
    old = pd.read_csv(experiments_log)
    exp_num = len(old) + 1
else:
    exp_num = 1"""

# Model save path
model_save_path = os.path.join(model_save_dir, f"patchtst_exp{exp_num}")

# --------- CALLBACKS ---------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
]

# --------- ENTRENAR ---------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks
)

model.export(model_save_path)

# --------- EVALUACIÓN DIRECCIÓN ---------
# Predicciones validación
val_preds = model.predict(X_val)

# Inverse transform
val_preds_real = target_scaler.inverse_transform(val_preds)
val_reals_real = target_scaler.inverse_transform(y_val.reshape(-1, 1))

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


exp_data = pd.DataFrame([{
    "Experimento": exp_num,
    "Fecha": fecha,
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
plt.savefig(f"Plots/training_loss_exp{exp_num}.png")
plt.show()
