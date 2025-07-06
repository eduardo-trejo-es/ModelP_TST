import tensorflow as tf
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from patchtst_tf_model import PatchTST, PatchEmbedding, TransformerEncoder
from datetime import datetime
import ta

# --------- CONFIGURACIÓN EXPERIMENTO ---------
patch_len = 25
embed_dim = 256
n_layers = 3
dropout_rate = 0.2
batch_size = 32
epochs = 30
exp_num = 41

# Paths
csv_path = "OIL_CRUDE/Id90/DataSet_lastPoppingColums.csv"
model_save_dir = "Models/"
scaler_save_path = "Models/scaler_inputs.pkl"
experiments_log = "Models/experiments_patchtst_tf.csv"

# Device setup
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# --------- CARGA DATA ---------
df = pd.read_csv(csv_path)
df['SMA_Close'] = df['Close'].rolling(window=10).mean().fillna(method='bfill')

# Nuevas features técnicas
df['Return_1D'] = df['Close'].pct_change().fillna(0)
df['SMA_20'] = df['Close'].rolling(window=20).mean().fillna(method='bfill')
df['SMA_Trend'] = df['SMA_20'].diff().fillna(0)
df['Volatility_10'] = df['Close'].rolling(window=10).std().fillna(method='bfill')
rolling_mean = df['Close'].rolling(20).mean()
rolling_std = df['Close'].rolling(20).std()
df['BB_rel_pos'] = ((df['Close'] - rolling_mean) / (2 * rolling_std)).fillna(0)

# Features técnicas adicionales
df['Momentum_20'] = df['Close'] - df['SMA_20']
df['Return_1D_lag1'] = df['Return_1D'].shift(1).fillna(0)
df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi().fillna(50)

# --------- FFT ENERGY RATIO FEATURE ---------
def extract_fft_energy_ratio(series, window=25):
    result = []
    for i in range(len(series)):
        if i < window:
            result.append(0)
            continue
        segment = series[i - window:i]
        fft_vals = np.abs(np.fft.fft(segment))
        energy = np.sum(fft_vals)
        peak_energy = np.max(fft_vals)
        ratio = peak_energy / energy if energy != 0 else 0
        result.append(ratio)
    return result

df['FFT_energy_ratio'] = extract_fft_energy_ratio(df['Close'], window=25)

# Redefinir dirección con umbral mínimo (0.3%)
df['Return_1D'] = df['Close'].pct_change().fillna(0)
df['Target_dir'] = np.where(df['Return_1D'] > 0.003, 1,
                    np.where(df['Return_1D'] < -0.003, 0, np.nan))
df = df.dropna(subset=['Target_dir'])

# Guardar CSV con features técnicas para evaluación posterior
os.makedirs("OIL_CRUDE/Id90/FEATURES_DF", exist_ok=True)
df.to_csv(f"OIL_CRUDE/Id90/FEATURES_DF/DataSet_with_features_exp{exp_num}.csv", index=False)
print(f"Dataset con features guardado en: OIL_CRUDE/Id90/FEATURES_DF/DataSet_with_features_exp{exp_num}.csv")

# --- Distribución de clases y balanceo manual opcional ---
"""
print("Distribución de clases Target_dir (después del filtrado):")
print(df['Target_dir'].value_counts(normalize=True))

# Balanceo manual opcional si hay fuerte desbalance
counts = df['Target_dir'].value_counts()
min_class = counts.min()
df_balanced = pd.concat([
    df[df['Target_dir'] == 0].sample(min_class, random_state=42),
    df[df['Target_dir'] == 1].sample(min_class, random_state=42)
])
df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
print("Distribución balanceada aplicada:")
print(df['Target_dir'].value_counts(normalize=True))
"""

# Selección de columnas de entrada
input_cols = ['Close', 'SMA_Close', 'Return_1D', 'SMA_20', 'SMA_Trend',
              'Volatility_10', 'BB_rel_pos', 'Momentum_20', 'Return_1D_lag1', 'RSI_14', 'FFT_energy_ratio']
target_col = 'Return_1D'

# --------- NORMALIZAR INPUTS Y TARGET ---------
input_scaler = MinMaxScaler()
df[input_cols] = input_scaler.fit_transform(df[input_cols])

target_scaler = None

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
print("y stats:", y.min(), y.max(), np.std(y))

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
    loss='mean_squared_error',
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


# --------- CALCULAR PESOS DE CLASE ---------
"""
from sklearn.utils import class_weight

cw = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
cw_dict = {0: cw[0], 1: cw[1]}
print(f"Pesos aplicados a clases: {cw_dict}")
"""

# --------- ENTRENAR ---------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks,
    # class_weight=cw_dict
)

model.export(model_save_path)

# --------- EVALUACIÓN DIRECCIÓN ---------
# Predicciones validación
val_preds = model.predict(X_val)
print("Val preds (escalados):", val_preds[:10].flatten())

# Inverse transform
val_preds_real = val_preds
val_reals_real = y_val.reshape(-1, 1)

# Evaluación binaria
"""
val_preds_binary = (val_preds > 0.5).astype(int)
correct_dirs = (val_preds_binary.flatten() == val_reals_real.flatten())
direction_accuracy = correct_dirs.mean()
"""
pred_dir = np.sign(val_preds.flatten())
real_dir = np.sign(val_reals_real.flatten())
correct_dirs = (pred_dir == real_dir)
direction_accuracy = correct_dirs.mean()

# Métricas de regresión comentadas (no aplican en clasificación binaria)
# mae = np.mean(np.abs(val_reals_real - val_preds_real))
# rmse = np.sqrt(np.mean((val_reals_real - val_preds_real)**2))

print(f"\n--- Evaluación ---")
# print(f"MAE: {mae:.4f}")
# print(f"RMSE: {rmse:.4f}")
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
    # "MAE": round(mae, 6),
    # "RMSE": round(rmse, 6)
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

# --------- PLOT VAL PREDS vs LABELS (CLASIFICACIÓN BINARIA) ---------
plt.figure(figsize=(12,5))
plt.scatter(range(100), val_reals_real[:100], label='Real (0=baja, 1=sube)', alpha=0.6)
plt.plot(val_preds_real[:100], label='Predicted (prob)', color='orange')
plt.title('Val Predicted Return vs Real Return (primeros 100)')
plt.xlabel('Timestep')
plt.ylabel('Return')
plt.legend()
plt.grid()
plt.savefig(f"Plots/val_preds_vs_labels_exp{exp_num}.png")
plt.show()
