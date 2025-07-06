import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from patchtst_tf_model import PatchTST, PatchEmbedding, TransformerEncoder


import re
exp = 41
# --------- CONFIGURACIÓN ---------
model_path = "Models/patchtst_exp"+str(exp)
input_scaler_path = "Models/scaler_inputs.pkl"
target_scaler_path = "Models/scaler_target.pkl"
csv_path = f"OIL_CRUDE/Id90/FEATURES_DF/DataSet_with_features_exp{exp}.csv"
Seq_len = 80
Prediction_days = 200
save_predictions_path = "DataSet_with_features_exp"+str(exp)+".csv"


# --------- CARGA MODELO ---------
from keras.layers import TFSMLayer

model = tf.keras.Sequential([
    TFSMLayer(model_path, call_endpoint="serving_default")
])

# --------- CARGA SCALER ---------
with open(input_scaler_path, 'rb') as f:
    input_scaler = pickle.load(f)

# --------- CARGA DATA ---------
df = pd.read_csv(csv_path)
all_columns = df.columns.tolist()
input_cols = input_scaler.feature_names_in_.tolist()

# Normalizar inputs
missing_cols = [col for col in input_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Faltan columnas en el dataset: {missing_cols}")

df[input_cols] = input_scaler.transform(df[input_cols])

# --------- PREPARAR DATA ---------
predictions = []
real_prices = []

for i in range(-(Prediction_days + Seq_len), -Seq_len):
    input_data = df[input_cols].iloc[i:i+Seq_len].values
    input_data = np.expand_dims(input_data, axis=0)  # (1, 80, num_features)

    pred_value = model.predict(input_data, verbose=0)

    # Extraer el valor real del diccionario si es necesario
    if isinstance(pred_value, dict):
        pred_array = list(pred_value.values())[0]
    else:
        pred_array = pred_value

    pred = pred_array.flatten()[0]
    print(f"Pred normalizado: {pred}")

    pred_close_real = pred
    real_close_real = df['Close'].iloc[i + Seq_len]
    print(f"Predicción en escala real (sin normalizar): {pred_close_real}")

    predictions.append(pred_close_real)
    real_prices.append(real_close_real)

# Convert predictions and real prices to numpy arrays
predictions = np.array(predictions)
real_prices = np.array(real_prices)

# --------- MÉTRICAS ---------
mae = np.mean(np.abs(real_prices - predictions))
rmse = np.sqrt(np.mean((real_prices - predictions) ** 2))

print(f"\nEvaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# --------- PRECISIÓN DIRECCIONAL ---------
direction_real = np.sign(real_prices[1:] - real_prices[:-1])
direction_pred = np.sign(predictions[1:] - predictions[:-1])
direction_real_full = np.append([np.nan], direction_real)
direction_pred_full = np.append([np.nan], direction_pred)
direction_accuracy = np.mean(direction_real == direction_pred)

print(f"Directional Accuracy: {direction_accuracy*100:.2f}%")

comparison = pd.DataFrame({
    'Real': real_prices,
    'Pred': predictions,
    'Dir_Real': direction_real_full,
    'Dir_Pred': direction_pred_full
})
print("\nPrimeros 10 ejemplos de comparación:")
print(comparison.head(10))

diff_real = np.abs(real_prices[1:] - real_prices[:-1])
mask_big_moves = diff_real > np.percentile(diff_real, 75)
accuracy_big_moves = np.mean(direction_real[mask_big_moves] == direction_pred[mask_big_moves])
print(f"Directional accuracy on large moves: {accuracy_big_moves*100:.2f}%")

# --------- GUARDAR CSV ---------
predictions_df = pd.DataFrame({
    'Real_Close': real_prices,
    'Predicted_Close': predictions,
    'Absolute_Error': np.abs(real_prices - predictions)
})
predictions_df['Direction_Real'] = direction_real_full
predictions_df['Direction_Pred'] = direction_pred_full

save_dir = os.path.dirname(save_predictions_path)
if save_dir:
    os.makedirs(save_dir, exist_ok=True)
predictions_df.to_csv(save_predictions_path, index=False)
print(f"Predictions saved to: {save_predictions_path}")

# --------- PLOT DIRECTIONS ---------
plt.figure(figsize=(12, 5))
plt.plot(direction_real[:100], label='Dirección Real', marker='o')
plt.plot(direction_pred[:100], label='Dirección Predicha', marker='x')
plt.title('Dirección Real vs Predicha (últimos 100 días)')
plt.xlabel('Día')
plt.ylabel('Dirección (signo)')
plt.legend()
plt.grid()
plt.savefig(f"Plots/direction_comparison_patchtst_exp{exp}.png")
plt.show()

# --------- PLOT ---------
os.makedirs("Plots/", exist_ok=True)

plt.figure(figsize=(14, 6))
plt.plot(real_prices, label='Real Close Price', color='black')
plt.plot(predictions, label='Predicted Close Price', color='blue', linestyle='--')
plt.title('PatchTST (TensorFlow) - Predictions vs Real Close Prices (Last 200 days)')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.savefig(f"Plots/predictions_patchtst_exp{exp}.png")
plt.show()
