import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from patchtst_tf_model import PatchTST, PatchEmbedding, TransformerEncoder

exp=9
# --------- CONFIGURACIÓN ---------
model_path = "Models/patchtst_exp"+str(exp)
input_scaler_path = "Models/scaler_inputs.pkl"
target_scaler_path = "Models/scaler_target.pkl"
csv_path = "OIL_CRUDE/Id90/DataSet_lastPoppingColums.csv"
Seq_len = 80
Prediction_days = 200
save_predictions_path = "PreditionEval/predictions_patchtst_exp"+str(exp)+"_tf.csv"

# --------- CARGA MODELO ---------
from keras.layers import TFSMLayer

model = tf.keras.Sequential([
    TFSMLayer(model_path, call_endpoint="serving_default")
])

# --------- CARGA SCALER ---------
with open(input_scaler_path, 'rb') as f:
    input_scaler = pickle.load(f)

with open(target_scaler_path, 'rb') as f:
    target_scaler = pickle.load(f)

# --------- CARGA DATA ---------
df = pd.read_csv(csv_path)
all_columns = df.columns.tolist()
input_cols = all_columns[1:]  # Ignorar Date column

# Normalizar inputs
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

    # Inverse transform prediction
    pred_close_real = target_scaler.inverse_transform([[pred]])[0][0]
    print(f"Predicción en escala real: {pred_close_real}")
    # Inverse transform real close
    real_close_real = target_scaler.inverse_transform(
        [[df['Close'].iloc[i + Seq_len]]])[0][0]

    predictions.append(pred_close_real)
    real_prices.append(real_close_real)

# Convert predictions and real prices to numpy arrays
predictions = np.array(predictions)
real_prices = np.array(real_prices)

# --------- GUARDAR CSV ---------
predictions_df = pd.DataFrame({
    'Real_Close': real_prices,
    'Predicted_Close': predictions,
    'Absolute_Error': np.abs(real_prices - predictions)
})

os.makedirs(os.path.dirname(save_predictions_path), exist_ok=True)
predictions_df.to_csv(save_predictions_path, index=False)
print(f"Predictions saved to: {save_predictions_path}")

# --------- MÉTRICAS ---------
mae = np.mean(np.abs(real_prices - predictions))
rmse = np.sqrt(np.mean((real_prices - predictions) ** 2))

print(f"\nEvaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

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
plt.savefig("Plots/predictions_patchtst.png")
plt.show()
