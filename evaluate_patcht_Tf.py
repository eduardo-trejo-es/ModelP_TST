import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from patchtst_tf_model import PatchTST, PatchEmbedding, TransformerEncoder

# --------- CONFIGURACIÓN ---------
model_path = "Models/best_patchtst_tf.keras"
scaler_save_path = "Models/scaler_inputs.pkl"
csv_path = "OIL_CRUDE/Id90/DataSet_lastPoppingColums.csv"
Seq_len = 80
Prediction_days = 200
save_predictions_path = "Models/predictions_patchtst_tf.csv"

# --------- CARGA MODELO ---------
model = tf.keras.models.load_model(
    model_path,
    compile=False,
    custom_objects={
        "Custom>PatchTST": PatchTST,
        "Custom>PatchEmbedding": PatchEmbedding,
        "Custom>TransformerEncoder": TransformerEncoder
    }
)

# --------- CARGA SCALER ---------
with open(scaler_save_path, 'rb') as f:
    scaler = pickle.load(f)

# --------- CARGA DATA ---------
df = pd.read_csv(csv_path)
all_columns = df.columns.tolist()
input_cols = all_columns[1:]  # Ignorar Date column

# Normalizar inputs
df[input_cols] = scaler.transform(df[input_cols])

# --------- PREPARAR DATA ---------
predictions = []
real_prices = []

for i in range(-(Prediction_days + Seq_len), -Seq_len):
    input_data = df[input_cols].iloc[i:i+Seq_len].values
    input_data = np.expand_dims(input_data, axis=0)  # (1, 80, num_features)

    pred = model.predict(input_data, verbose=0)

    # Inverse transform prediction
    pred_close_scaled = np.zeros((1, len(input_cols)))
    pred_close_scaled[0, input_cols.index('Close')] = pred.flatten()[0]
    pred_close_real = scaler.inverse_transform(pred_close_scaled)[0, input_cols.index('Close')]

    # Inverse transform real close
    real_input_scaled = np.zeros((1, len(input_cols)))
    real_input_scaled[0, :] = df[input_cols].iloc[i + Seq_len].values
    real_close_real = scaler.inverse_transform(real_input_scaled)[0, input_cols.index('Close')]

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
