import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from patchtst_tf_model import PatchTST, PatchEmbedding, TransformerEncoder

# Configuration
model_path = "Models/best_patchtst_tf.keras"
scaler_path = "Models/scaler_inputs.pkl"
csv_path = "OIL_CRUDE/Id90/DataSet_lastPoppingColums.csv"
Seq_len = 80

# Load model
model = tf.keras.models.load_model(
    model_path,
    compile=False,
    custom_objects={
        "Custom>PatchTST": PatchTST,
        "Custom>PatchEmbedding": PatchEmbedding,
        "Custom>TransformerEncoder": TransformerEncoder
    }
)

# Load scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load dataset
df = pd.read_csv(csv_path)
all_columns = df.columns.tolist()
input_cols = all_columns[1:]  # Skip Date column

# Normalize inputs
df[input_cols] = scaler.transform(df[input_cols])

# Prepare last 80 days input
input_data = df[input_cols].iloc[-Seq_len:].values
input_data = np.expand_dims(input_data, axis=0)  # Shape (1, 80, num_features)

# Predict
prediction = model.predict(input_data, verbose=0)
predicted_scaled = prediction.flatten()[0]

# Reverse scale only the 'Close' column
close_idx = input_cols.index('Close')  # Index of Close in input columns
close_min = scaler.data_min_[close_idx]
close_max = scaler.data_max_[close_idx]

predicted_real = predicted_scaled * (close_max - close_min) + close_min

print(f"Predicted Close Price for Next Day: {predicted_real:.2f} USD")
