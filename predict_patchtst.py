import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from patchtst_tf_model import PatchTST, PatchEmbedding, TransformerEncoder

# Configuration
model_path = "Models/best_patchtst_patch30_embed512.keras"
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

# Reverse scale using scaler
predicted_scaled = prediction.flatten()
full_input = np.zeros((1, len(input_cols)))
full_input[0, input_cols.index('Close')] = predicted_scaled[0]

predicted_real_close = scaler.inverse_transform(full_input)[0, input_cols.index('Close')]

print(f"Predicted Close Price for Next Day: {predicted_real_close:.2f} USD")
