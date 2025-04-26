import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="Custom", name="PatchEmbedding")
class PatchEmbedding(layers.Layer):
    def __init__(self, seq_len, patch_len, input_dim, embed_dim,**kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_patches = seq_len // patch_len

        self.projection = layers.Dense(embed_dim)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, input_dim)
        batch_size = tf.shape(inputs)[0]
        x = inputs[:, :self.num_patches * self.patch_len, :]  # Trim if needed
        x = tf.reshape(x, (batch_size, self.num_patches, self.patch_len * self.input_dim))
        x = self.projection(x)
        return x  # (batch_size, num_patches, embed_dim)

@tf.keras.utils.register_keras_serializable(package="Custom", name="TransformerEncoder")
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.3,**kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

@tf.keras.utils.register_keras_serializable(package="Custom", name="PatchTST")
class PatchTST(tf.keras.Model):
    def __init__(self, seq_len=80, patch_len=25, input_dim=12, embed_dim=128, n_layers=2, n_heads=4, dropout_rate=0.1,**kwargs):
        super().__init__(**kwargs)
        self.embedding = PatchEmbedding(seq_len, patch_len, input_dim, embed_dim)
        self.encoder_layers = [
            TransformerEncoder(embed_dim, n_heads, embed_dim * 4, dropout_rate)
            for _ in range(n_layers)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.head = layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(1e-4))

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        for encoder in self.encoder_layers:
            x = encoder(x, training=training)
        x = tf.reduce_mean(x, axis=1)  # Global average pooling
        x = self.norm(x)
        return self.head(x)

# Example to instantiate
# model = PatchTST(seq_len=80, patch_len=25, input_dim=12)
# output = model(tf.random.normal((8, 80, 12)))
# print(output.shape)  # -> (8, 1)
