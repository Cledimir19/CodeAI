import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#---------------------------
# 1) Positional Encoding Layer
#---------------------------
class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        # pré-computar encoding para todas as posições
        pos = np.arange(seq_len)[:, np.newaxis]
        i   = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2*(i//2)) / np.float32(d_model))
        angle_rads  = pos * angle_rates
        # aplicar sin nos índices pares, cos nos ímpares
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]  # (1, seq_len, d_model)
        self.pos_encoding = tf.cast(pos_encoding, tf.float32)

    def call(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


#---------------------------
# 2) Um bloco de Transformer Encoder
#---------------------------
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        # atenção multi-head (self-attention)
        attn_output = self.mha(x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


#---------------------------
# 3) Construção do Autoencoder
#---------------------------
def build_transformer_autoencoder(
    seq_len=60,
    feature_dim=281,
    d_model=128,
    num_heads=4,
    dff=512,
    num_layers=2,
    dropout_rate=0.1
):
    # Encoder
    inputs = layers.Input(shape=(seq_len, feature_dim))  # (batch, 60, 281)
    # 1) projetar features para dimensão d_model
    x = layers.Dense(d_model)(inputs)
    # 2) somar encoding posicional
    x = PositionalEncoding(seq_len, d_model)(x)
    # 3) empilhar N blocos de TransformerEncoder
    for _ in range(num_layers):
        x = TransformerEncoderBlock(d_model, num_heads, dff, dropout_rate)(x)

    # Representação latente: agregado por pooling
    latent = layers.GlobalAveragePooling1D()(x)  # (batch, d_model)

    # Decoder simples: reconstrói a sequência original
    x = layers.RepeatVector(seq_len)(latent)          # (batch, 60, d_model)
    x = layers.TimeDistributed(layers.Dense(d_model, activation='relu'))(x)
    outputs = layers.TimeDistributed(layers.Dense(feature_dim))(x)  # (batch, 60, 281)

    return Model(inputs, outputs, name="transformer_autoencoder")


#---------------------------
# 4) Instanciar e compilar
#---------------------------
autoencoder = build_transformer_autoencoder(
    seq_len=60,
    feature_dim=281,
    d_model=128,
    num_heads=4,
    dff=512,
    num_layers=2,
    dropout_rate=0.1
)

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse'
)

autoencoder.summary()
