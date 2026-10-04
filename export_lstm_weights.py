import tensorflow as tf
import numpy as np

MODEL_PATH = "models/crop_lstm.keras"
OUTPUT_PATH = "models/lstm_weights.npz"

print("Loading trained LSTM model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\nModel loaded successfully.")
print("\nModel layers:")

for layer in model.layers:
    print(
        layer.name,
        layer.__class__.__name__,
        layer.get_weights()
        and [w.shape for w in layer.get_weights()]
    )


# ------------------------------------------------------------
# Find LSTM layer
# ------------------------------------------------------------

lstm_layer = None
dense_layers = []

for layer in model.layers:

    if isinstance(layer, tf.keras.layers.LSTM):
        lstm_layer = layer

    if isinstance(layer, tf.keras.layers.Dense):
        dense_layers.append(layer)


if lstm_layer is None:
    raise RuntimeError("LSTM layer not found.")


if len(dense_layers) < 2:
    raise RuntimeError("Expected two Dense layers.")


# ------------------------------------------------------------
# LSTM weights
# ------------------------------------------------------------

lstm_weights = lstm_layer.get_weights()

lstm_kernel = lstm_weights[0]
lstm_recurrent_kernel = lstm_weights[1]
lstm_bias = lstm_weights[2]


# ------------------------------------------------------------
# Dense 1
# ------------------------------------------------------------

dense1_weights = dense_layers[0].get_weights()

dense1_kernel = dense1_weights[0]
dense1_bias = dense1_weights[1]


# ------------------------------------------------------------
# Dense 2
# ------------------------------------------------------------

dense2_weights = dense_layers[1].get_weights()

dense2_kernel = dense2_weights[0]
dense2_bias = dense2_weights[1]


# ------------------------------------------------------------
# Save weights
# ------------------------------------------------------------

np.savez(
    OUTPUT_PATH,

    lstm_kernel=lstm_kernel,

    lstm_recurrent_kernel=lstm_recurrent_kernel,

    lstm_bias=lstm_bias,

    dense1_kernel=dense1_kernel,

    dense1_bias=dense1_bias,

    dense2_kernel=dense2_kernel,

    dense2_bias=dense2_bias
)


print("\nLSTM weights exported successfully!")

print(
    "Saved to:",
    OUTPUT_PATH
)

print("\nWeight shapes:")

print(
    "LSTM kernel:",
    lstm_kernel.shape
)

print(
    "LSTM recurrent kernel:",
    lstm_recurrent_kernel.shape
)

print(
    "LSTM bias:",
    lstm_bias.shape
)

print(
    "Dense 1:",
    dense1_kernel.shape
)

print(
    "Dense 2:",
    dense2_kernel.shape
)