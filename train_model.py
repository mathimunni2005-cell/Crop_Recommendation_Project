import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================================
# SETTINGS
# ==========================================================

DATA_PATH = "datasets/crop_recommendation.csv"

MODEL_DIR = "models"

SEQUENCE_LENGTH = 3


# ==========================================================
# CREATE MODEL DIRECTORY
# ==========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==========================================================
# LOAD DATASET
# ==========================================================

data = pd.read_csv(
    DATA_PATH
)

print(
    "Dataset shape:",
    data.shape
)


# ==========================================================
# FEATURES
# ==========================================================

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]


X = data[features].values

y = data["label"].values


# ==========================================================
# LABEL ENCODER
# ==========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print(
    "Crop classes:",
    label_encoder.classes_
)


# ==========================================================
# SCALE FEATURES
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================================
# CREATE SEQUENCES
# ==========================================================

X_sequences = []

y_sequences = []


for crop_class in np.unique(y_encoded):

    crop_indices = np.where(
        y_encoded == crop_class
    )[0]

    crop_data = X_scaled[
        crop_indices
    ]

    crop_labels = y_encoded[
        crop_indices
    ]


    for i in range(
        len(crop_data) - SEQUENCE_LENGTH + 1
    ):

        X_sequences.append(
            crop_data[
                i:i + SEQUENCE_LENGTH
            ]
        )

        y_sequences.append(
            crop_labels[i + SEQUENCE_LENGTH - 1]
        )


X_sequences = np.array(
    X_sequences
)

y_sequences = np.array(
    y_sequences
)


print(
    "Sequence shape:",
    X_sequences.shape
)


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_sequences,

    y_sequences,

    test_size=0.25,

    random_state=42,

    stratify=y_sequences
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ==========================================================
# LIGHTWEIGHT LSTM MODEL
# ==========================================================

model = Sequential()


model.add(
    Input(
        shape=(
            SEQUENCE_LENGTH,
            len(features)
        )
    )
)


model.add(
    LSTM(
        16,
        return_sequences=False
    )
)


model.add(
    Dropout(0.10)
)


model.add(
    Dense(
        8,
        activation="relu"
    )
)


model.add(
    Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
)


# ==========================================================
# COMPILE MODEL
# ==========================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ==========================================================
# MODEL SUMMARY
# ==========================================================

model.summary()


# ==========================================================
# EARLY STOPPING
# ==========================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)


# ==========================================================
# TRAIN MODEL
# ==========================================================

history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=100,

    batch_size=4,

    callbacks=[
        early_stopping
    ],

    verbose=1

)


# ==========================================================
# TEST ACCURACY
# ==========================================================

loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)


print()
print(
    "=========================================="
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

print(
    "=========================================="
)


# ==========================================================
# SAVE LSTM MODEL
# ==========================================================

model.save(
    "models/crop_lstm.keras"
)


# ==========================================================
# SAVE SCALER
# ==========================================================

with open(
    "models/scaler.pkl",
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )


# ==========================================================
# SAVE LABEL ENCODER
# ==========================================================

with open(
    "models/label_encoder.pkl",
    "wb"
) as file:

    pickle.dump(
        label_encoder,
        file
    )


# ==========================================================
# COMPLETION MESSAGE
# ==========================================================

print()

print(
    "LSTM model saved successfully."
)

print(
    "Model : models/crop_lstm.keras"
)

print(
    "Scaler: models/scaler.pkl"
)

print(
    "Encoder: models/label_encoder.pkl"
)