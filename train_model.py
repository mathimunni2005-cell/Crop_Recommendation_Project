import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================================
# SETTINGS
# ==========================================================

DATASET_PATH = "datasets/crop_recommendation.csv"
MODEL_DIR = "models"

SEQUENCE_LENGTH = 3


# ==========================================================
# CREATE MODEL FOLDER
# ==========================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================================
# LOAD DATASET
# ==========================================================

data = pd.read_csv(DATASET_PATH)

print("\n==========================================")
print("LSTM CROP RECOMMENDATION SYSTEM")
print("==========================================")

print("\nDataset shape:", data.shape)

print("\nCrop distribution:")
print(data["label"].value_counts())


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
# ENCODE CROP LABELS
# ==========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nCrop classes:")
print(label_encoder.classes_)


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

    crop_data = X_scaled[crop_indices]

    crop_labels = y_encoded[crop_indices]

    if len(crop_data) < SEQUENCE_LENGTH:
        continue

    for i in range(
        len(crop_data) - SEQUENCE_LENGTH + 1
    ):

        sequence = crop_data[
            i:i + SEQUENCE_LENGTH
        ]

        label = crop_labels[
            i + SEQUENCE_LENGTH - 1
        ]

        X_sequences.append(sequence)
        y_sequences.append(label)


X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)


print("\nSequence shape:")
print(X_sequences.shape)


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.25,
    random_state=42,
    stratify=y_sequences
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================================
# BUILD LSTM MODEL
# ==========================================================

model = Sequential()

model.add(
    Bidirectional(
        LSTM(
            64,
            return_sequences=True
        ),
        input_shape=(
            SEQUENCE_LENGTH,
            len(features)
        )
    )
)

model.add(Dropout(0.25))


model.add(
    LSTM(32)
)

model.add(Dropout(0.20))


model.add(
    Dense(
        32,
        activation="relu"
    )
)


model.add(
    Dropout(0.15)
)


model.add(
    Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
)


# ==========================================================
# COMPILE
# ==========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n==========================================")
print("MODEL SUMMARY")
print("==========================================")

model.summary()


# ==========================================================
# EARLY STOPPING
# ==========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)


# ==========================================================
# TRAIN
# ==========================================================

print("\n==========================================")
print("TRAINING LSTM")
print("==========================================")


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
# EVALUATION
# ==========================================================

predictions = model.predict(
    X_test,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)


accuracy = accuracy_score(
    y_test,
    predicted_classes
)


print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predicted_classes,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ==========================================================
# SAVE MODEL
# ==========================================================

model_path = os.path.join(
    MODEL_DIR,
    "crop_lstm.keras"
)

model.save(model_path)


# ==========================================================
# SAVE SCALER
# ==========================================================

scaler_path = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

joblib.dump(
    scaler,
    scaler_path
)


# ==========================================================
# SAVE LABEL ENCODER
# ==========================================================

encoder_path = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)

joblib.dump(
    label_encoder,
    encoder_path
)


# ==========================================================
# FINISHED
# ==========================================================

print("\n==========================================")
print("TRAINING COMPLETED")
print("==========================================")

print(
    "\nModel saved:"
)

print(model_path)

print(
    "\nScaler saved:"
)

print(scaler_path)

print(
    "\nLabel encoder saved:"
)

print(encoder_path)

print("\n==========================================")