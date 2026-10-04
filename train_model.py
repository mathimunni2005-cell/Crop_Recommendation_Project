import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load crop dataset
data = pd.read_csv("datasets/crop_recommendation.csv")

print("Dataset loaded successfully!")
print("Number of rows:", len(data))

# Input features
X = data[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]

# Target
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create ML model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test model
prediction = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, prediction)

print("--------------------------------")
print("CROP RECOMMENDATION MODEL")
print("--------------------------------")
print("Model trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("--------------------------------")

# Load investment dataset
investment_data = pd.read_csv(
    "datasets/crop_investment.csv"
)

print("Investment dataset loaded successfully!")
print(
    "Number of investment records:",
    len(investment_data)
)

print("--------------------------------")
print("Both datasets loaded successfully!")
print("--------------------------------")