import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Read CSV without header
df = pd.read_csv("dataset/gestures.csv", header=None)

# First column = labels
y = df.iloc[:, 0]

# Remaining columns = features
X = df.iloc[:, 1:]

print("Dataset Shape:", df.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "models/gesture_model.pkl")

print("Model saved successfully!")