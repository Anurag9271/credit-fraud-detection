import pandas as pd
import joblib
import sys
import os

# Allow importing modules from src folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import build_preprocessor
from model import xgboost_model, evaluate_model

from sklearn.model_selection import train_test_split

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/credit_card_fraud.csv")

print("Columns:", df.columns.tolist())
print("Dataset Shape:", df.shape)

# -----------------------------
# Define Features
# -----------------------------
selected_features = [
    "transaction_amount",
    "num_transactions_24h",
    "num_transactions_7d",
    "previous_fraud_count",
    "avg_transaction_amount"
]

X = df[selected_features]
y = df["fraud"]

numeric_features = X.select_dtypes(include=['int64','float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# -----------------------------
# Preprocessing Pipeline
# -----------------------------
preprocessor = build_preprocessor(
    numeric_features,
    categorical_features
)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# Build Models
# -----------------------------
model = xgboost_model(preprocessor)

# -----------------------------
# Evaluate Models
# -----------------------------
best_pipeline = evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
)

# -----------------------------
# Save Best Model
# -----------------------------
os.makedirs("../models", exist_ok=True)

joblib.dump(best_pipeline, "models/fraud_model_reduced.joblib")

print("✅ Model retrained and saved successfully!")