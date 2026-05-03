import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# Load & clean dataset
# -----------------------------
df = pd.read_csv("link_training_data.csv")

# Remove duplicated header rows or junk rows
df = df[df["timestamp"] != "timestamp"]

# Parse timestamps safely
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"   # invalid strings → NaT
)

# Drop rows with invalid timestamps
df = df.dropna(subset=["timestamp"])

# Sort by time (important!)
df = df.sort_values("timestamp").reset_index(drop=True)

# -----------------------------
# Features & target
# -----------------------------
FEATURES = ["snr_db", "packet_loss", "load_factor"]
TARGET = "link_quality"

X = df[FEATURES]
y = df[TARGET]

# -----------------------------
# Time-aware split
# -----------------------------
split_idx = int(0.8 * len(df))

X_train = X.iloc[:split_idx]
X_val   = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_val   = y.iloc[split_idx:]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# -----------------------------
# XGBoost model (Ubuntu-safe)
# -----------------------------
model = XGBRegressor(
    objective="reg:squarederror",

    n_estimators=200,
    learning_rate=0.05,

    max_depth=3,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.7,

    reg_alpha=1.0,
    reg_lambda=2.0,

    random_state=42
)

# -----------------------------
# Train
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("\n--- Model Performance ---")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# -----------------------------
# Feature importance
# -----------------------------
print("\n--- Feature Importance ---")
for name, imp in zip(FEATURES, model.feature_importances_):
    print(f"{name:15s}: {imp:.4f}")

# -----------------------------
# Save model & scaler
# -----------------------------
joblib.dump(model, "xgb_link_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")

print("\n Model and scaler saved successfully.")

