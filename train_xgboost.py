#!/usr/bin/env python3

import os
import sys
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ==================================================
# CONFIG
# ==================================================
CSV_FILE = "link_training_data.csv"

# ==================================================
# CHECK FILE
# ==================================================
if not os.path.exists(CSV_FILE):
    print(f"ERROR: {CSV_FILE} not found")
    sys.exit(1)

# ==================================================
# LOAD DATASET
# ==================================================
print(f"\nLoading dataset: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

print(f"Original dataset shape: {df.shape}")

# ==================================================
# REMOVE TIMESTAMP
# ==================================================
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])

# ==================================================
# FORCE NUMERIC VALUES
# ==================================================
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove broken rows
df = df.dropna()

print(f"Cleaned dataset shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

# ==================================================
# FEATURES + TARGET
# ==================================================
TARGET = "link_quality"
FEATURES = [c for c in df.columns if c != TARGET]

print(f"\nTarget   : {TARGET}")
print(f"Features : {FEATURES}")

# ==================================================
# DATA
# ==================================================
X = df[FEATURES]
y = df[TARGET]

# ==================================================
# SCALE
# ==================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================================================
# SPLIT
# ==================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining samples  : {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# ==================================================
# MODEL
# ==================================================
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

# ==================================================
# TRAIN
# ==================================================
print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete.")

# ==================================================
# PREDICT
# ==================================================
y_pred = model.predict(X_val)

# ==================================================
# METRICS
# ==================================================
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ==================================================
# FEATURE IMPORTANCE
# ==================================================
print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

for feature, importance in zip(FEATURES, model.feature_importances_):
    print(f"{feature:20s}: {importance:.4f}")

# ==================================================
# SAVE
# ==================================================
joblib.dump(model, "xgb_link_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")

print("\nModel + scaler saved successfully.")
