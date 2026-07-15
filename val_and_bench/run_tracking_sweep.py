import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import r2_score, f1_score

# Add local src path and root path to import modules
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.satellite_link_sim import simulate_station
from satlinksim.domain.observation import ObservationConfig
from val_and_bench.evaluate_stage_b import extract_features_and_targets

def run_tracking_experiment(tracking_sigma: float):
    # Setup config with scenario typical but override tracking sigma
    # We keep only tracking and scintillation active to measure tracking loop noise effect in isolation
    config = ObservationConfig(
        scenario="typical",
        enable_scintillation=True,
        enable_tracking=True if tracking_sigma > 0.0 else False,
        enable_calibration=False,
        enable_agc=False,
        enable_multipath=False,
        enable_wet_antenna=False,
        tracking_sigma_override=tracking_sigma
    )
    
    # Retrieve and copy Ground Stations
    gs_delhi = [s for s in GROUND_STATIONS if s["name"] == "Delhi"][0].copy()
    gs_saopaulo = [s for s in GROUND_STATIONS if s["name"] == "Sao Paulo"][0].copy()
    gs_tokyo = [s for s in GROUND_STATIONS if s["name"] == "Tokyo"][0].copy()
    gs_berlin = [s for s in GROUND_STATIONS if s["name"] == "Berlin"][0].copy()
    
    # Assign NORAD IDs for propagation
    gs_delhi["norad_id"] = 26766
    gs_saopaulo["norad_id"] = 33153
    gs_tokyo["norad_id"] = 26900
    gs_berlin["norad_id"] = 27380
    
    stations = [gs_delhi, gs_saopaulo, gs_tokyo, gs_berlin]
    months = [7, 1, 9, 11]
    
    n_steps = 7200
    freq_hz = 14e9
    bandwidth_hz = 36e6
    polarization = "vertical"
    
    # 1. Generate Training Data
    X_train_list, y_train_list = [], []
    for i, gs in enumerate(stations):
        for force_rain in [True, False]:
            start_time = datetime(2026, months[i], 15, 12, 0, 0, tzinfo=timezone.utc)
            res = simulate_station(
                gs, n_steps=n_steps, seed=100, freq_hz=freq_hz,
                bandwidth_hz=bandwidth_hz, polarization=polarization,
                force_rain=force_rain, start_time=start_time
            )
            X_df, y_df, _ = extract_features_and_targets(
                res, gs, freq_hz, bandwidth_hz, polarization, start_time, obs_config=config
            )
            X_train_list.append(X_df)
            y_train_list.append(y_df)
            
    X_train_full = pd.concat(X_train_list, ignore_index=True)
    y_train_full = pd.concat(y_train_list, ignore_index=True)
    
    train_idx = []
    for r in range(8):
        offset = r * n_steps
        train_idx.extend(range(offset, offset + int(n_steps * 0.70)))
        
    X_train = X_train_full.iloc[train_idx]
    y_train_class = (y_train_full.iloc[train_idx]["true_rain_rate"].values > 0.1).astype(int)
    y_train_reg = y_train_full.iloc[train_idx]["true_rain_rate"].values
    
    # 2. Generate Test Data (Seed 200)
    X_test_list, y_test_list = [], []
    for i, gs in enumerate(stations):
        for force_rain in [True, False]:
            start_time = datetime(2026, months[i], 15, 12, 0, 0, tzinfo=timezone.utc)
            res = simulate_station(
                gs, n_steps=n_steps, seed=200, freq_hz=freq_hz,
                bandwidth_hz=bandwidth_hz, polarization=polarization,
                force_rain=force_rain, start_time=start_time
            )
            X_df, y_df, _ = extract_features_and_targets(
                res, gs, freq_hz, bandwidth_hz, polarization, start_time, obs_config=config
            )
            test_start = int(n_steps * 0.85)
            X_test_list.append(X_df.iloc[test_start:])
            y_test_list.append(y_df.iloc[test_start:])
            
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_test_full = pd.concat(y_test_list, ignore_index=True)
    y_test_class = (y_test_full["true_rain_rate"].values > 0.1).astype(int)
    y_test_reg = y_test_full["true_rain_rate"].values
    
    train_valid = ~X_train.isna().any(axis=1)
    X_train_clean = X_train[train_valid]
    y_train_class_clean = y_train_class[train_valid]
    y_train_reg_clean = y_train_reg[train_valid]
    
    test_valid = ~X_test.isna().any(axis=1)
    X_test_clean = X_test[test_valid]
    y_test_class_clean = y_test_class[test_valid]
    y_test_reg_clean = y_test_reg[test_valid]
    
    # Define models
    clf = XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    
    reg = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    clf.fit(X_train_clean, y_train_class_clean)
    reg.fit(X_train_clean, y_train_reg_clean)
    
    # Predict
    pred_class = clf.predict(X_test_clean)
    pred_reg = reg.predict(X_test_clean)
    
    # Metrics
    f1 = f1_score(y_test_class_clean, pred_class, zero_division=0)
    r2 = r2_score(y_test_reg_clean, pred_reg)
    
    print(f"Tracking Sigma: {tracking_sigma:.2f}° | F1-Score: {f1:.4f} | R²: {r2:.4f}")
    
    return {
        "sigma": tracking_sigma,
        "f1": f1,
        "r2": r2
    }

def main():
    sigmas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
    results = []
    
    print("Running tracking noise sweep...")
    for sigma in sigmas:
        res = run_tracking_experiment(sigma)
        results.append(res)
        
    print("\n\n==========================================")
    print("TRACKING LOOP NOISE SWEEP SUMMARY TABLE")
    print("==========================================")
    print(f"{'Tracking σ':<12} | {'F1-Score':<8} | {'R² Score':<8}")
    print("-" * 34)
    for r in results:
        print(f"{str(r['sigma']) + '°':<12} | {r['f1']:<8.4f} | {r['r2']:<8.4f}")
    print("==========================================")
    
    # Save to a Markdown table in docs/tracking_sweep_study.md
    docs_dir = os.path.join(root_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    study_file = os.path.join(docs_dir, "tracking_sweep_study.md")
    
    with open(study_file, "w") as f:
        f.write("# Tracking Noise Sweep Experiment Study\n\n")
        f.write("This document logs the impact of increasing nominal antenna tracking loop error standard deviation (σ_tracking) on rainfall narrowcasting classification F1 and regression $R^2$ scores.\n\n")
        f.write("| Tracking σ | F1-Score | Regressor $R^2$ Score |\n")
        f.write("| :--- | :---: | :---: |\n")
        for r in results:
            f.write(f"| {r['sigma']:.2f}° | {r['f1']:.4f} | {r['r2']:.4f} |\n")
            
    print(f"Results recorded in: {study_file}")

if __name__ == "__main__":
    main()
