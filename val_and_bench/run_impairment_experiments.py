import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score

# Add local src path and root path to import modules
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.satellite_link_sim import simulate_station
from satlinksim.domain.observation import ObservationConfig, ObservationModel
from val_and_bench.evaluate_stage_b import extract_features_and_targets

def run_experiment(config: ObservationConfig, name: str):
    print(f"\n========================================\nRunning Experiment: {name}\n========================================")
    
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
            # Pass our custom config to the feature extraction function
            X_df, y_df, _ = extract_features_and_targets(
                res, gs, freq_hz, bandwidth_hz, polarization, start_time, obs_config=config
            )
            X_train_list.append(X_df)
            y_train_list.append(y_df)
            
    X_train_full = pd.concat(X_train_list, ignore_index=True)
    y_train_full = pd.concat(y_train_list, ignore_index=True)
    
    # Split training into train/val chronologically to replicate dataset generator splits
    train_idx = []
    val_idx = []
    # 8 runs total, each with n_steps
    for r in range(8):
        offset = r * n_steps
        train_idx.extend(range(offset, offset + int(n_steps * 0.70)))
        val_idx.extend(range(offset + int(n_steps * 0.70), offset + int(n_steps * 0.85)))
        
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
            # Test split is the last 15% of each run
            test_start = int(n_steps * 0.85)
            X_test_list.append(X_df.iloc[test_start:])
            y_test_list.append(y_df.iloc[test_start:])
            
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_test_full = pd.concat(y_test_list, ignore_index=True)
    y_test_class = (y_test_full["true_rain_rate"].values > 0.1).astype(int)
    y_test_reg = y_test_full["true_rain_rate"].values
    
    # Drop rows containing NaNs in features (due to shift/lag features)
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
    rmse = np.sqrt(mean_squared_error(y_test_reg_clean, pred_reg))
    mae = mean_absolute_error(y_test_reg_clean, pred_reg)
    r2 = r2_score(y_test_reg_clean, pred_reg)
    
    print(f"Metrics for {name}:")
    print(f"  F1-Score (Rain Event): {f1:.4f}")
    print(f"  Regressor RMSE:       {rmse:.4f} mm/h")
    print(f"  Regressor MAE:        {mae:.4f} mm/h")
    print(f"  Regressor R² Score:   {r2:.4f}")
    
    return {
        "name": name,
        "f1": f1,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def main():
    incremental_experiments = [
        (
            ObservationConfig(
                scenario="typical",
                enable_scintillation=True,
                enable_tracking=False,
                enable_calibration=False,
                enable_agc=False,
                enable_multipath=False,
                enable_wet_antenna=False
            ),
            "1. Only Scintillation"
        ),
        (
            ObservationConfig(
                scenario="typical",
                enable_scintillation=True,
                enable_tracking=True,
                enable_calibration=False,
                enable_agc=False,
                enable_multipath=False,
                enable_wet_antenna=False
            ),
            "2. + Tracking"
        ),
        (
            ObservationConfig(
                scenario="typical",
                enable_scintillation=True,
                enable_tracking=True,
                enable_calibration=True,
                enable_agc=False,
                enable_multipath=False,
                enable_wet_antenna=False
            ),
            "3. + Calibration"
        ),
        (
            ObservationConfig(
                scenario="typical",
                enable_scintillation=True,
                enable_tracking=True,
                enable_calibration=True,
                enable_agc=True,
                enable_multipath=False,
                enable_wet_antenna=False
            ),
            "4. + AGC & ADC"
        ),
        (
            ObservationConfig(
                scenario="typical",
                enable_scintillation=True,
                enable_tracking=True,
                enable_calibration=True,
                enable_agc=True,
                enable_multipath=True,
                enable_wet_antenna=False
            ),
            "5. + Multipath"
        ),
        (
            ObservationConfig(
                scenario="typical",
                enable_scintillation=True,
                enable_tracking=True,
                enable_calibration=True,
                enable_agc=True,
                enable_multipath=True,
                enable_wet_antenna=True
            ),
            "6. + Wet Antenna"
        )
    ]
    
    print("\n==========================================")
    print("RUNNING INCREMENTAL IMPAIRMENT STUDY")
    print("==========================================")
    inc_results = []
    for config, name in incremental_experiments:
        res = run_experiment(config, name)
        inc_results.append(res)
        
    solo_experiments = [
        (
            ObservationConfig(
                scenario="typical",
                enable_scintillation=True,
                enable_tracking=False,
                enable_calibration=False,
                enable_agc=False,
                enable_multipath=False,
                enable_wet_antenna=False
            ),
            "1. Scintillation Solo"
        ),
        (
            ObservationConfig(
                scenario="typical",
                enable_scintillation=False,
                enable_tracking=True,
                enable_calibration=False,
                enable_agc=False,
                enable_multipath=False,
                enable_wet_antenna=False
            ),
            "2. Tracking Solo"
        ),
        (
            ObservationConfig(
                scenario="typical",
                enable_scintillation=False,
                enable_tracking=False,
                enable_calibration=True,
                enable_agc=False,
                enable_multipath=False,
                enable_wet_antenna=False
            ),
            "3. Calibration Solo"
        ),
        (
            ObservationConfig(
                scenario="typical",
                enable_scintillation=False,
                enable_tracking=False,
                enable_calibration=False,
                enable_agc=True,
                enable_multipath=False,
                enable_wet_antenna=False
            ),
            "4. AGC & ADC Solo"
        ),
        (
            ObservationConfig(
                scenario="typical",
                enable_scintillation=False,
                enable_tracking=False,
                enable_calibration=False,
                enable_agc=False,
                enable_multipath=True,
                enable_wet_antenna=False
            ),
            "5. Multipath Solo"
        ),
        (
            ObservationConfig(
                scenario="typical",
                enable_scintillation=False,
                enable_tracking=False,
                enable_calibration=False,
                enable_agc=False,
                enable_multipath=False,
                enable_wet_antenna=True
            ),
            "6. Wet Antenna Solo"
        )
    ]
    
    print("\n==========================================")
    print("RUNNING SOLO IMPAIRMENT STUDY")
    print("==========================================")
    solo_results = []
    for config, name in solo_experiments:
        res = run_experiment(config, name)
        solo_results.append(res)
        
    print("\n\n==================================================")
    print("FINAL INCREMENTAL DEGRADATION SUMMARY TABLE")
    print("==================================================")
    print(f"{'Experiment':<25} | {'F1-Score':<8} | {'RMSE':<8} | {'MAE':<8} | {'R² Score':<8}")
    print("-" * 68)
    for r in inc_results:
        print(f"{r['name']:<25} | {r['f1']:<8.4f} | {r['rmse']:<8.4f} | {r['mae']:<8.4f} | {r['r2']:<8.4f}")
    print("==================================================")
    
    print("\n\n==================================================")
    print("FINAL SOLO IMPAIRMENT SUMMARY TABLE")
    print("==================================================")
    print(f"{'Experiment':<25} | {'F1-Score':<8} | {'RMSE':<8} | {'MAE':<8} | {'R² Score':<8}")
    print("-" * 68)
    for r in solo_results:
        print(f"{r['name']:<25} | {r['f1']:<8.4f} | {r['rmse']:<8.4f} | {r['mae']:<8.4f} | {r['r2']:<8.4f}")
    print("==================================================")
    
    # Save to a Markdown table in docs/impairment_study.md
    docs_dir = os.path.join(root_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    study_file = os.path.join(docs_dir, "impairment_study.md")
    
    with open(study_file, "w") as f:
        f.write("# Isolated Impairment Impact Experiment Study\n\n")
        f.write("This document logs the impact of physical receiver and link impairments on rainfall narrowcasting, evaluating both incremental addition and individual (solo) effects.\n\n")
        
        f.write("## 1. Incremental Degradation\n")
        f.write("| Experiment | F1-Score | Regressor RMSE (mm/h) | Regressor MAE (mm/h) | Regressor $R^2$ Score |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for r in inc_results:
            f.write(f"| {r['name']} | {r['f1']:.4f} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['r2']:.4f} |\n")
            
        f.write("\n## 2. Solo Impairment Effects\n")
        f.write("| Experiment | F1-Score | Regressor RMSE (mm/h) | Regressor MAE (mm/h) | Regressor $R^2$ Score |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for r in solo_results:
            f.write(f"| {r['name']} | {r['f1']:.4f} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['r2']:.4f} |\n")
            
    print(f"Results recorded in: {study_file}")

if __name__ == "__main__":
    main()
