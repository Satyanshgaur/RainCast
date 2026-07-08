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

def run_scenario(scenario_name: str, environment: str = "rural"):
    print(f"\n========================================\nRunning Scenario: {scenario_name.upper()} (Env: {environment})\n========================================")
    
    config = ObservationConfig(scenario=scenario_name, environment=environment)
    
    # Retrieve and copy Ground Stations
    gs_delhi = [s for s in GROUND_STATIONS if s["name"] == "Delhi"][0].copy()
    gs_saopaulo = [s for s in GROUND_STATIONS if s["name"] == "Sao Paulo"][0].copy()
    gs_tokyo = [s for s in GROUND_STATIONS if s["name"] == "Tokyo"][0].copy()
    gs_berlin = [s for s in GROUND_STATIONS if s["name"] == "Berlin"][0].copy()
    
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
    
    # Models
    clf = XGBClassifier(objective="binary:logistic", n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1)
    reg = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1)
    
    clf.fit(X_train_clean, y_train_class_clean)
    reg.fit(X_train_clean, y_train_reg_clean)
    
    pred_class = clf.predict(X_test_clean)
    pred_reg = reg.predict(X_test_clean)
    
    f1 = f1_score(y_test_class_clean, pred_class, zero_division=0)
    rmse = np.sqrt(mean_squared_error(y_test_reg_clean, pred_reg))
    mae = mean_absolute_error(y_test_reg_clean, pred_reg)
    r2 = r2_score(y_test_reg_clean, pred_reg)
    
    return {
        "scenario": scenario_name,
        "environment": environment,
        "f1": f1,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def main():
    scenarios = [
        ("ideal", "rural"),
        ("typical", "rural"),
        ("severe", "urban")
    ]
    
    results = []
    for scenario, env in scenarios:
        res = run_scenario(scenario, env)
        results.append(res)
        
    print("\n\n==================================================")
    print("SCENARIO BENCHMARK SUMMARY TABLE")
    print("==================================================")
    print(f"{'Scenario':<12} | {'Environment':<12} | {'F1-Score':<8} | {'RMSE':<8} | {'MAE':<8} | {'R² Score':<8}")
    print("-" * 75)
    for r in results:
        print(f"{r['scenario'].upper():<12} | {r['environment'].upper():<12} | {r['f1']:<8.4f} | {r['rmse']:<8.4f} | {r['mae']:<8.4f} | {r['r2']:<8.4f}")
    print("==================================================")
    
    # Save to Markdown
    docs_dir = os.path.join(root_dir, "docs")
    study_file = os.path.join(docs_dir, "scenario_study.md")
    with open(study_file, "w") as f:
        f.write("# Scenario Benchmark Analysis (Ideal vs. Typical vs. Severe)\n\n")
        f.write("This study maps model performance across different environmental operating conditions.\n\n")
        f.write("| Scenario | Environment | F1-Score | Regressor RMSE (mm/h) | Regressor MAE (mm/h) | Regressor $R^2$ Score |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            f.write(f"| {r['scenario'].upper()} | {r['environment'].upper()} | {r['f1']:.4f} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['r2']:.4f} |\n")
            
    print(f"Results recorded in: {study_file}")

if __name__ == "__main__":
    main()
