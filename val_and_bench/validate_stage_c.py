import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import r2_score
from scipy.spatial.distance import jensenshannon

# Ensure we can import satlinksim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from satlinksim.satellite_link_sim import simulate_station
from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.domain.link.itu_models import itu_rain_coefficients, itu_rain_height, effective_path_length, gaseous_absorption_db
from satlinksim.domain.link.budget import fspl_db, noise_power_dbw
from satlinksim.domain.rain.engine import CorrelatedRainProcess
from evaluate_stage_b5 import extract_features_and_targets_b5

def main():
    print("======================================================================")
    print("STAGE C ROBUSTNESS AND GENERALIZATION VALIDATION SUITE")
    print("======================================================================")
    
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
    
    frequencies_hz = [10e9, 12e9, 14e9, 20e9, 30e9]
    bandwidth_hz = 36e6
    polarization = "vertical"
    n_steps = 7200
    
    # 1. Dataset Generation
    print("\nGenerating training data (Seed 100)...")
    X_train_list, y_train_list = [], []
    train_meta_list = []
    for f_hz in frequencies_hz:
        for i, gs in enumerate(stations):
            for force_rain in [True, False]:
                start_time = datetime(2026, months[i], 15, 12, 0, 0, tzinfo=timezone.utc)
                res = simulate_station(
                    gs, n_steps=n_steps, seed=100, freq_hz=f_hz,
                    bandwidth_hz=bandwidth_hz, polarization=polarization,
                    force_rain=force_rain, start_time=start_time
                )
                X_df, y_df, meta = extract_features_and_targets_b5(res, gs, f_hz, bandwidth_hz, polarization, start_time)
                X_train_list.append(X_df)
                y_train_list.append(y_df)
                train_meta_list.append(pd.DataFrame(meta))
                
    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)["true_rain_rate"].values
    train_meta = pd.concat(train_meta_list, ignore_index=True)
    
    print("Generating testing data (Seed 200)...")
    X_test_list, y_test_list = [], []
    test_meta_list = []
    for f_hz in frequencies_hz:
        for i, gs in enumerate(stations):
            for force_rain in [True, False]:
                start_time = datetime(2026, months[i], 15, 12, 0, 0, tzinfo=timezone.utc)
                res = simulate_station(
                    gs, n_steps=n_steps, seed=200, freq_hz=f_hz,
                    bandwidth_hz=bandwidth_hz, polarization=polarization,
                    force_rain=force_rain, start_time=start_time
                )
                X_df, y_df, meta = extract_features_and_targets_b5(res, gs, f_hz, bandwidth_hz, polarization, start_time)
                X_test_list.append(X_df)
                y_test_list.append(y_df)
                test_meta_list.append(pd.DataFrame(meta))
                
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)["true_rain_rate"].values
    test_meta = pd.concat(test_meta_list, ignore_index=True)
    
    # Helper to evaluate cascade
    def evaluate_cascade(X_tr, y_tr, X_te, y_te):
        y_tr_c = (y_tr > 0.1).astype(int)
        
        clf = XGBClassifier(objective="binary:logistic", n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr_c)
        
        reg = XGBRegressor(objective="reg:squarederror", n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
        reg.fit(X_tr, y_tr)
        
        pred_c = clf.predict(X_te)
        pred_raw_r = reg.predict(X_te)
        pred_r = np.where(pred_c == 1, np.maximum(pred_raw_r, 0.0), 0.0)
        
        rmse = np.sqrt(np.mean((y_te - pred_r)**2))
        mae = np.mean(np.abs(y_te - pred_r))
        corr = np.corrcoef(y_te, pred_r)[0, 1] if np.std(pred_r) > 0 else 0.0
        r2 = r2_score(y_te, pred_r)
        
        # Classification F1
        tp = np.sum((y_te > 0.1) & (pred_r > 0.1))
        fp = np.sum((y_te <= 0.1) & (pred_r > 0.1))
        fn = np.sum((y_te > 0.1) & (pred_r <= 0.1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        return rmse, mae, corr, r2, f1, pred_r

    # Train base model
    print("\nEvaluating base Stage C model...")
    rmse, mae, corr, r2, f1, base_preds = evaluate_cascade(X_train, y_train, X_test, y_test)
    print(f"Base: RMSE: {rmse:.4f} | R²: {r2:.4f} | Corr: {corr:.4f} | F1: {f1:.4f}")
    
    # 1. Leave-One-Station-Out (LOSO) Generalization
    print("\nRunning LOSO for Stage C...")
    loso_results = {}
    for i, test_station in enumerate(stations):
        name = test_station["name"]
        
        # Train on other 3
        train_idx = (train_meta["station"] != name)
        test_idx = (test_meta["station"] != name)
        
        X_tr_loso = pd.concat([X_train[train_idx], X_test[test_idx]], ignore_index=True)
        y_tr_loso = np.concatenate([y_train[train_idx.values], y_test[test_idx.values]])
        
        # Test on the excluded station (all frequencies, test split only)
        val_idx = (test_meta["station"] == name) & (test_meta["force_rain"] == False)
        X_te_loso = X_test[val_idx]
        y_te_loso = y_test[val_idx.values]
        
        l_rmse, _, _, l_r2, l_f1, _ = evaluate_cascade(X_tr_loso, y_tr_loso, X_te_loso, y_te_loso)
        print(f"  Excluded: {name:12s} | RMSE: {l_rmse:.4f} | R²: {l_r2:.4f} | F1: {l_f1:.4f}")
        loso_results[name] = {"rmse": l_rmse, "r2": l_r2, "f1": l_f1}
        
    # 2. Simulator Parameter Modification (Noise shift)
    print("\nRunning Parameter Noise Shift for Stage C...")
    X_mod_list, y_mod_list = [], []
    for f_hz in frequencies_hz:
        for i, gs in enumerate(stations):
            for force_rain in [True, False]:
                start_time = datetime(2026, months[i], 15, 12, 0, 0, tzinfo=timezone.utc)
                res = simulate_station(
                    gs, n_steps=n_steps, seed=400, freq_hz=f_hz,
                    bandwidth_hz=bandwidth_hz, polarization=polarization,
                    force_rain=force_rain, start_time=start_time,
                    rain_model_factory=lambda g: CorrelatedRainProcess([g], dt_s=1.0, tau_c=600.0, force_rain=force_rain, rain_rate_scale=1.5)
                )
                X_df, y_df, _ = extract_features_and_targets_b5(res, gs, f_hz, bandwidth_hz, polarization, start_time)
                
                # Double the standard scintillation
                from satlinksim.domain.link.itu_models import scintillation_sigma_db
                ss = scintillation_sigma_db(f_hz/1e9, np.array(res.elevation_series), gs["antenna_diam_m"], gs["humidity_pct"])
                additional_noise = np.random.normal(0, ss)
                X_df["excess_attn"] += additional_noise
                
                X_mod_list.append(X_df)
                y_mod_list.append(y_df)
                
    X_mod = pd.concat(X_mod_list, ignore_index=True)
    y_mod = pd.concat(y_mod_list, ignore_index=True)["true_rain_rate"].values
    
    m_rmse, _, m_corr, m_r2, m_f1, _ = evaluate_cascade(X_train, y_train, X_mod, y_mod)
    print(f"  Noise Shift | RMSE: {m_rmse:.4f} | R²: {m_r2:.4f} | Corr: {m_corr:.4f} | F1: {m_f1:.4f}")
    
    # 3. Distribution Matching
    print("\nRunning JS Divergence for Stage C...")
    idx_sp = (test_meta["station"] == "Sao Paulo") & (test_meta["force_rain"] == False)
    y_true_sp = y_test[idx_sp]
    y_pred_sp = base_preds[idx_sp]
    
    bins = np.linspace(0.0, 10.0, 100)
    p_true, _ = np.histogram(y_true_sp, bins=bins, density=True)
    p_pred, _ = np.histogram(y_pred_sp, bins=bins, density=True)
    
    p_true = p_true + 1e-12
    p_pred = p_pred + 1e-12
    p_true /= np.sum(p_true)
    p_pred /= np.sum(p_pred)
    
    js_div = jensenshannon(p_true, p_pred)
    print(f"  Sao Paulo JS Divergence: {js_div:.4f}")

if __name__ == "__main__":
    main()
