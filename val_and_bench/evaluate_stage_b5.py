import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import r2_score

# Ensure we can import satlinksim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from satlinksim.satellite_link_sim import simulate_station
from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.domain.link.itu_models import itu_rain_coefficients, itu_rain_height, effective_path_length, gaseous_absorption_db
from satlinksim.domain.link.budget import fspl_db, noise_power_dbw

def extract_features_and_targets_b5(res, gs, freq_hz, bandwidth_hz, polarization, start_time):
    n_steps = len(res.snr_series)
    snr_series = np.array(res.snr_series)
    el_series = np.array(res.elevation_series)
    slant_series = np.array(res.slant_range_series)
    
    # Calculate physical constants
    eirp = gs["eirp_dbw"]
    g_rx = gs["g_rx_dbi"]
    noise_floor = noise_power_dbw(gs["system_temp_k"], bandwidth_hz)
    rain_h = itu_rain_height(gs["latitude"])
    itu_k, itu_alpha = itu_rain_coefficients(freq_hz / 1e9, polarization)
    
    # Subtract FSPL and gas loss
    pl = fspl_db(freq_hz, slant_series)
    gas_loss = gaseous_absorption_db(freq_hz / 1e9, el_series, gs["wv_g_m3"])
    total_gain = eirp + g_rx - noise_floor
    excess_attn = total_gain - snr_series - pl - gas_loss
    
    # Effective path length
    ep = effective_path_length(el_series, rain_h, gs["altitude_km"], itu_k)
    
    # Build dataframe for features
    df = pd.DataFrame({
        "excess_attn": excess_attn,
        "elevation": el_series,
        "L_eff": ep,
    })
    
    features = {}
    features["excess_attn"] = excess_attn
    features["elevation"] = el_series
    features["L_eff"] = ep
    features["freq_ghz"] = np.full(n_steps, freq_hz / 1e9)
    
    # EXPLICIT PHYSICAL PARAMETERS FOR FREQUENCY AWARENESS
    features["itu_k"] = np.full(n_steps, itu_k)
    features["itu_alpha"] = np.full(n_steps, itu_alpha)
    
    # Season (month of the year proxy)
    month = start_time.month
    features["season_sin"] = np.full(n_steps, np.sin(2 * np.pi * month / 12))
    features["season_cos"] = np.full(n_steps, np.cos(2 * np.pi * month / 12))
    
    # Ground station climatology features
    features["gs_latitude"] = np.full(n_steps, gs["latitude"])
    features["gs_humidity"] = np.full(n_steps, gs["humidity_pct"])
    features["gs_wv"] = np.full(n_steps, gs["wv_g_m3"])
    features["itu_R001"] = np.full(n_steps, gs["itu_rain"]["R001"])
    features["itu_P_rain"] = np.full(n_steps, gs["itu_rain"]["P_rain"])
    
    # Rolling window stats
    for window in [30, 60, 300]:
        features[f"rolling_mean_{window}s"] = df["excess_attn"].rolling(window, min_periods=1).mean().values
        features[f"rolling_std_{window}s"] = df["excess_attn"].rolling(window, min_periods=1).std().fillna(0.0).values
        features[f"rolling_max_{window}s"] = df["excess_attn"].rolling(window, min_periods=1).max().values
        features[f"rolling_min_{window}s"] = df["excess_attn"].rolling(window, min_periods=1).min().values
        
    # Lag features
    for lag in [1, 5, 10]:
        features[f"lag_excess_attn_{lag}s"] = df["excess_attn"].shift(lag).fillna(method='bfill').values
        
    feature_df = pd.DataFrame(features)
    targets = pd.DataFrame({
        "true_rain_rate": np.array(res.rain_series),
    })
    
    metadata = {
        "station": [gs["name"]] * n_steps,
        "force_rain": [res.rain_fraction == 1.0] * n_steps,
        "freq_ghz": [freq_hz / 1e9] * n_steps
    }
    
    return feature_df, targets, metadata

def run_stage_b5():
    print("======================================================================")
    print("STAGE B.5: FREQUENCY-AWARE XGBOOST NARROWCASTER")
    print("======================================================================")
    
    # Retrieve Ground Stations
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
    
    # Frequencies to train and evaluate on
    frequencies_hz = [10e9, 12e9, 14e9, 20e9, 30e9]
    bandwidth_hz = 36e6
    polarization = "vertical"
    n_steps = 7200
    
    # 1. Dataset Generation (Seed 100 for Train, Seed 200 for Test)
    print("\n[Data Prep] Generating multi-frequency training data (Seed 100)...")
    X_train_list, y_train_list = [], []
    for f_hz in frequencies_hz:
        for i, gs in enumerate(stations):
            for force_rain in [True, False]:
                start_time = datetime(2026, months[i], 15, 12, 0, 0, tzinfo=timezone.utc)
                res = simulate_station(
                    gs, n_steps=n_steps, seed=100, freq_hz=f_hz,
                    bandwidth_hz=bandwidth_hz, polarization=polarization,
                    force_rain=force_rain, start_time=start_time
                )
                X_df, y_df, _ = extract_features_and_targets_b5(res, gs, f_hz, bandwidth_hz, polarization, start_time)
                X_train_list.append(X_df)
                y_train_list.append(y_df)
                
    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)["true_rain_rate"].values
    
    print("[Data Prep] Generating multi-frequency testing data (Seed 200)...")
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
    
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape : {X_test.shape}")
    
    # 2. Train Models
    print("\nTraining Frequency-Aware XGBClassifier...")
    y_train_class = (y_train > 0.1).astype(int)
    clf = XGBClassifier(objective="binary:logistic", n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train_class)
    
    print("Training Frequency-Aware XGBRegressor...")
    reg = XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)
    
    # 3. Predict & Gating
    print("\nInference on test dataset...")
    pred_c = clf.predict(X_test)
    pred_raw_r = reg.predict(X_test)
    pred_rain_rate = np.where(pred_c == 1, np.maximum(pred_raw_r, 0.0), 0.0)
    
    # 4. Overall metrics
    overall_rmse = np.sqrt(np.mean((y_test - pred_rain_rate)**2))
    overall_mae = np.mean(np.abs(y_test - pred_rain_rate))
    overall_corr = np.corrcoef(y_test, pred_rain_rate)[0, 1]
    overall_r2 = r2_score(y_test, pred_rain_rate)
    
    print(f"\nOverall Frequency-Aware Performance:")
    print(f"  RMSE: {overall_rmse:.4f} mm/h")
    print(f"  MAE : {overall_mae:.4f} mm/h")
    print(f"  Corr: {overall_corr:.4f}")
    print(f"  R²  : {overall_r2:.4f}")
    
    # 5. Split metrics by frequency to check generalization
    freq_metrics = []
    
    # Hardcoded results from Stage B (frequency-unaware model, trained at 14 GHz) for comparison:
    # 12 GHz: RMSE=2.1962, Corr=0.9956, R2=0.8978
    # 20 GHz: RMSE=3.6022, Corr=0.9745, R2=0.7250
    # 30 GHz: RMSE=7.7491, Corr=0.9293, R2=-0.2727
    unaware_stats = {
        10.0: {"rmse": np.nan, "corr": np.nan, "r2": np.nan},
        12.0: {"rmse": 2.1962, "corr": 0.9956, "r2": 0.8978},
        14.0: {"rmse": 0.4934, "corr": 0.9973, "r2": 0.9945},
        20.0: {"rmse": 3.6022, "corr": 0.9745, "r2": 0.7250},
        30.0: {"rmse": 7.7491, "corr": 0.9293, "r2": -0.2727}
    }
    
    for f in [10.0, 12.0, 14.0, 20.0, 30.0]:
        idx_f = (test_meta["freq_ghz"] == f)
        y_true_f = y_test[idx_f]
        y_pred_f = pred_rain_rate[idx_f]
        
        f_rmse = np.sqrt(np.mean((y_true_f - y_pred_f)**2))
        f_mae = np.mean(np.abs(y_true_f - y_pred_f))
        f_corr = np.corrcoef(y_true_f, y_pred_f)[0, 1] if np.std(y_pred_f) > 0 else 0.0
        f_r2 = r2_score(y_true_f, y_pred_f)
        
        tp = np.sum((y_true_f > 0.1) & (y_pred_f > 0.1))
        fp = np.sum((y_true_f <= 0.1) & (y_pred_f > 0.1))
        fn = np.sum((y_true_f > 0.1) & (y_pred_f <= 0.1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        print(f"\nFrequency: {f} GHz")
        print(f"  RMSE: {f_rmse:.4f} | R²: {f_r2:.4f} | Corr: {f_corr:.4f} | F1: {f_f1:.4f}")
        
        freq_metrics.append({
            "freq": f,
            "rmse": f_rmse,
            "mae": f_mae,
            "corr": f_corr,
            "r2": f_r2,
            "f1": f_f1,
            "unaware_rmse": unaware_stats[f]["rmse"],
            "unaware_r2": unaware_stats[f]["r2"]
        })
        
    # Generate bar chart comparing R² scores before vs after frequency awareness
    freqs_label = ['12 GHz', '14 GHz', '20 GHz', '30 GHz']
    r2_unaware = [unaware_stats[f]["r2"] for f in [12.0, 14.0, 20.0, 30.0]]
    r2_aware = [next(item for item in freq_metrics if item["freq"] == f)["r2"] for f in [12.0, 14.0, 20.0, 30.0]]
    
    x = np.arange(len(freqs_label))
    width = 0.35
    
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, r2_unaware, width, label='Stage B: Frequency-Unaware (Trained at 14 GHz)', color='red', alpha=0.7)
    plt.bar(x + width/2, r2_aware, width, label='Stage B.5: Frequency-Aware Multi-channel', color='green', alpha=0.7)
    
    plt.ylabel('R² Score')
    plt.title('Cross-Frequency Generalization: R² Score Comparison')
    plt.xticks(x, freqs_label)
    plt.ylim(-0.5, 1.05)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    
    artifact_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "plots"))
    os.makedirs(artifact_dir, exist_ok=True)
    plot_path = os.path.join(artifact_dir, "stage_b5_frequency_generalization.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved generalization comparison plot to {plot_path}")
    
    # 6. Append results to inverse_rain_rate.md
    report_path = "/home/satyansh/leo_meo/docs/inverse_rain_rate.md"
    with open(report_path, "a") as f:
        f.write("\n## Stage B.5: Frequency-Aware XGBoost Narrowcaster\n\n")
        f.write("Stage B.5 introduces explicit physical carrier frequency parameters into the feature set to solve the bottleneck of cross-frequency transferability:\n")
        f.write("1. **Multi-Frequency Training**: Models trained on simulated datasets spanning $10$, $12$, $14$, $20$, and $30\\text{ GHz}$.\n")
        f.write("2. **Explicit Attenuation Coupling Features**: Features now explicitly include the carrier frequency ($f_c$), along with the ITU-R P.838 coefficients $k$ and $\\alpha$, and the dynamic effective path length $L_{\\text{eff}}$.\n\n")
        
        f.write("### Cross-Frequency Performance Comparison (R² and RMSE)\n\n")
        f.write("Evaluating the generalization of the $14\\text{ GHz}$ trained model (Stage B) vs. the multi-frequency trained model (Stage B.5):\n\n")
        f.write("| Frequency | Stage B (Unaware) R² | Stage B.5 (Aware) R² | Stage B (Unaware) RMSE | Stage B.5 (Aware) RMSE | Stage B.5 F1 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for m in freq_metrics:
            un_r2 = f"{m['unaware_r2']:.4f}" if not np.isnan(m["unaware_r2"]) else "N/A"
            un_rmse = f"{m['unaware_rmse']:.4f}" if not np.isnan(m["unaware_rmse"]) else "N/A"
            f.write(f"| {m['freq']:.0f} GHz | {un_r2} | {m['r2']:.4f} | {un_rmse} | {m['rmse']:.4f} | {m['f1']:.4f} |\n")
        
        f.write("\n### Visual Validation\n\n")
        f.write("#### Cross-Frequency Generalization Improvement Plot\n")
        f.write("![Cross Frequency Generalization Comparison](plots/stage_b5_frequency_generalization.png)\n\n")
        
    print(f"Completed Stage B.5 evaluation. Report updated in {report_path}")

if __name__ == "__main__":
    run_stage_b5()
