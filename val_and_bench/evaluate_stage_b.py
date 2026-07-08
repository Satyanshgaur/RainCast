import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Ensure we can import satlinksim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from satlinksim.satellite_link_sim import simulate_station
from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.domain.link.itu_models import itu_rain_coefficients, itu_rain_height, effective_path_length, gaseous_absorption_db
from satlinksim.domain.link.budget import fspl_db, noise_power_dbw

def extract_features_and_targets(res, gs, freq_hz, bandwidth_hz, polarization, start_time):
    from satlinksim.domain.observation import ObservationModel
    
    obs_model = ObservationModel(seed=42) # Fixed seed for deterministic evaluation
    obs_data = obs_model.observe(gs, freq_hz, bandwidth_hz, polarization, res)
    
    n_steps = len(res.snr_series)
    snr_series_obs = obs_data["observed_snr_db"]
    el_series_obs = obs_data["observed_elevation_deg"]
    slant_series_obs = obs_data["observed_slant_range_km"]
    
    # Calculate physical constants
    eirp = gs["eirp_dbw"]
    g_rx = gs["g_rx_dbi"]
    noise_floor = noise_power_dbw(gs["system_temp_k"], bandwidth_hz)
    rain_h = itu_rain_height(gs["latitude"])
    itu_k, _ = itu_rain_coefficients(freq_hz / 1e9, polarization)
    
    # Subtract FSPL and gas loss using observed coordinates
    pl_obs = fspl_db(freq_hz, slant_series_obs)
    gas_loss_obs = gaseous_absorption_db(freq_hz / 1e9, el_series_obs, gs["wv_g_m3"])
    total_gain = eirp + g_rx - noise_floor
    excess_attn_obs = total_gain - snr_series_obs - pl_obs - gas_loss_obs
    
    # Effective path length
    ep_obs = effective_path_length(el_series_obs, rain_h, gs["altitude_km"], itu_k)
    
    # Build dataframe for rolling features
    df = pd.DataFrame({
        "excess_attn": excess_attn_obs,
        "elevation": el_series_obs,
        "L_eff": ep_obs,
    })
    
    features = {}
    features["excess_attn"] = excess_attn_obs
    features["elevation"] = el_series_obs
    features["L_eff"] = ep_obs
    features["freq_ghz"] = np.full(n_steps, freq_hz / 1e9)
    features["received_snr_db"] = snr_series_obs
    features["slant_range_km"] = slant_series_obs
    features["observed_snr_uncertainty_db"] = obs_data["observed_snr_uncertainty_db"]
    features["calibration_state"] = obs_data["calibration_state"]
    
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
    
    # Also save metadata for tracking
    metadata = {
        "station": [gs["name"]] * n_steps,
        "force_rain": [res.rain_fraction == 1.0] * n_steps
    }
    
    return feature_df, targets, metadata

def run_stage_b():
    print("--- Stage B: Feature Engineered XGBoost Narrowcasting ---")
    
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
    
    # Different months to represent different seasons
    months = [7, 1, 9, 11] # Delhi (July), Sao Paulo (Jan), Tokyo (Sept), Berlin (Nov)
    
    n_steps = 7200
    freq_hz = 14e9
    bandwidth_hz = 36e6
    polarization = "vertical"
    
    # 1. Generate Training Data (Seed 100)
    print("\nGenerating training dataset...")
    X_train_list, y_train_list = [], []
    for i, gs in enumerate(stations):
        for force_rain in [True, False]:
            start_time = datetime(2026, months[i], 15, 12, 0, 0, tzinfo=timezone.utc)
            res = simulate_station(
                gs, n_steps=n_steps, seed=100, freq_hz=freq_hz,
                bandwidth_hz=bandwidth_hz, polarization=polarization,
                force_rain=force_rain, start_time=start_time
            )
            X_df, y_df, _ = extract_features_and_targets(res, gs, freq_hz, bandwidth_hz, polarization, start_time)
            X_train_list.append(X_df)
            y_train_list.append(y_df)
            
    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)["true_rain_rate"].values
    
    # 2. Generate Test Data (Seed 200)
    print("Generating testing dataset...")
    X_test_list, y_test_list = [], []
    test_meta_list = []
    for i, gs in enumerate(stations):
        for force_rain in [True, False]:
            start_time = datetime(2026, months[i], 15, 12, 0, 0, tzinfo=timezone.utc)
            res = simulate_station(
                gs, n_steps=n_steps, seed=200, freq_hz=freq_hz,
                bandwidth_hz=bandwidth_hz, polarization=polarization,
                force_rain=force_rain, start_time=start_time
            )
            X_df, y_df, meta = extract_features_and_targets(res, gs, freq_hz, bandwidth_hz, polarization, start_time)
            X_test_list.append(X_df)
            y_test_list.append(y_df)
            test_meta_list.append(pd.DataFrame(meta))
            
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)["true_rain_rate"].values
    test_meta = pd.concat(test_meta_list, ignore_index=True)
    
    print(f"Train features shape: {X_train.shape}")
    print(f"Test features shape : {X_test.shape}")
    
    # 3. Train models
    print("\nTraining XGBoost Classifier (Rain Detection)...")
    # Define rain threshold as 0.1 mm/h
    y_train_class = (y_train > 0.1).astype(int)
    y_test_class = (y_test > 0.1).astype(int)
    
    clf = XGBClassifier(
        objective="binary:logistic",
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train_class)
    
    print("Training XGBoost Regressor (Rain Rate)...")
    reg = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    reg.fit(X_train, y_train)
    
    # 4. Predict
    print("\nRunning inference on test dataset...")
    pred_class = clf.predict(X_test)
    pred_raw_rate = reg.predict(X_test)
    
    # Gated model output: if classifier says clear (0), output exactly 0.0 mm/h
    pred_rain_rate = np.where(pred_class == 1, np.maximum(pred_raw_rate, 0.0), 0.0)
    
    # 5. Evaluate overall
    overall_rmse = np.sqrt(np.mean((y_test - pred_rain_rate)**2))
    overall_mae = np.mean(np.abs(y_test - pred_rain_rate))
    overall_corr = np.corrcoef(y_test, pred_rain_rate)[0, 1]
    
    print(f"\nOverall Test Performance:")
    print(f"  RMSE: {overall_rmse:.4f} mm/h")
    print(f"  MAE : {overall_mae:.4f} mm/h")
    print(f"  Corr: {overall_corr:.4f}")
    
    # 6. Detailed threshold analysis for stochastic rain scenarios
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Directories for saving artifacts
    artifact_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "plots"))
    os.makedirs(artifact_dir, exist_ok=True)
    
    results_by_station = {}
    
    for gs in [gs_delhi, gs_saopaulo]:
        # Filter test set for this ground station and stochastic_rain
        idx = (test_meta["station"] == gs["name"]) & (test_meta["force_rain"] == False)
        
        y_true_gs = y_test[idx]
        y_pred_gs = pred_rain_rate[idx]
        
        station_metrics = []
        for thresh in thresholds:
            true_b = y_true_gs > thresh
            pred_b = y_pred_gs > thresh
            
            tp = np.sum(true_b & pred_b)
            fp = np.sum((~true_b) & pred_b)
            fn = np.sum(true_b & (~pred_b))
            tn = np.sum((~true_b) & (~pred_b))
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            station_metrics.append({
                "threshold": thresh,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": prec,
                "recall": rec,
                "f1": f1
            })
            
        # Get PR Curve curves
        fine_thresholds = np.linspace(0.01, 10.0, 200)
        precisions = []
        recalls = []
        for thresh in fine_thresholds:
            true_b = y_true_gs > thresh
            pred_b = y_pred_gs > thresh
            tp = np.sum(true_b & pred_b)
            fp = np.sum((~true_b) & pred_b)
            fn = np.sum(true_b & (~pred_b))
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
            
        results_by_station[gs["name"]] = {
            "metrics": station_metrics,
            "fine_thresholds": fine_thresholds,
            "precisions": precisions,
            "recalls": recalls,
            "true_rain": y_true_gs,
            "pred_rain": y_pred_gs
        }
        
        # Plot Time Series Comparison
        plt.figure(figsize=(12, 5))
        time_axis = np.arange(len(y_true_gs)) / 3600.0
        plt.plot(time_axis, y_true_gs, label="True Rain Rate", color="green", linewidth=2)
        plt.plot(time_axis, y_pred_gs, label="XGBoost Predicted Rain Rate", color="red", linestyle="--", alpha=0.8)
        plt.title(f"Stage B XGBoost Inversion: {gs['name']} (Stochastic Rain)")
        plt.xlabel("Time (hours)")
        plt.ylabel("Rain Rate (mm/h)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(artifact_dir, f"stage_b_{gs['name'].lower().replace(' ', '')}_stochastic_rain.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Plot Histogram Comparison
        plt.figure(figsize=(8, 5))
        bins = np.linspace(0, max(max(y_true_gs), max(y_pred_gs), 5.0), 50)
        plt.hist(y_true_gs, bins=bins, alpha=0.6, label="True Rain Rate", color="green", edgecolor="black")
        plt.hist(y_pred_gs, bins=bins, alpha=0.6, label="XGBoost Predicted Rain Rate", color="red", edgecolor="black")
        plt.title(f"XGBoost Rain Rate Histogram Comparison: {gs['name']}")
        plt.xlabel("Rain Rate (mm/h)")
        plt.ylabel("Frequency (Counts)")
        plt.yscale("log", nonpositive='clip')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        hist_path = os.path.join(artifact_dir, f"stage_b_{gs['name'].lower().replace(' ', '')}_stochastic_rain_hist.png")
        plt.savefig(hist_path)
        plt.close()
        
        # Plot PR Curve Comparison (Stage A vs Stage B)
        # Load Stage A PR data by simulating or using a simplified placeholder
        # Let's recreate Stage A values on the fly to compare properly
        from scipy.signal import butter, filtfilt
        def lp_filt(data):
            nyquist = 0.5
            normal_cutoff = 0.005 / nyquist
            b, a = butter(2, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, data)
            
        # Re-run Stage A on the same test dataset slice
        # Retrieve test slice snr and el
        slice_idx = idx.values
        snr_slice = X_test.loc[slice_idx, "excess_attn"].values
        el_slice = X_test.loc[slice_idx, "elevation"].values
        ep_slice = X_test.loc[slice_idx, "L_eff"].values
        
        filtered_attn_a = lp_filt(snr_slice)
        filtered_attn_a = np.maximum(filtered_attn_a, 0.0)
        itu_k, itu_alpha = itu_rain_coefficients(freq_hz/1e9, polarization)
        pred_rain_a = (filtered_attn_a / (itu_k * np.maximum(ep_slice, 1e-6))) ** (1.0 / itu_alpha)
        
        recalls_a = []
        precisions_a = []
        for thresh in fine_thresholds:
            true_b = y_true_gs > thresh
            pred_b = pred_rain_a > thresh
            tp = np.sum(true_b & pred_b)
            fp = np.sum((~true_b) & pred_b)
            fn = np.sum(true_b & (~pred_b))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls_a.append(rec)
            precisions_a.append(prec)
            
        plt.figure(figsize=(8, 6))
        plt.plot(recalls_a, precisions_a, color="blue", linestyle=":", label="Stage A: Analytical Baseline")
        plt.plot(recalls, precisions, color="red", linewidth=2.5, label="Stage B: XGBoost Narrowcaster")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Comparison: {gs['name']}")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        pr_path = os.path.join(artifact_dir, f"stage_b_{gs['name'].lower().replace(' ', '')}_pr_comparison.png")
        plt.savefig(pr_path)
        plt.close()
        print(f"  Saved visual plot to {plot_path}")
        print(f"  Saved histogram comparison to {hist_path}")
        print(f"  Saved PR Comparison plot to {pr_path}")
        
    # Append Stage B to inverse rain rate docs
    docs_path = "/home/satyansh/leo_meo/docs/inverse_rain_rate.md"
    with open(docs_path, "a") as f:
        f.write("\n## Stage B: Feature Engineered XGBoost\n\n")
        f.write("Stage B frames the inverse problem as a cascaded supervised model to address the limits of the analytical baseline:\n")
        f.write("1. **XGBoost Classifier**: Predicts binary rain state (CLEAR vs RAIN) thresholded at $0.1\\text{ mm/h}$. Trained using rolling statistics (mean, std dev, max, min over 30s, 60s, 300s windows) of excess attenuation to separate scintillation noise from rain.\n")
        f.write("2. **XGBoost Regressor**: Predicts continuous rain rate (mm/h).\n")
        f.write("3. **Cascade Gating**: If the classifier predicts `CLEAR`, the output rain rate is forced to exactly $0.0\\text{ mm/h}$.\n\n")
        
        f.write("### Quantitative Performance Comparison (Stochastic Rain)\n\n")
        for name in ["Delhi", "Sao Paulo"]:
            f.write(f"#### {name} (Stochastic Rain)\n\n")
            f.write("| Threshold (mm/h) | TP | FP | FN | TN | Precision | Recall | F1 Score |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for m in results_by_station[name]["metrics"]:
                f.write(f"| {m['threshold']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['precision']*100:.1f}% | {m['recall']*100:.1f}% | {m['f1']:.4f} |\n")
            f.write("\n")
            
        f.write("### Visual Validation & PR Comparisons\n\n")
        for name in ["Delhi", "Sao Paulo"]:
            name_low = name.lower().replace(" ", "")
            f.write(f"#### {name} Stage B Plots\n")
            f.write(f"* **XGBoost Predicted Time-Series**: ![XGBoost Time Series](plots/stage_b_{name_low}_stochastic_rain.png)\n")
            f.write(f"* **XGBoost Distribution Comparison**: ![XGBoost Histogram](plots/stage_b_{name_low}_stochastic_rain_hist.png)\n")
            f.write(f"* **PR Curve Comparison (Stage A vs Stage B)**: ![PR Curve Comparison](plots/stage_b_{name_low}_pr_comparison.png)\n\n")
            
    print(f"\nCompleted Stage B. Report appended to {docs_path}")

if __name__ == "__main__":
    run_stage_b()
