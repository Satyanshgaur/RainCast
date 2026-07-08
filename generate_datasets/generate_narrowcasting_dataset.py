#!/usr/bin/env python3
import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Set matplotlib to run headlessly
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, f1_score
from scipy.signal import butter, filtfilt

# Add src to python path so we can import satlinksim
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(root_dir, "src"))

from satlinksim.satellite_link_sim import simulate_station
from satlinksim.domain.link.itu_models import itu_rain_coefficients, itu_rain_height, effective_path_length, gaseous_absorption_db
from satlinksim.domain.link.budget import fspl_db, noise_power_dbw
from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.domain.observation import ObservationModel

def lowpass_butterworth(data, cutoff_freq=0.005, fs=1.0, order=2):
    """
    Apply a zero-phase low-pass Butterworth filter to smooth out scintillation.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    # Ensure normal_cutoff is strictly between 0 and 1
    normal_cutoff = min(max(normal_cutoff, 0.0001), 0.9999)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if len(data) > 3 * max(len(b), len(a)):
        return filtfilt(b, a, data)
    return data

def main():
    print("================================================================")
    # 1. Define Directories
    output_dir = os.path.join(root_dir, "datasets", "rain-narrowcasting-dataset")
    plots_dir = os.path.join(output_dir, "sample_plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Creating rain narrowcasting benchmark dataset at: {output_dir}")
    
    # 2. Get Ground Stations and assign NORAD IDs
    gs_delhi = [s for s in GROUND_STATIONS if s["name"] == "Delhi"][0].copy()
    gs_saopaulo = [s for s in GROUND_STATIONS if s["name"] == "Sao Paulo"][0].copy()
    gs_tokyo = [s for s in GROUND_STATIONS if s["name"] == "Tokyo"][0].copy()
    gs_berlin = [s for s in GROUND_STATIONS if s["name"] == "Berlin"][0].copy()
    
    gs_delhi["norad_id"] = 26766
    gs_saopaulo["norad_id"] = 33153
    gs_tokyo["norad_id"] = 26900
    gs_berlin["norad_id"] = 27380
    
    stations = [gs_delhi, gs_saopaulo, gs_tokyo, gs_berlin]
    station_months = {
        "Delhi": 7,       # Monsoon season
        "Sao Paulo": 1,   # Summer rain
        "Tokyo": 9,       # Autumn rain
        "Berlin": 11      # Late autumn showers
    }
    
    # Parameters
    frequencies_hz = [10e9, 12e9, 14e9, 20e9, 30e9]
    bandwidth_hz = 36e6
    polarization = "vertical"
    n_steps = 15000  # ~10.4 days of 1-minute steps
    seed = 42
    
    # We will accumulate dataframes for train, validation, and test splits
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    full_dfs = []
    
    print(f"\nSimulating {len(stations)} stations x {len(frequencies_hz)} frequencies x 2 modes (stochastic & forced rain)...")
    
    # 3. Simulation Loop
    for gs in stations:
        name = gs["name"]
        month = station_months[name]
        start_time = datetime(2026, month, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        climate_type = "heavy monsoon" if name == "Delhi" else ("tropical" if name == "Sao Paulo" else ("temperate" if name == "Berlin" else "subtropical"))
        
        for freq_hz in frequencies_hz:
            freq_ghz = freq_hz / 1e9
            itu_k, itu_alpha = itu_rain_coefficients(freq_ghz, polarization)
            
            for force_rain in [True, False]:
                run_id = f"sim_{name}_{freq_ghz}ghz_rain_{force_rain}"
                print(f"  Running: {run_id} ...")
                
                # Run the physics-based simulator (Physical World)
                res = simulate_station(
                    gs,
                    n_steps=n_steps,
                    seed=seed,
                    freq_hz=freq_hz,
                    bandwidth_hz=bandwidth_hz,
                    polarization=polarization,
                    force_rain=force_rain,
                    start_time=start_time
                )
                
                # Observation Model Layer: distort the physical world state
                obs_model = ObservationModel(seed=seed)
                obs_data = obs_model.observe(gs, freq_hz, bandwidth_hz, polarization, res)
                
                # Telemetry Dataset Layer: extract telemetry observables
                snr_series_obs = obs_data["observed_snr_db"]
                el_series_obs = obs_data["observed_elevation_deg"]
                slant_series_obs = obs_data["observed_slant_range_km"]
                
                # Keep true rain rate for targets
                true_rain = np.array(res.rain_series)
                
                # Calculate physical constants
                rain_h = itu_rain_height(gs["latitude"])
                
                # Calculate observed path loss and gaseous attenuation (relying on observed coordinates)
                pl_obs = fspl_db(freq_hz, slant_series_obs)
                gas_loss_obs = gaseous_absorption_db(freq_ghz, el_series_obs, gs["wv_g_m3"])
                
                # Nominal total gain
                eirp = gs["eirp_dbw"]
                g_rx = gs["g_rx_dbi"]
                noise_floor = noise_power_dbw(gs["system_temp_k"], bandwidth_hz)
                total_gain = eirp + g_rx - noise_floor
                
                # Observed excess attenuation
                excess_attn_obs = total_gain - snr_series_obs - pl_obs - gas_loss_obs
                
                # Observed effective path length through rain layer
                ep_obs = effective_path_length(el_series_obs, rain_h, gs["altitude_km"], itu_k)
                ep_safe_obs = np.maximum(ep_obs, 1e-6)
                
                # Observed specific attenuation (dB/km)
                specific_attn_obs = np.maximum(excess_attn_obs, 0.0) / ep_safe_obs
                
                # Generate timestamps
                times = [start_time + timedelta(minutes=i) for i in range(n_steps)]
                
                # Assemble DataFrame using observed parameters
                run_df = pd.DataFrame({
                    "timestamp": times,
                    "received_snr_db": snr_series_obs,
                    "carrier_frequency_ghz": np.full(n_steps, freq_ghz),
                    "elevation_angle_deg": el_series_obs,
                    "slant_range_km": slant_series_obs,
                    "observed_snr_uncertainty_db": obs_data["observed_snr_uncertainty_db"],
                    "calibration_state": obs_data["calibration_state"],
                    "fspl_db": pl_obs,
                    "gaseous_attenuation_db": gas_loss_obs,
                    "excess_attenuation_db": excess_attn_obs,
                    "effective_path_length_km": ep_obs,
                    "specific_attenuation_db_per_km": specific_attn_obs,
                    "rain_height_km": np.full(n_steps, rain_h),
                    "frequency_ghz": np.full(n_steps, freq_ghz),
                    "itu_k": np.full(n_steps, itu_k),
                    "itu_alpha": np.full(n_steps, itu_alpha),
                    "station": [name] * n_steps,
                    "climate": [climate_type] * n_steps,
                    "simulation_id": [run_id] * n_steps,
                    "rain_rate_mm_per_hr": true_rain,
                    "rain_event": (true_rain > 0.1).astype(int),
                    # Ground truth physical parameters for validation and diagnostics (hidden from ML features list)
                    "true_snr_db": np.array(res.snr_series),
                    "true_elevation_deg": np.array(res.elevation_series),
                    "true_slant_range_km": np.array(res.slant_range_series),
                    "pointing_loss_db": obs_data["_latent_pointing_loss_db"],
                    "tracking_error_deg": obs_data["_latent_tracking_error_deg"],
                    "wet_antenna_loss_db": obs_data["_latent_wet_antenna_loss_db"],
                    "multipath_loss_db": obs_data["_latent_multipath_loss_db"],
                    "calibration_error_db": obs_data["_latent_calibration_error_db"],
                    # Season and ground station details for models
                    "season_sin": np.full(n_steps, np.sin(2 * np.pi * month / 12)),
                    "season_cos": np.full(n_steps, np.cos(2 * np.pi * month / 12)),
                    "gs_latitude": np.full(n_steps, gs["latitude"]),
                    "gs_humidity": np.full(n_steps, gs["humidity_pct"]),
                    "gs_wv": np.full(n_steps, gs["wv_g_m3"]),
                    "itu_R001": np.full(n_steps, gs["itu_rain"]["R001"]),
                    "itu_P_rain": np.full(n_steps, gs["itu_rain"]["P_rain"])
                })
                
                # Compute temporal rolling window statistics per run to prevent leakage
                # SNR rolling stats (5 minutes = 5 steps, 30 minutes = 30 steps)
                run_df["snr_roll_mean_5min"] = run_df["received_snr_db"].rolling(5, min_periods=1).mean()
                run_df["snr_roll_std_5min"] = run_df["received_snr_db"].rolling(5, min_periods=1).std().fillna(0.0)
                run_df["snr_roll_max_5min"] = run_df["received_snr_db"].rolling(5, min_periods=1).max()
                run_df["snr_roll_min_5min"] = run_df["received_snr_db"].rolling(5, min_periods=1).min()
                run_df["snr_roll_mean_30min"] = run_df["received_snr_db"].rolling(30, min_periods=1).mean()
                run_df["snr_roll_std_30min"] = run_df["received_snr_db"].rolling(30, min_periods=1).std().fillna(0.0)
                
                # Attenuation rolling stats (5 minutes)
                run_df["attenuation_roll_mean"] = run_df["excess_attenuation_db"].rolling(5, min_periods=1).mean()
                run_df["attenuation_roll_std"] = run_df["excess_attenuation_db"].rolling(5, min_periods=1).std().fillna(0.0)
                
                # Deltas
                run_df["attenuation_delta"] = run_df["excess_attenuation_db"].diff().fillna(0.0)
                run_df["snr_delta"] = run_df["received_snr_db"].diff().fillna(0.0)
                
                # Save full dataset copy for plotting
                full_dfs.append(run_df)
                
                # Split chronologically: Train 70%, Validation 15%, Test 15%
                split_tr = int(n_steps * 0.70)
                split_val = int(n_steps * 0.85)
                
                train_dfs.append(run_df.iloc[:split_tr])
                val_dfs.append(run_df.iloc[split_tr:split_val])
                test_dfs.append(run_df.iloc[split_val:])

    print("\nConcatenating splits...")
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    full_dataset_df = pd.concat(full_dfs, ignore_index=True)
    
    # 4. Save Parquets
    print("Saving parquets...")
    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_dir, "validation.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
    
    print(f"  train.parquet:      {len(train_df)} rows")
    print(f"  validation.parquet: {len(val_df)} rows")
    print(f"  test.parquet:       {len(test_df)} rows")
    
    # 5. Baseline Model Training and Evaluation
    print("\nTraining baseline machine learning models...")
    
    # Features to use for models (excluding target variables and metadata labels)
    ML_FEATURES = [
        "received_snr_db",
        "carrier_frequency_ghz",
        "elevation_angle_deg",
        "slant_range_km",
        "observed_snr_uncertainty_db",
        "calibration_state",
        "fspl_db",
        "gaseous_attenuation_db",
        "excess_attenuation_db",
        "effective_path_length_km",
        "specific_attenuation_db_per_km",
        "rain_height_km",
        "itu_k",
        "itu_alpha",
        "snr_roll_mean_5min",
        "snr_roll_std_5min",
        "snr_roll_max_5min",
        "snr_roll_min_5min",
        "snr_roll_mean_30min",
        "snr_roll_std_30min",
        "attenuation_roll_mean",
        "attenuation_roll_std",
        "attenuation_delta",
        "snr_delta",
        "season_sin",
        "season_cos",
        "gs_latitude",
        "gs_humidity",
        "gs_wv",
        "itu_R001",
        "itu_P_rain"
    ]
    
    # Get Train/Test data
    X_train = train_df[ML_FEATURES]
    y_train_class = train_df["rain_event"].values
    y_train_reg = train_df["rain_rate_mm_per_hr"].values
    
    X_test = test_df[ML_FEATURES]
    y_test_class = test_df["rain_event"].values
    y_test_reg = test_df["rain_rate_mm_per_hr"].values
    
    # 5.1. Train Stage C classifier
    print("  Training XGBoost Classifier...")
    clf_c = XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=seed,
        n_jobs=-1
    )
    clf_c.fit(X_train, y_train_class)
    pred_class_stage_c = clf_c.predict(X_test)
    f1_classifier = f1_score(y_test_class, pred_class_stage_c)
    
    # 5.2. Train Stage C regressor
    print("  Training XGBoost Regressor...")
    reg_c = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=seed,
        n_jobs=-1
    )
    reg_c.fit(X_train, y_train_reg)
    pred_raw_reg_stage_c = reg_c.predict(X_test)
    
    # Gated model: if classifier predicts 0, output exactly 0.0 mm/h
    pred_rain_rate_stage_c = np.where(pred_class_stage_c == 1, np.maximum(pred_raw_reg_stage_c, 0.0), 0.0)
    
    rmse_regressor = np.sqrt(np.mean((y_test_reg - pred_rain_rate_stage_c)**2))
    mae_regressor = np.mean(np.abs(y_test_reg - pred_rain_rate_stage_c))
    r2_regressor = r2_score(y_test_reg, pred_rain_rate_stage_c)
    
    print(f"    Classifier F1: {f1_classifier:.4f}")
    print(f"    Regressor RMSE: {rmse_regressor:.4f} | MAE: {mae_regressor:.4f} | R²: {r2_regressor:.4f}")
    
    # 5.3. Evaluate Analytical Inverse Model
    print("  Running Analytical Inverse Model...")
    pred_rain_analytical_list = []
    
    for sim_id in test_df["simulation_id"].unique():
        subset = test_df[test_df["simulation_id"] == sim_id]
        excess_attn = subset["excess_attenuation_db"].values
        
        # Apply Butter filter per run to avoid cross-run boundary distortion
        filtered_attn = lowpass_butterworth(excess_attn, cutoff_freq=0.005, fs=1.0, order=2)
        filtered_attn = np.maximum(filtered_attn, 0.0)
        
        itu_k = subset["itu_k"].values
        itu_alpha = subset["itu_alpha"].values
        ep = subset["effective_path_length_km"].values
        ep_safe = np.maximum(ep, 1e-6)
        
        pred_rain = (filtered_attn / (itu_k * ep_safe)) ** (1.0 / itu_alpha)
        pred_rain_analytical_list.extend(pred_rain)
        
    pred_rain_analytical = np.array(pred_rain_analytical_list)
    rmse_analytical = np.sqrt(np.mean((y_test_reg - pred_rain_analytical)**2))
    
    # Analytical F1 score (thresholded at 0.1 mm/h)
    f1_analytical = f1_score(y_test_class, (pred_rain_analytical > 0.1).astype(int))
    print(f"    Analytical RMSE: {rmse_analytical:.4f} | F1: {f1_analytical:.4f}")
    
    # 5.4. Train Stage B model (Frequency-Unaware, trained ONLY on 14 GHz, using Stage B features)
    print("  Training Stage B models (Frequency-Unaware) for Stage B vs C comparison...")
    stage_b_features = [c for c in ML_FEATURES if c not in ["carrier_frequency_ghz", "frequency_ghz", "itu_k", "itu_alpha"]]
    
    train_df_14 = train_df[train_df["carrier_frequency_ghz"] == 14.0]
    X_train_b = train_df_14[stage_b_features]
    y_train_b_class = train_df_14["rain_event"].values
    y_train_b_reg = train_df_14["rain_rate_mm_per_hr"].values
    
    clf_b = XGBClassifier(objective="binary:logistic", n_estimators=100, learning_rate=0.1, max_depth=4, random_state=seed, n_jobs=-1)
    clf_b.fit(X_train_b, y_train_b_class)
    
    reg_b = XGBRegressor(objective="reg:squarederror", n_estimators=150, learning_rate=0.1, max_depth=5, random_state=seed, n_jobs=-1)
    reg_b.fit(X_train_b, y_train_b_reg)
    
    # Compute R2 for Stage B and Stage C models across test frequencies
    r2_scores_stage_b = {}
    r2_scores_stage_c = {}
    
    for freq in [12.0, 14.0, 20.0, 30.0]:
        test_subset = test_df[test_df["carrier_frequency_ghz"] == freq]
        y_test_sub = test_subset["rain_rate_mm_per_hr"].values
        
        # Stage B predictions (unaware)
        X_test_b = test_subset[stage_b_features]
        pred_c_b = clf_b.predict(X_test_b)
        pred_raw_r_b = reg_b.predict(X_test_b)
        pred_r_b = np.where(pred_c_b == 1, np.maximum(pred_raw_r_b, 0.0), 0.0)
        r2_scores_stage_b[freq] = r2_score(y_test_sub, pred_r_b)
        
        # Stage C predictions (aware)
        X_test_c = test_subset[ML_FEATURES]
        pred_c_c = clf_c.predict(X_test_c)
        pred_raw_r_c = reg_c.predict(X_test_c)
        pred_r_c = np.where(pred_c_c == 1, np.maximum(pred_raw_r_c, 0.0), 0.0)
        r2_scores_stage_c[freq] = r2_score(y_test_sub, pred_r_c)
        
        print(f"    Freq {freq:.0f} GHz | Stage B R²: {r2_scores_stage_b[freq]:.4f} | Stage C R²: {r2_scores_stage_c[freq]:.4f}")
        
    # 6. Save baseline_metrics.json
    baseline_metrics = {
        "analytical_inverse": {
            "rmse": float(round(rmse_analytical, 4)),
            "f1": float(round(f1_analytical, 4))
        },
        "xgboost_classifier": {
            "f1": float(round(f1_classifier, 4))
        },
        "xgboost_regressor": {
            "rmse": float(round(rmse_regressor, 4)),
            "mae": float(round(mae_regressor, 4)),
            "r2": float(round(r2_regressor, 4))
        }
    }
    
    metrics_path = os.path.join(output_dir, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(baseline_metrics, f, indent=4)
    print(f"Saved baseline metrics to: {metrics_path}")
    
    # 7. Generate Sample Plots
    print("\nGenerating sample plots...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    
    # Plot 7.1: Rain rate distribution
    plt.figure(figsize=(9, 5.5), dpi=150)
    # Log bins
    bins = np.logspace(-1, 2, 50)
    plt.hist(y_test_reg[y_test_reg > 0.1], bins=bins, alpha=0.45, label="True Rain Rate", color="#1F77B4", edgecolor="#1F77B4", log=True)
    plt.hist(pred_rain_rate_stage_c[pred_rain_rate_stage_c > 0.1], bins=bins, alpha=0.45, label="XGBoost Regressor (Stage C)", color="#FF7F0E", edgecolor="#FF7F0E", log=True)
    plt.hist(pred_rain_analytical[pred_rain_analytical > 0.1], bins=bins, alpha=0.35, label="Analytical Inverse (Stage A)", color="#2CA02C", edgecolor="#2CA02C", log=True)
    plt.xscale('log')
    plt.title("Rain Rate Distribution (Log-Log Scale, Rain Events Only)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Rain Rate (mm/h)", fontsize=11)
    plt.ylabel("Frequency (Log Counts)", fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rain_distribution.png"), dpi=150)
    plt.close()
    
    # Plot 7.2: Feature Importance
    plt.figure(figsize=(9, 7.5), dpi=150)
    importances = reg_c.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": ML_FEATURES,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)
    
    colors_imp = plt.cm.viridis(np.linspace(0.3, 0.85, 15))
    plt.barh(feat_imp_df["Feature"][-15:], feat_imp_df["Importance"][-15:], color=colors_imp, edgecolor="#CCCCCC", height=0.6)
    plt.title("XGBoost Regressor Top 15 Feature Importances", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.ylabel("Feature Name", fontsize=11)
    plt.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=150)
    plt.close()
    
    # Plot 7.3: Attenuation vs Rain Rate
    plt.figure(figsize=(9, 5.5), dpi=150)
    colors_scat = {10.0: "#1f77b4", 12.0: "#7f7f7f", 14.0: "#ff7f0e", 20.0: "#9467bd", 30.0: "#2ca02c"}
    # Sample points to make plot clean and fast
    np.random.seed(seed)
    sample_idx = np.random.choice(len(test_df), min(len(test_df), 12000), replace=False)
    sampled_df = test_df.iloc[sample_idx]
    
    for freq in sorted(sampled_df["carrier_frequency_ghz"].unique()):
        subset = sampled_df[sampled_df["carrier_frequency_ghz"] == freq]
        plt.scatter(subset["rain_rate_mm_per_hr"], subset["excess_attenuation_db"], label=f"{freq:.1f} GHz", color=colors_scat.get(freq, "#7f7f7f"), alpha=0.5, s=8)
        
    plt.title("Excess Attenuation vs True Rain Rate by Frequency", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("True Rain Rate (mm/h)", fontsize=11)
    plt.ylabel("Excess Attenuation (dB)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Carrier Frequency", loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "attenuation_vs_rain.png"), dpi=150)
    plt.close()
    
    # Plot 7.4: Temporal Train/Validation/Test Split
    visual_run_id = "sim_Delhi_14.0ghz_rain_False"
    run_df = full_dataset_df[full_dataset_df["simulation_id"] == visual_run_id].sort_values("timestamp")
    
    plt.figure(figsize=(11, 5.0), dpi=150)
    n_run = len(run_df)
    train_limit = int(n_run * 0.7)
    val_limit = int(n_run * 0.85)
    t_hours = np.arange(n_run) * 60 / 3600.0  # Convert minutes to hours
    
    plt.plot(t_hours, run_df["rain_rate_mm_per_hr"], color="#2B5C8F", linewidth=1.5, label="Rain Rate")
    
    # Shading regions
    plt.axvspan(0, t_hours[train_limit - 1], color="#EAF2F8", alpha=0.75, label="Training Split (70%)")
    plt.axvspan(t_hours[train_limit], t_hours[val_limit - 1], color="#FEF9E7", alpha=0.75, label="Validation Split (15%)")
    plt.axvspan(t_hours[val_limit], t_hours[-1], color="#E8F8F5", alpha=0.75, label="Testing Split (15%)")
    
    plt.title(f"Temporal Split Configuration (Timeline: Delhi, 14 GHz Link)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Time (Hours)", fontsize=11)
    plt.ylabel("True Rain Rate (mm/h)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "train_test_split.png"), dpi=150)
    plt.close()
    
    # Plot 7.5: Stage B vs Stage C comparison
    freqs_label = ['12 GHz', '14 GHz', '20 GHz', '30 GHz']
    r2_unaware = [r2_scores_stage_b.get(f, 0.0) for f in [12.0, 14.0, 20.0, 30.0]]
    r2_aware = [r2_scores_stage_c.get(f, 0.0) for f in [12.0, 14.0, 20.0, 30.0]]
    
    x = np.arange(len(freqs_label))
    width = 0.35
    
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.bar(x - width/2, r2_unaware, width, label='Stage B: Frequency-Unaware (Trained @ 14 GHz)', color='#E31A1C', alpha=0.75, edgecolor="#990000")
    plt.bar(x + width/2, r2_aware, width, label='Stage C: Frequency-Aware Multi-channel', color='#2CA02C', alpha=0.75, edgecolor="#006600")
    
    plt.ylabel('R² Score', fontsize=11)
    plt.title('Cross-Frequency Generalization: R² Score Comparison', fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, freqs_label, fontsize=10)
    plt.ylim(-0.4, 1.05)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(True, ls='--', alpha=0.4)
    plt.legend(loc='lower left', frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "stageB_vs_stageC.png"), dpi=150)
    plt.close()
    
    print("Sample plots created successfully.")
    
    # 8. Save metadata.json
    metadata = {
        "task": "Rain Narrowcasting",
        "primary_target": "rain_rate_mm_per_hr",
        "secondary_target": "rain_event",
        "evaluation_metrics": [
            "RMSE",
            "MAE",
            "R²",
            "F1"
        ],
        "dataset_version": "1.0.0",
        "simulator_version": "0.1.0",
        "generation_seed": seed,
        "temporal_resolution": "1-minute",
        "number_of_stations": len(stations),
        "stations": [gs["name"] for gs in stations],
        "frequencies_ghz": [f / 1e9 for f in frequencies_hz],
        "split_ratio": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15
        },
        "total_rows": len(full_dataset_df),
        "columns_count": len(full_dataset_df.columns),
        "creation_timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata.json to: {meta_path}")
    
    # 9. Save column_dictionary.csv
    column_dict_data = [
        {"Column": "timestamp", "Type": "datetime", "Units": "UTC", "Description": "Simulation timestamp (1-minute resolution)"},
        {"Column": "received_snr_db", "Type": "float", "Units": "dB", "Description": "Received signal-to-noise ratio at the receiver"},
        {"Column": "carrier_frequency_ghz", "Type": "float", "Units": "GHz", "Description": "Carrier frequency of the satellite link"},
        {"Column": "elevation_angle_deg", "Type": "float", "Units": "degrees", "Description": "Ground-to-satellite elevation angle"},
        {"Column": "slant_range_km", "Type": "float", "Units": "km", "Description": "Slant range distance from ground station to satellite"},
        {"Column": "fspl_db", "Type": "float", "Units": "dB", "Description": "Free Space Path Loss"},
        {"Column": "gaseous_attenuation_db", "Type": "float", "Units": "dB", "Description": "Atmospheric gas absorption loss (ITU-R P.676)"},
        {"Column": "excess_attenuation_db", "Type": "float", "Units": "dB", "Description": "Path attenuation beyond FSPL and gaseous losses (rain + scintillation)"},
        {"Column": "effective_path_length_km", "Type": "float", "Units": "km", "Description": "Effective path length through the rain layer (ITU-R P.618)"},
        {"Column": "specific_attenuation_db_per_km", "Type": "float", "Units": "dB/km", "Description": "Specific attenuation (excess attenuation divided by effective path length)"},
        {"Column": "rain_height_km", "Type": "float", "Units": "km", "Description": "Rain height above sea level (ITU-R P.839)"},
        {"Column": "frequency_ghz", "Type": "float", "Units": "GHz", "Description": "Carrier frequency (replicated for frequency-aware models)"},
        {"Column": "itu_k", "Type": "float", "Units": "N/A", "Description": "ITU-R P.838 specific attenuation coefficient k"},
        {"Column": "itu_alpha", "Type": "float", "Units": "N/A", "Description": "ITU-R P.838 specific attenuation exponent alpha"},
        {"Column": "station", "Type": "string", "Units": "N/A", "Description": "Name of the ground station location"},
        {"Column": "climate", "Type": "string", "Units": "N/A", "Description": "Local climate description"},
        {"Column": "simulation_id", "Type": "string", "Units": "N/A", "Description": "Unique simulation run identifier"},
        {"Column": "snr_roll_mean_5min", "Type": "float", "Units": "dB", "Description": "5-minute rolling mean of received SNR"},
        {"Column": "snr_roll_std_5min", "Type": "float", "Units": "dB", "Description": "5-minute rolling standard deviation of received SNR"},
        {"Column": "snr_roll_max_5min", "Type": "float", "Units": "dB", "Description": "5-minute rolling maximum of received SNR"},
        {"Column": "snr_roll_min_5min", "Type": "float", "Units": "dB", "Description": "5-minute rolling minimum of received SNR"},
        {"Column": "snr_roll_mean_30min", "Type": "float", "Units": "dB", "Description": "30-minute rolling mean of received SNR"},
        {"Column": "snr_roll_std_30min", "Type": "float", "Units": "dB", "Description": "30-minute rolling standard deviation of received SNR"},
        {"Column": "attenuation_roll_mean", "Type": "float", "Units": "dB", "Description": "5-minute rolling mean of excess attenuation"},
        {"Column": "attenuation_roll_std", "Type": "float", "Units": "dB", "Description": "5-minute rolling standard deviation of excess attenuation"},
        {"Column": "attenuation_delta", "Type": "float", "Units": "dB", "Description": "Difference in excess attenuation from previous minute"},
        {"Column": "snr_delta", "Type": "float", "Units": "dB", "Description": "Difference in received SNR from previous minute"},
        {"Column": "season_sin", "Type": "float", "Units": "N/A", "Description": "Sine representation of season (month of year)"},
        {"Column": "season_cos", "Type": "float", "Units": "N/A", "Description": "Cosine representation of season (month of year)"},
        {"Column": "gs_latitude", "Type": "float", "Units": "degrees", "Description": "Ground station latitude"},
        {"Column": "gs_humidity", "Type": "float", "Units": "%", "Description": "Ground station average relative humidity"},
        {"Column": "gs_wv", "Type": "float", "Units": "g/m^3", "Description": "Ground station average water vapor density"},
        {"Column": "itu_R001", "Type": "float", "Units": "mm/h", "Description": "ITU-R P.837 local rain rate exceeded 0.01% of the year"},
        {"Column": "itu_P_rain", "Type": "float", "Units": "fraction", "Description": "ITU-R P.837 annual rain probability fraction"},
        {"Column": "rain_rate_mm_per_hr", "Type": "float", "Units": "mm/h", "Description": "True instantaneous rain rate (primary regression target)"},
        {"Column": "rain_event", "Type": "int", "Units": "binary", "Description": "True rain indicator (secondary classification target; 1 if rain rate > 0.1 mm/h, else 0)"}
    ]
    col_df = pd.DataFrame(column_dict_data)
    col_dict_path = os.path.join(output_dir, "column_dictionary.csv")
    col_df.to_csv(col_dict_path, index=False)
    print(f"Saved column_dictionary.csv to: {col_dict_path}")
    
    # 10. Save feature_description.csv
    feature_desc_data = [
        {"Feature": "received_snr_db", "Type": "float", "Units": "dB", "Stage": "A/B/C", "Description": "Received signal-to-noise ratio"},
        {"Feature": "carrier_frequency_ghz", "Type": "float", "Units": "GHz", "Stage": "A/B/C", "Description": "Carrier frequency of the satellite link"},
        {"Feature": "elevation_angle_deg", "Type": "float", "Units": "degrees", "Stage": "A/B/C", "Description": "Ground-to-satellite elevation angle"},
        {"Feature": "slant_range_km", "Type": "float", "Units": "km", "Stage": "A/B/C", "Description": "Slant range distance from ground station to satellite"},
        {"Feature": "timestamp", "Type": "datetime", "Units": "UTC", "Stage": "A/B/C", "Description": "Simulation timestamp"},
        {"Feature": "fspl_db", "Type": "float", "Units": "dB", "Stage": "A", "Description": "Free Space Path Loss"},
        {"Feature": "gaseous_attenuation_db", "Type": "float", "Units": "dB", "Stage": "A", "Description": "Atmospheric gas absorption loss"},
        {"Feature": "excess_attenuation_db", "Type": "float", "Units": "dB", "Stage": "A", "Description": "Path attenuation beyond FSPL and gaseous losses"},
        {"Feature": "effective_path_length_km", "Type": "float", "Units": "km", "Stage": "A", "Description": "Effective path length through the rain layer"},
        {"Feature": "specific_attenuation_db_per_km", "Type": "float", "Units": "dB/km", "Stage": "A", "Description": "Specific attenuation (excess attenuation divided by effective path length)"},
        {"Feature": "rain_height_km", "Type": "float", "Units": "km", "Stage": "A", "Description": "Rain height above sea level"},
        {"Feature": "snr_roll_mean_5min", "Type": "float", "Units": "dB", "Stage": "B", "Description": "Rolling standard mean over 5 minutes"},
        {"Feature": "snr_roll_std_5min", "Type": "float", "Units": "dB", "Stage": "B", "Description": "Rolling standard deviation over 5 minutes"},
        {"Feature": "snr_roll_max_5min", "Type": "float", "Units": "dB", "Stage": "B", "Description": "Rolling maximum over 5 minutes"},
        {"Feature": "snr_roll_min_5min", "Type": "float", "Units": "dB", "Stage": "B", "Description": "Rolling minimum over 5 minutes"},
        {"Feature": "snr_roll_mean_30min", "Type": "float", "Units": "dB", "Stage": "B", "Description": "Rolling mean over 30 minutes"},
        {"Feature": "snr_roll_std_30min", "Type": "float", "Units": "dB", "Stage": "B", "Description": "Rolling standard deviation over 30 minutes"},
        {"Feature": "attenuation_roll_mean", "Type": "float", "Units": "dB", "Stage": "B", "Description": "Rolling excess attenuation mean over 5 minutes"},
        {"Feature": "attenuation_roll_std", "Type": "float", "Units": "dB", "Stage": "B", "Description": "Rolling excess attenuation standard deviation over 5 minutes"},
        {"Feature": "attenuation_delta", "Type": "float", "Units": "dB", "Stage": "B", "Description": "Difference in excess attenuation from previous step"},
        {"Feature": "snr_delta", "Type": "float", "Units": "dB", "Stage": "B", "Description": "Difference in SNR from previous step"},
        {"Feature": "frequency_ghz", "Type": "float", "Units": "GHz", "Stage": "C", "Description": "Replicated carrier frequency for cross-frequency models"},
        {"Feature": "itu_k", "Type": "float", "Units": "—", "Stage": "C", "Description": "ITU-R specific attenuation coefficient k"},
        {"Feature": "itu_alpha", "Type": "float", "Units": "—", "Stage": "C", "Description": "ITU-R specific attenuation exponent alpha"},
        {"Feature": "station", "Type": "string", "Units": "—", "Stage": "B/C", "Description": "Ground station name for cross-site validation"},
        {"Feature": "climate", "Type": "string", "Units": "—", "Stage": "B/C", "Description": "Climate type classification"},
        {"Feature": "simulation_id", "Type": "string", "Units": "—", "Stage": "B/C", "Description": "Simulation run identifier"}
    ]
    feat_df = pd.DataFrame(feature_desc_data)
    feat_desc_path = os.path.join(output_dir, "feature_description.csv")
    feat_df.to_csv(feat_desc_path, index=False)
    print(f"Saved feature_description.csv to: {feat_desc_path}")
    
    # 11. Save LICENSE
    license_content = """MIT License

Copyright (c) 2026 Satyansh Gaur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    license_path = os.path.join(output_dir, "LICENSE")
    with open(license_path, "w") as f:
        f.write(license_content)
    print(f"Saved LICENSE to: {license_path}")
    
    # 12. Save README.md
    readme_content = """# Rain Narrowcasting Dataset

## Objective

This dataset is designed as a benchmark for the machine learning task of **rain narrowcasting**—estimating instantaneous rainfall intensity from satellite communication link telemetry.

Unlike the general-purpose Satellite Link Time Series dataset, this dataset is **ML-ready**. All features required for model training are already extracted and organized so that a researcher can immediately begin training models without additional preprocessing.

The benchmark supports two complementary machine learning tasks:

1. **Binary Classification**
   - Determine whether it is currently raining.
   - Target: `rain_event` (0 = No Rain, 1 = Rain)

2. **Regression**
   - Estimate the instantaneous rain rate in **mm/h**.
   - Target: `rain_rate_mm_per_hr`

This dataset reproduces the Stage B and Stage C experiments from the project while remaining easy to extend with new models.

---

## Dataset Split

The dataset is divided into three parts preserving **temporal ordering** to prevent data leakage:
- `train.parquet` (70% - first 7.3 days of simulation timeline)
- `validation.parquet` (15% - middle 1.5 days)
- `test.parquet` (15% - last 1.5 days)

Each file contains the columns defined in `column_dictionary.csv`.

---

## Folder Structure

The dataset folder contains the following assets:

```text
rain-narrowcasting-dataset/
│
├── README.md
├── LICENSE
├── metadata.json
├── column_dictionary.csv
│
├── train.parquet
├── validation.parquet
├── test.parquet
│
├── feature_description.csv
├── baseline_metrics.json
│
└── sample_plots/
    ├── rain_distribution.png
    ├── feature_importance.png
    ├── attenuation_vs_rain.png
    ├── train_test_split.png
    └── stageB_vs_stageC.png
```

---

## Features to Exclude (Physics Leaking Prevention)

To prevent target leakage and simulate real operational environments, direct physical simulation internal state variables (like `rain_attenuation` or `rain_db` and `scintillation_loss` or `scint_db` in isolation) have been excluded. A satellite operator only measures the total received SNR. Models must learn to map the excess attenuation (which contains atmospheric scintillation noise) to the rain rate.

---

## Benchmark Baseline Performance

The baseline results computed on `test.parquet` are saved in `baseline_metrics.json`.
- **Analytical Inverse (Stage A)**: Butterworth lowpass filtering + physical inversion.
- **XGBoost Classifier (Stage B/C)**: Rain event detection.
- **XGBoost Regressor (Stage B/C)**: Gated cascaded regression model.

Please refer to the `baseline_metrics.json` file for the exact scores.
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"Saved README.md to: {readme_path}")
    print("================================================================")
    print("Rain narrowcasting dataset generation completed successfully!")

if __name__ == "__main__":
    main()
