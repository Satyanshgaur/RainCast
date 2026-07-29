import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, f1_score, root_mean_squared_error, mean_absolute_error

# Add local src path and root path to import modules
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.satellite_link_sim import simulate_station
from satlinksim.domain.observation import ObservationConfig
from satlinksim.infrastructure.ml.tcnn_model import TemporalSequenceDataset, TemporalCNN
from val_and_bench.evaluate_stage_b import extract_features_and_targets

def run_temporal_cnn_experiment(seq_len: int = 60, epochs: int = 30, batch_size: int = 256):
    print(f"--- Training Bai et al. (2018) Dilated Causal TCN (Sequence Length: {seq_len} minutes) ---")
    
    # Configure Typical Scenario Observation Model
    config = ObservationConfig(scenario="typical", environment="rural")
    
    # Retrieve Ground Stations
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
    
    # 1. Generate Training Telemetry Data (Seed 100)
    X_train_raw_list, y_train_raw_list = [], []
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
            X_train_raw_list.append(X_df)
            y_train_raw_list.append(y_df["true_rain_rate"].values)
            
    # 2. Generate Testing Telemetry Data (Seed 200)
    X_test_raw_list, y_test_raw_list = [], []
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
            X_test_raw_list.append(X_df)
            y_test_raw_list.append(y_df["true_rain_rate"].values)
            
    # Select raw features without hand-crafted rolling stats (let Temporal CNN learn temporal dynamics)
    raw_feature_cols = [
        "excess_attn", "elevation", "L_eff", "freq_ghz",
        "received_snr_db", "slant_range_km", "observed_snr_uncertainty_db", "calibration_state"
    ]
    
    # Scale features
    scaler = StandardScaler()
    
    # Concatenate training runs
    X_train_full = pd.concat(X_train_raw_list, ignore_index=True)[raw_feature_cols].fillna(0.0).values
    y_train_full = np.concatenate(y_train_raw_list)
    
    X_test_full = pd.concat(X_test_raw_list, ignore_index=True)[raw_feature_cols].fillna(0.0).values
    y_test_full = np.concatenate(y_test_raw_list)
    
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)
    
    # Create 60-minute sequence datasets
    train_dataset = TemporalSequenceDataset(X_train_scaled, y_train_full, seq_len=seq_len)
    test_dataset = TemporalSequenceDataset(X_test_scaled, y_test_full, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    # Instantiate PyTorch Model
    in_channels = len(raw_feature_cols)
    model = TemporalCNN(in_channels=in_channels).to(device)
    
    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print(f"Dataset Size - Train Windows: {len(train_dataset)}, Test Windows: {len(test_dataset)}")
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_b, y_clf_b, y_reg_b in train_loader:
            X_b, y_clf_b, y_reg_b = X_b.to(device), y_clf_b.to(device), y_reg_b.to(device)
            optimizer.zero_grad()
            pred_prob, pred_rate = model(X_b)
            
            loss_clf = criterion_bce(pred_prob, y_clf_b)
            loss_reg = criterion_mse(pred_rate, y_reg_b)
            loss = loss_clf + 0.1 * loss_reg
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_b.size(0)
            
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            epoch_loss = running_loss / len(train_dataset)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")
            
    # Evaluation Loop
    model.eval()
    test_clf_preds, test_reg_preds = [], []
    test_clf_targets, test_reg_targets = [], []
    
    with torch.no_grad():
        for X_b, y_clf_b, y_reg_b in test_loader:
            X_b = X_b.to(device)
            pred_prob, pred_rate = model(X_b)
            test_clf_preds.extend(pred_prob.cpu().numpy().flatten())
            test_reg_preds.extend(pred_rate.cpu().numpy().flatten())
            test_clf_targets.extend(y_clf_b.numpy().flatten())
            test_reg_targets.extend(y_reg_b.numpy().flatten())
            
    test_clf_preds = np.array(test_clf_preds)
    test_reg_preds = np.array(test_reg_preds)
    test_clf_targets = np.array(test_clf_targets)
    test_reg_targets = np.array(test_reg_targets)
    
    pred_class_binary = (test_clf_preds > 0.5).astype(int)
    
    # Metrics
    f1 = f1_score(test_clf_targets, pred_class_binary, zero_division=0)
    rmse = root_mean_squared_error(test_reg_targets, test_reg_preds)
    mae = mean_absolute_error(test_reg_targets, test_reg_preds)
    r2 = r2_score(test_reg_targets, test_reg_preds)
    
    print("\n==========================================")
    print("TEMPORAL CNN (60-MIN SEQUENCE) EVALUATION RESULTS")
    print("==========================================")
    print(f"Classification F1-Score: {f1:.4f}")
    print(f"Regressor RMSE:          {rmse:.4f} mm/h")
    print(f"Regressor MAE:           {mae:.4f} mm/h")
    print(f"Regressor R² Score:      {r2:.4f}")
    print("==========================================")
    
    # Save results to docs/temporal_cnn_study.md
    docs_dir = os.path.join(root_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    study_file = os.path.join(docs_dir, "temporal_cnn_study.md")
    
    with open(study_file, "w") as f:
        f.write("# Bai et al. (2018) Dilated Causal TCN Experiment Study\n\n")
        f.write("This document evaluates the performance of the official Bai et al. (2018) Temporal Convolutional Network (TCN) architecture trained directly on raw 60-minute telemetry sequences under typical receiver impairments.\n\n")
        f.write("## 1. Model Architecture & Input Structure\n")
        f.write("- **Architecture**: Bai et al. (2018) Dilated Causal Residual TCN.\n")
        f.write("- **Input Window**: 60 consecutive timesteps (60 minutes of raw telemetry).\n")
        f.write("- **Input Channels**: 8 raw features (Observed SNR, Excess Attenuation, Elevation, Slant Range, SNR Uncertainty, Calibration State, Carrier Frequency, Effective Path Length).\n")
        f.write("- **Dilated Residual Blocks**: 5 stacked blocks with exponential dilations $d \\in [1, 2, 4, 8, 16]$, kernel size $k=3$, and spatial dropout = 0.20.\n")
        f.write("- **Receptive Field**: $1 + 2 \\times (3-1) \\times (1+2+4+8+16) = 125$ timesteps ($> 60$ minutes).\n")
        f.write("- **Head**: Dual-head architecture outputting Rain Detection Probability (BCE) and Rain Rate mm/h (MSE) from final causal timestep $t=60$.\n\n")
        f.write("## 2. Benchmark Comparison (XGBoost Stage C vs. Bai et al. TCN)\n\n")
        f.write("| Model Architecture | Feature Representation | F1-Score | Regressor RMSE (mm/h) | Regressor MAE (mm/h) | Regressor $R^2$ Score |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **XGBoost (Stage C)** | Hand-crafted Rolling Mean, Std, Lags | 0.8487 | 5.3262 | 3.0163 | 0.5162 |\n")
        f.write(f"| **Bai et al. (2018) TCN** | **60-Min Raw Sequence Window** | **{f1:.4f}** | **{rmse:.4f}** | **{mae:.4f}** | **{r2:.4f}** |\n")
        
    print(f"\nResults recorded in: {study_file}")
    return {"f1": f1, "rmse": rmse, "mae": mae, "r2": r2}

if __name__ == "__main__":
    run_temporal_cnn_experiment()
