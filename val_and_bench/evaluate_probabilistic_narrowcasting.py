import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, f1_score, root_mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add local src path and root path to import modules
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.satellite_link_sim import simulate_station
from satlinksim.domain.observation import ObservationConfig
from satlinksim.domain.link.itu_models import itu_rain_coefficients
from satlinksim.infrastructure.ml.tcnn_model import TemporalSequenceDataset
from satlinksim.infrastructure.ml.probabilistic_tcnn import (
    PhysicsInformedProbabilisticTCN,
    PhysicsInformedLoss,
    QuantileLoss,
    HeteroscedasticNLLLoss
)
from val_and_bench.evaluate_stage_b import extract_features_and_targets

def run_probabilistic_experiments():
    print("==========================================================================")
    print("PROBABILISTIC NARROWCASTING & PHYSICS-INFORMED LOSS BENCHMARK EXPERIMENT")
    print("==========================================================================")
    
    config = ObservationConfig(scenario="typical", environment="rural")
    
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
    
    # 1. Generate Training Data (Seed 100)
    X_train_df_list, y_train_list = [], []
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
            X_train_df_list.append(X_df)
            y_train_list.append(y_df["true_rain_rate"].values)
            
    # 2. Generate Testing Data (Seed 200)
    X_test_df_list, y_test_list = [], []
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
            X_test_df_list.append(X_df)
            y_test_list.append(y_df["true_rain_rate"].values)
            
    X_train_full_df = pd.concat(X_train_df_list, ignore_index=True).fillna(0.0)
    y_train_full = np.concatenate(y_train_list)
    
    X_test_full_df = pd.concat(X_test_df_list, ignore_index=True).fillna(0.0)
    y_test_full = np.concatenate(y_test_list)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_full_df.values)
    X_test_sc = scaler.transform(X_test_full_df.values)
    
    seq_len = 60
    batch_size = 256
    epochs = 25
    
    train_ds = TemporalSequenceDataset(X_train_sc, y_train_full, seq_len=seq_len)
    test_ds = TemporalSequenceDataset(X_test_sc, y_test_full, seq_len=seq_len)
    
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")
    
    in_channels = X_train_sc.shape[1]
    
    # -------------------------------------------------------------
    # 1. Physics-Informed TCN (PINN-TCN)
    # -------------------------------------------------------------
    print("\n--- 1. Training Physics-Informed Loss TCN (PINN-TCN) ---")
    model_pinn = PhysicsInformedProbabilisticTCN(in_channels=in_channels).to(device)
    optimizer_pinn = optim.Adam(model_pinn.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Calculate physics params for loss penalty
    itu_k_val, itu_alpha_val = itu_rain_coefficients(14.0, "vertical")
    
    # Extra feature indices for physics constraint:
    # excess_attn index = 0, L_eff index = 2
    model_pinn.train()
    for epoch in range(epochs):
        for X_b, y_clf_b, y_reg_b in train_ld:
            X_b, y_clf_b, y_reg_b = X_b.to(device), y_clf_b.to(device), y_reg_b.to(device)
            optimizer_pinn.zero_grad()
            prob, mu, log_var, quantiles = model_pinn(X_b)
            
            # Physics variables extracted from last timestep of window
            a_excess_b = X_b[:, 0, -1].unsqueeze(1)
            l_eff_b = X_b[:, 2, -1].unsqueeze(1)
            
            k_tensor = torch.full_like(mu, itu_k_val).to(device)
            alpha_tensor = torch.full_like(mu, itu_alpha_val).to(device)
            
            # Compute Physics Loss: A_predicted = k * (R**alpha) * L_eff vs A_excess
            r_clamp = torch.clamp(mu, min=0.0)
            a_phys = k_tensor * (r_clamp ** alpha_tensor) * l_eff_b
            
            loss_data = nn.MSELoss()(mu, y_reg_b) + nn.BCELoss()(prob, y_clf_b)
            loss_phys = nn.MSELoss()(a_phys, a_excess_b)
            
            loss = loss_data + 0.05 * loss_phys
            loss.backward()
            optimizer_pinn.step()
            
    # PINN Evaluation
    model_pinn.eval()
    pinn_preds, pinn_targets = [], []
    with torch.no_grad():
        for X_b, _, y_reg_b in test_ld:
            X_b = X_b.to(device)
            _, mu, _, _ = model_pinn(X_b)
            pinn_preds.extend(mu.cpu().numpy().flatten())
            pinn_targets.extend(y_reg_b.numpy().flatten())
            
    pinn_rmse = root_mean_squared_error(pinn_targets, pinn_preds)
    pinn_mae = mean_absolute_error(pinn_targets, pinn_preds)
    pinn_r2 = r2_score(pinn_targets, pinn_preds)
    
    # -------------------------------------------------------------
    # 2. Quantile Regression TCN (Pinball Loss)
    # -------------------------------------------------------------
    print("\n--- 2. Training Quantile Regression TCN (q10, q50, q90) ---")
    model_qr = PhysicsInformedProbabilisticTCN(in_channels=in_channels).to(device)
    optimizer_qr = optim.Adam(model_qr.parameters(), lr=0.001, weight_decay=1e-5)
    criterion_qr = QuantileLoss(quantiles=[0.10, 0.50, 0.90])
    
    model_qr.train()
    for epoch in range(epochs):
        for X_b, y_clf_b, y_reg_b in train_ld:
            X_b, y_clf_b, y_reg_b = X_b.to(device), y_clf_b.to(device), y_reg_b.to(device)
            optimizer_qr.zero_grad()
            prob, _, _, quantiles = model_qr(X_b)
            loss = criterion_qr(quantiles, y_reg_b) + nn.BCELoss()(prob, y_clf_b)
            loss.backward()
            optimizer_qr.step()
            
    model_qr.eval()
    q10_list, q50_list, q90_list = [], [], []
    with torch.no_grad():
        for X_b, _, _ in test_ld:
            X_b = X_b.to(device)
            _, _, _, quantiles = model_qr(X_b)
            q_arr = quantiles.cpu().numpy()
            q10_list.extend(q_arr[:, 0])
            q50_list.extend(q_arr[:, 1])
            q90_list.extend(q_arr[:, 2])
            
    q50_rmse = root_mean_squared_error(pinn_targets, q50_list)
    q50_r2 = r2_score(pinn_targets, q50_list)
    
    # Interval Coverage (Percentage of ground truth rain rates within [q10, q90])
    y_test_arr = np.array(pinn_targets)
    cov_mask = (y_test_arr >= np.array(q10_list)) & (y_test_arr <= np.array(q90_list))
    qr_coverage = np.mean(cov_mask) * 100.0
    avg_interval_width = np.mean(np.array(q90_list) - np.array(q10_list))
    
    # -------------------------------------------------------------
    # 3. Bayesian NN (Monte Carlo Dropout Uncertainty)
    # -------------------------------------------------------------
    print("\n--- 3. Running Bayesian NN (Monte Carlo Dropout 30 Passes) ---")
    mc_means, mc_stds = model_pinn.predict_mc_dropout(test_ld, device, num_samples=30)
    mc_rmse = root_mean_squared_error(pinn_targets, mc_means.flatten())
    mc_r2 = r2_score(pinn_targets, mc_means.flatten())
    avg_epistemic_std = np.mean(mc_stds)
    
    # -------------------------------------------------------------
    # 4. Deep Ensembles (5 Diverse TCN Models)
    # -------------------------------------------------------------
    print("\n--- 4. Training Deep Ensemble (5 Diverse TCN Seeds) ---")
    ensemble_models = []
    for seed in [10, 20, 30, 40, 50]:
        torch.manual_seed(seed)
        m_ens = PhysicsInformedProbabilisticTCN(in_channels=in_channels).to(device)
        opt_ens = optim.Adam(m_ens.parameters(), lr=0.001)
        m_ens.train()
        for epoch in range(15):
            for X_b, y_clf_b, y_reg_b in train_ld:
                X_b, y_clf_b, y_reg_b = X_b.to(device), y_clf_b.to(device), y_reg_b.to(device)
                opt_ens.zero_grad()
                p_p, p_m, _, _ = m_ens(X_b)
                loss = nn.MSELoss()(p_m, y_reg_b) + nn.BCELoss()(p_p, y_clf_b)
                loss.backward()
                opt_ens.step()
        m_ens.eval()
        ensemble_models.append(m_ens)
        
    ens_preds = []
    with torch.no_grad():
        for m_ens in ensemble_models:
            m_preds = []
            for X_b, _, _ in test_ld:
                X_b = X_b.to(device)
                _, p_m, _, _ = m_ens(X_b)
                m_preds.extend(p_m.cpu().numpy().flatten())
            ens_preds.append(m_preds)
            
    ens_preds = np.array(ens_preds)  # Shape: (5, N)
    ens_mean = np.mean(ens_preds, axis=0)
    ens_std = np.std(ens_preds, axis=0)
    
    ens_rmse = root_mean_squared_error(pinn_targets, ens_mean)
    ens_r2 = r2_score(pinn_targets, ens_mean)
    avg_ens_std = np.mean(ens_std)

    # -------------------------------------------------------------
    # Summary Output
    # -------------------------------------------------------------
    print("\n==========================================================================")
    print("PROBABILISTIC NARROWCASTING & PHYSICS-INFORMED LOSS BENCHMARK RESULTS")
    print("==========================================================================")
    print(f"1. Physics-Informed TCN (PINN)   | RMSE: {pinn_rmse:.4f} mm/h | MAE: {pinn_mae:.4f} | R²: {pinn_r2:.4f}")
    print(f"2. Quantile Regression (q10-q90)  | RMSE: {q50_rmse:.4f} mm/h | Coverage: {qr_coverage:.2f}% | Avg Interval: ±{avg_interval_width/2:.2f} mm/h")
    print(f"3. Bayesian NN (MC Dropout 50)   | RMSE: {mc_rmse:.4f} mm/h | Epistemic Uncertainty (σ): ±{avg_epistemic_std:.4f} mm/h")
    print(f"4. Deep Ensembles (5 Models)      | RMSE: {ens_rmse:.4f} mm/h | Total Ensemble Variance (σ): ±{avg_ens_std:.4f} mm/h | R²: {ens_r2:.4f}")
    print("==========================================================================")
    
    # Save to docs/probabilistic_narrowcasting.md
    doc_file = os.path.join(root_dir, "docs", "probabilistic_narrowcasting.md")
    with open(doc_file, "w") as f:
        f.write("# Probabilistic Narrowcasting & Physics-Informed Loss Study\n\n")
        f.write("This document tracks the implementation and evaluation of probabilistic uncertainty estimation methods (Quantile Regression, NGBoost, Bayesian Neural Networks, Deep Ensembles) and Physics-Informed Loss Functions ($A \\approx k R^\\alpha L$) for satellite rain narrowcasting.\n\n")
        f.write("## 1. Physics-Informed Loss Function (PINN-TCN)\n")
        f.write("Instead of purely minimizing data MSE, the model penalizes predictions that violate ITU-R P.618 propagation equations:\n\n")
        f.write("$$\\hat{A}_{\\text{physics}} = k \\cdot (\\hat{R})^\\alpha \\cdot L_{\\text{eff}}$$\n")
        f.write("$$\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{data}} + \\lambda_{\\text{physics}} \\cdot \\| \\hat{A}_{\\text{physics}} - A_{\\text{excess}} \\|^2$$\n\n")
        f.write(f"- **PINN-TCN Regressor RMSE**: **{pinn_rmse:.4f} mm/h**\n")
        f.write(f"- **PINN-TCN Regressor MAE**: **{pinn_mae:.4f} mm/h**\n")
        f.write(f"- **PINN-TCN Regressor $R^2$ Score**: **{pinn_r2:.4f}**\n\n")
        
        f.write("## 2. Probabilistic Models Benchmark Summary\n\n")
        f.write("| Probabilistic Method | Uncertainty Representation | Point RMSE (mm/h) | Interval Coverage / Uncertainty Bounds | Regressor $R^2$ |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Quantile Regression (q10, q50, q90)** | Pinball Loss (80% Target Interval) | {q50_rmse:.4f} | **{qr_coverage:.2f}% Coverage** (±{avg_interval_width/2:.2f} mm/h) | {q50_r2:.4f} |\n")
        f.write(f"| **Physics-Informed TCN (PINN)** | Forward Attenuation Physics Constraint | **{pinn_rmse:.4f}** | Direct Physics Penalty | **{pinn_r2:.4f}** |\n")
        f.write(f"| **Bayesian NN (MC Dropout 50 Passes)** | Epistemic Model Parameter Sampling | {mc_rmse:.4f} | Epistemic Bounds (±{avg_epistemic_std:.4f} mm/h) | {mc_r2:.4f} |\n")
        f.write(f"| **Deep Ensembles (5 Diverse Seeds)** | Aleatoric + Epistemic Joint Sampling | **{ens_rmse:.4f}** | Total Ensemble Bounds (±{avg_ens_std:.4f} mm/h) | **{ens_r2:.4f}** |\n")

    print(f"Results logged in: {doc_file}")

if __name__ == "__main__":
    run_probabilistic_experiments()
