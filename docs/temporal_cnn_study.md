# 1D Temporal CNN (60-Minute Sequence) Experiment Study

This document evaluates the performance of a 1D Temporal Convolutional Network (TCNN) trained directly on raw 60-minute telemetry sequences under typical receiver impairments.

## 1. Model Architecture & Input Structure
- **Input Window**: 60 consecutive timesteps (60 minutes of raw telemetry).
- **Input Channels**: 8 raw features (Observed SNR, Excess Attenuation, Elevation, Slant Range, SNR Uncertainty, Calibration State, Carrier Frequency, Effective Path Length).
- **Conv Layers**: Conv1D(8->32, k=5) -> Conv1D(32->64, k=5, s=2) -> Conv1D(64->128, k=3, s=2).
- **Head**: Dual-head architecture outputting Rain Detection Probability (BCE) and Rain Rate mm/h (MSE).

## 2. Benchmark Comparison (XGBoost Stage C vs. Temporal CNN)

| Model Architecture | Feature Representation | F1-Score | Regressor RMSE (mm/h) | Regressor MAE (mm/h) | Regressor $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Stage C)** | Rolling Mean, Rolling Std, Lags | 0.8487 | 5.3262 | 3.0163 | 0.5162 |
| **Temporal CNN (1D CNN)** | **60-Min Raw Sequence Window** | **0.8115** | **4.9787** | **2.7968** | **0.5055** |
