# Bai et al. (2018) Dilated Causal TCN Experiment Study

This document evaluates the performance of the official Bai et al. (2018) Temporal Convolutional Network (TCN) architecture trained directly on raw 60-minute telemetry sequences under typical receiver impairments.

## 1. Model Architecture & Input Structure
- **Architecture**: Bai et al. (2018) Dilated Causal Residual TCN.
- **Input Window**: 60 consecutive timesteps (60 minutes of raw telemetry).
- **Input Channels**: 8 raw features (Observed SNR, Excess Attenuation, Elevation, Slant Range, SNR Uncertainty, Calibration State, Carrier Frequency, Effective Path Length).
- **Dilated Residual Blocks**: 5 stacked blocks with exponential dilations $d \in [1, 2, 4, 8, 16]$, kernel size $k=3$, and spatial dropout = 0.20.
- **Receptive Field**: $1 + 2 \times (3-1) \times (1+2+4+8+16) = 125$ timesteps ($> 60$ minutes).
- **Head**: Dual-head architecture outputting Rain Detection Probability (BCE) and Rain Rate mm/h (MSE) from final causal timestep $t=60$.

## 2. Benchmark Comparison (XGBoost Stage C vs. Bai et al. TCN)

| Model Architecture | Feature Representation | F1-Score | Regressor RMSE (mm/h) | Regressor MAE (mm/h) | Regressor $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Stage C)** | Hand-crafted Rolling Mean, Std, Lags | 0.8487 | 5.3262 | 3.0163 | 0.5162 |
| **Bai et al. (2018) TCN** | **60-Min Raw Sequence Window** | **0.8043** | **5.1951** | **2.7964** | **0.4616** |
