# Model Computational Efficiency Benchmark Study

This document tracks model parameters, training wall-clock time, GPU inference latency (ms per 1,000 samples), and $R^2$ accuracy across all model architectures.

## 1. Computational Efficiency Comparison Matrix

| Model Architecture | Trainable Parameters | Training Time (s) | Inference Latency (ms / 1k) | Regressor $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Analytical Inversion (Stage A)** | 0 (Physics Equation) | 0.00 s | 0.08 ms / 1k | **0.1110** |
| **XGBoost (Raw Only)** | 3,000 Nodes (200 Trees) | 3.25 s | 0.49 ms / 1k | **0.2924** |
| **XGBoost (With Rolling)** | 3,000 Nodes (200 Trees) | 1.18 s | 0.30 ms / 1k | **0.4996** |
| **MLP (Raw Only)** | 44,482 | 16.50 s | 5.24 ms / 1k | **0.2552** |
| **MLP (With Rolling)** | 50,114 | 15.94 s | 5.42 ms / 1k | **0.5346** |
| **Bai et al. TCN (Raw Sequence)** | 147,746 | 75.84 s | 29.68 ms / 1k | **0.4496** |
| **Bai et al. TCN (With Rolling)** | 150,562 | 76.54 s | 27.66 ms / 1k | **0.5214** |
| **PINN-TCN (Physics Loss)** | 150,822 | 73.73 s | 27.26 ms / 1k | **0.5257** |
| **Deep Ensemble (5 Models)** | 752,810 (5x TCN) | 217.46 s | 134.75 ms / 1k | **0.5643** |
