# Statistical Significance Benchmark Study (Mean ± Std)

This document tracks 5-seed statistical evaluations across all model architectures to establish statistically significant performance metrics.

## 1. Statistical Significance Comparison Matrix (5-Seed Runs)

| Model Architecture | Feature Input Representation | F1-Score ($\mu \pm \sigma$) | Regressor RMSE ($\mu \pm \sigma$) | Regressor $R^2$ Score ($\mu \pm \sigma$) |
| :--- | :--- | :---: | :---: | :---: |
| **XGBoost (Raw Only)** | Standard Features | 0.726 ± 0.000 | 5.954 ± 0.000 | **0.292 ± 0.000** |
| **XGBoost (With Rolling)** | Standard Features | 0.912 ± 0.000 | 5.007 ± 0.000 | **0.500 ± 0.000** |
| **MLP (Raw Only)** | Standard Features | 0.722 ± 0.005 | 6.090 ± 0.009 | **0.260 ± 0.002** |
| **MLP (With Rolling)** | Standard Features | 0.923 ± 0.007 | 4.794 ± 0.056 | **0.541 ± 0.011** |
| **TCN (Raw Sequence)** | Standard Features | 0.809 ± 0.006 | 5.076 ± 0.062 | **0.486 ± 0.013** |
| **TCN (With Rolling)** | Standard Features | 0.911 ± 0.037 | 4.905 ± 0.046 | **0.520 ± 0.009** |
| **PINN-TCN (Physics Loss)** | Standard Features | 0.914 ± 0.010 | 4.836 ± 0.050 | **0.533 ± 0.010** |
| **Deep Ensemble (5 Models)** | Standard Features | 0.921 ± 0.000 | 4.699 ± 0.022 | **0.560 ± 0.004** |
