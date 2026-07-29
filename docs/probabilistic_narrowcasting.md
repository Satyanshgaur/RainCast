# Probabilistic Narrowcasting & Physics-Informed Loss Study

This document tracks the implementation and evaluation of probabilistic uncertainty estimation methods (Quantile Regression, NGBoost, Bayesian Neural Networks, Deep Ensembles) and Physics-Informed Loss Functions ($A \approx k R^\alpha L$) for satellite rain narrowcasting.

## 1. Physics-Informed Loss Function (PINN-TCN)
Instead of purely minimizing data MSE, the model penalizes predictions that violate ITU-R P.618 propagation equations:

$$\hat{A}_{\text{physics}} = k \cdot (\hat{R})^\alpha \cdot L_{\text{eff}}$$
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{physics}} \cdot \| \hat{A}_{\text{physics}} - A_{\text{excess}} \|^2$$

- **PINN-TCN Regressor RMSE**: **4.9681 mm/h**
- **PINN-TCN Regressor MAE**: **2.5211 mm/h**
- **PINN-TCN Regressor $R^2$ Score**: **0.5076**

## 2. Probabilistic Models Benchmark Summary

| Probabilistic Method | Uncertainty Representation | Point RMSE (mm/h) | Interval Coverage / Uncertainty Bounds | Regressor $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **Quantile Regression (q10, q50, q90)** | Pinball Loss (80% Target Interval) | 5.1498 | **81.86% Coverage** (±2.59 mm/h) | 0.4710 |
| **Physics-Informed TCN (PINN)** | Forward Attenuation Physics Constraint | **4.9681** | Direct Physics Penalty | **0.5076** |
| **Bayesian NN (MC Dropout 50 Passes)** | Epistemic Model Parameter Sampling | 6.2443 | Epistemic Bounds (±1.5445 mm/h) | 0.2222 |
| **Deep Ensembles (5 Diverse Seeds)** | Aleatoric + Epistemic Joint Sampling | **4.7169** | Total Ensemble Bounds (±0.6537 mm/h) | **0.5562** |
