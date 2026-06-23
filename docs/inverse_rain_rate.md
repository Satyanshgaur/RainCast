# Inverse Rain Rate Modeling Results

This document tracks the results of our multi-stage rain rate narrowcasting implementation.

## Stage A: Pure Analytical Inversion (No ML)

The analytical inversion pipeline is formulated as:
1. **Calculate Total Gain**: $G_{\text{total}} = \text{EIRP} + G_{rx} - N_{\text{floor}}$
2. **Calculate Excess Attenuation**: $\text{Attn}_{\text{excess}} = G_{\text{total}} - \text{SNR} - \text{FSPL} - \text{GL}$
3. **Filter Scintillation**: Apply zero-phase low-pass Butterworth filter (cutoff = 0.005 Hz) to get $\widehat{\text{RA}}$
4. **Invert ITU-R P.618 Model**: $\widehat{R} = \left( \frac{\max(0, \widehat{\text{RA}})}{k \cdot L_{\text{eff}}} \right)^{1/\alpha}$

### Rain Threshold Sensitivity Analysis
Tropospheric scintillation noise mimics low-rate rain, introducing massive False Positives when using a low rain/clear threshold (e.g. $0.1\text{ mm/h}$). We analyze the sensitivity of the classification performance across different detection thresholds for stochastic rain scenarios:

#### Delhi (Stochastic Rain)

| Threshold (mm/h) | TP | FP | FN | TN | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| 0.1 | 228 | 2306 | 35 | 4631 | 9.0% | 86.7% | 0.1630 |
| 0.5 | 60 | 245 | 163 | 6732 | 19.7% | 26.9% | 0.2273 |
| 1.0 | 19 | 50 | 150 | 6981 | 27.5% | 11.2% | 0.1597 |
| 2.0 | 0 | 0 | 83 | 7117 | 0.0% | 0.0% | 0.0000 |
| 5.0 | 0 | 0 | 19 | 7181 | 0.0% | 0.0% | 0.0000 |

#### Sao Paulo (Stochastic Rain)

| Threshold (mm/h) | TP | FP | FN | TN | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| 0.1 | 633 | 5880 | 10 | 677 | 9.7% | 98.4% | 0.1769 |
| 0.5 | 624 | 4677 | 19 | 1880 | 11.8% | 97.0% | 0.2100 |
| 1.0 | 578 | 3527 | 64 | 3031 | 14.1% | 90.0% | 0.2435 |
| 2.0 | 392 | 1514 | 242 | 5052 | 20.6% | 61.8% | 0.3087 |
| 5.0 | 61 | 41 | 489 | 6609 | 59.8% | 11.1% | 0.1871 |

### Visual Validation & PR Analysis

#### Delhi
* **Stochastic Rain Time-Series**: ![Time Series](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_a_delhi_stochastic_rain.png)
* **Histogram Distribution Comparison**: ![Histogram](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_a_delhi_stochastic_rain_hist.png)
* **Precision-Recall Analysis**: ![PR Analysis](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_a_delhi_pr_analysis.png)

#### Sao Paulo
* **Stochastic Rain Time-Series**: ![Time Series](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_a_saopaulo_stochastic_rain.png)
* **Histogram Distribution Comparison**: ![Histogram](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_a_saopaulo_stochastic_rain_hist.png)
* **Precision-Recall Analysis**: ![PR Analysis](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_a_saopaulo_pr_analysis.png)


## Stage B: Feature Engineered XGBoost

Stage B frames the inverse problem as a cascaded supervised model to address the limits of the analytical baseline:
1. **XGBoost Classifier**: Predicts binary rain state (CLEAR vs RAIN) thresholded at $0.1\text{ mm/h}$. Trained using rolling statistics (mean, std dev, max, min over 30s, 60s, 300s windows) of excess attenuation to separate scintillation noise from rain.
2. **XGBoost Regressor**: Predicts continuous rain rate (mm/h).
3. **Cascade Gating**: If the classifier predicts `CLEAR`, the output rain rate is forced to exactly $0.0\text{ mm/h}$.

### Quantitative Performance Comparison (Stochastic Rain)

#### Delhi (Stochastic Rain)

| Threshold (mm/h) | TP | FP | FN | TN | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| 0.1 | 247 | 0 | 1 | 6952 | 100.0% | 99.6% | 0.9980 |
| 0.5 | 229 | 4 | 0 | 6967 | 98.3% | 100.0% | 0.9913 |
| 1.0 | 182 | 0 | 7 | 7011 | 100.0% | 96.3% | 0.9811 |
| 2.0 | 97 | 0 | 3 | 7100 | 100.0% | 97.0% | 0.9848 |
| 5.0 | 33 | 0 | 0 | 7167 | 100.0% | 100.0% | 1.0000 |

#### Sao Paulo (Stochastic Rain)

| Threshold (mm/h) | TP | FP | FN | TN | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| 0.1 | 556 | 0 | 0 | 6644 | 100.0% | 100.0% | 1.0000 |
| 0.5 | 556 | 0 | 0 | 6644 | 100.0% | 100.0% | 1.0000 |
| 1.0 | 556 | 0 | 0 | 6644 | 100.0% | 100.0% | 1.0000 |
| 2.0 | 554 | 0 | 0 | 6646 | 100.0% | 100.0% | 1.0000 |
| 5.0 | 484 | 0 | 1 | 6715 | 100.0% | 99.8% | 0.9990 |

### Visual Validation & PR Comparisons

#### Delhi Stage B Plots
* **XGBoost Predicted Time-Series**: ![XGBoost Time Series](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_b_delhi_stochastic_rain.png)
* **XGBoost Distribution Comparison**: ![XGBoost Histogram](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_b_delhi_stochastic_rain_hist.png)
* **PR Curve Comparison (Stage A vs Stage B)**: ![PR Curve Comparison](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_b_delhi_pr_comparison.png)

#### Sao Paulo Stage B Plots
* **XGBoost Predicted Time-Series**: ![XGBoost Time Series](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_b_saopaulo_stochastic_rain.png)
* **XGBoost Distribution Comparison**: ![XGBoost Histogram](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_b_saopaulo_stochastic_rain_hist.png)
* **PR Curve Comparison (Stage A vs Stage B)**: ![PR Curve Comparison](file:///home/satyansh/.gemini/antigravity-cli/brain/b30d89ad-2cb9-4e00-a92c-0ae53cdb775b/stage_b_saopaulo_pr_comparison.png)

