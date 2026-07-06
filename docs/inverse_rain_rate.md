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
| 0.1 | 242 | 2856 | 23 | 4079 | 7.8% | 91.3% | 0.1439 |
| 0.5 | 97 | 480 | 154 | 6469 | 16.8% | 38.6% | 0.2343 |
| 1.0 | 38 | 135 | 173 | 6854 | 22.0% | 18.0% | 0.1979 |
| 2.0 | 0 | 0 | 140 | 7060 | 0.0% | 0.0% | 0.0000 |
| 5.0 | 0 | 0 | 40 | 7160 | 0.0% | 0.0% | 0.0000 |

#### Sao Paulo (Stochastic Rain)

| Threshold (mm/h) | TP | FP | FN | TN | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| 0.1 | 633 | 5888 | 10 | 669 | 9.7% | 98.4% | 0.1767 |
| 0.5 | 624 | 4712 | 19 | 1845 | 11.7% | 97.0% | 0.2087 |
| 1.0 | 580 | 3564 | 62 | 2994 | 14.0% | 90.3% | 0.2424 |
| 2.0 | 405 | 1571 | 231 | 4993 | 20.5% | 63.7% | 0.3101 |
| 5.0 | 68 | 51 | 486 | 6595 | 57.1% | 12.3% | 0.2021 |

### Visual Validation & PR Analysis

#### Delhi
* **Stochastic Rain Time-Series**: ![Time Series](plots/stage_a_delhi_stochastic_rain.png)
* **Histogram Distribution Comparison**: ![Histogram](plots/stage_a_delhi_stochastic_rain_hist.png)
* **Precision-Recall Analysis**: ![PR Analysis](plots/stage_a_delhi_pr_analysis.png)

#### Sao Paulo
* **Stochastic Rain Time-Series**: ![Time Series](plots/stage_a_saopaulo_stochastic_rain.png)
* **Histogram Distribution Comparison**: ![Histogram](plots/stage_a_saopaulo_stochastic_rain_hist.png)
* **Precision-Recall Analysis**: ![PR Analysis](plots/stage_a_saopaulo_pr_analysis.png)

