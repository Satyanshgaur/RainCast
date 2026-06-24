# Inverse Rain Rate Modeling Results

This document tracks the results of our multi-stage rain rate narrowcasting implementation.

## Executive Summary: Stage A vs. Stage B Performance

The following table summarizes the overall regression performance of the analytical baseline (Stage A) vs. the cascaded XGBoost model (Stage B) on the test dataset (across all ground stations, stochastic rain scenarios):

| Model | RMSE | MAE | Correlation | R² |
| :--- | :--- | :--- | :--- | :--- |
| **Analytical Inversion (Stage A)** | 2.10 | 0.76 | 0.346 | 0.111 |
| **XGBoost Cascade (Stage B)** | 0.49 | 0.06 | 0.997 | 0.995 |

This work demonstrates that rain rate can be inferred from link telemetry through a hybrid physics-ML pipeline. A purely analytical inversion achieves high recall but suffers from significant false-positive rates due to scintillation leakage. A cascaded XGBoost architecture reduces these errors while preserving physical consistency, achieving $R^2 = 0.9945$ and maintaining robustness under climate, station, and simulator parameter shifts.


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
* **Stochastic Rain Time-Series**: ![Time Series](plots/stage_a_delhi_stochastic_rain.png)
* **Histogram Distribution Comparison**: ![Histogram](plots/stage_a_delhi_stochastic_rain_hist.png)
* **Precision-Recall Analysis**: ![PR Analysis](plots/stage_a_delhi_pr_analysis.png)

#### Sao Paulo
* **Stochastic Rain Time-Series**: ![Time Series](plots/stage_a_saopaulo_stochastic_rain.png)
* **Histogram Distribution Comparison**: ![Histogram](plots/stage_a_saopaulo_stochastic_rain_hist.png)
* **Precision-Recall Analysis**: ![PR Analysis](plots/stage_a_saopaulo_pr_analysis.png)


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
* **XGBoost Predicted Time-Series**: ![XGBoost Time Series](plots/stage_b_delhi_stochastic_rain.png)
* **XGBoost Distribution Comparison**: ![XGBoost Histogram](plots/stage_b_delhi_stochastic_rain_hist.png)
* **PR Curve Comparison (Stage A vs Stage B)**: ![PR Curve Comparison](plots/stage_b_delhi_pr_comparison.png)

#### Sao Paulo Stage B Plots
* **XGBoost Predicted Time-Series**: ![XGBoost Time Series](plots/stage_b_saopaulo_stochastic_rain.png)
* **XGBoost Distribution Comparison**: ![XGBoost Histogram](plots/stage_b_saopaulo_stochastic_rain_hist.png)
* **PR Curve Comparison (Stage A vs Stage B)**: ![PR Curve Comparison](plots/stage_b_saopaulo_pr_comparison.png)


## Stage B Generalization and Robustness Validation

Following the Stage B Validation checklist, this section documents the generalization, leakage, and cross-domain validation testing:

### 1. Feature Importance Analysis

Below are the top 10 most important features for the XGBoost Regressor model:

| Feature | Importance |
|---|---|
| excess_attn | 0.7663 |
| rolling_max_30s | 0.0764 |
| lag_excess_attn_10s | 0.0373 |
| rolling_max_300s | 0.0292 |
| rolling_mean_30s | 0.0266 |
| rolling_mean_60s | 0.0087 |
| elevation | 0.0085 |
| rolling_max_60s | 0.0075 |
| L_eff | 0.0066 |
| rolling_min_300s | 0.0054 |

### 2. Feature Leakage & Ablation Study

Evaluating the model after stripping groups of features to ensure it is not overly reliant on raw/direct simulator attenuation mapping:

| Ablation Group | RMSE (mm/h) | MAE (mm/h) | Correlation | R² Score | F1 (0.1 mm/h) |
|---|---|---|---|---|---|
| All Features | 0.4934 | 0.0647 | 0.9973 | 0.9945 | 0.9992 |
| No Rolling Stats | 0.8430 | 0.0875 | 0.9921 | 0.9841 | 0.9986 |
| No Excess Attn & L_eff | 6.3290 | 3.8117 | 0.3270 | 0.1011 | 0.6891 |
| No Climatology | 0.4934 | 0.0647 | 0.9973 | 0.9945 | 0.9992 |

### 3. Leave-One-Station-Out (LOSO) Generalization

Evaluating how well the narrowcaster generalizes to a completely unseen geographic location (cross-climate validation):

| Excluded Ground Station | RMSE (mm/h) | MAE (mm/h) | Correlation | R² Score | F1 Score (0.1 mm/h) |
|---|---|---|---|---|---|
| Delhi | 0.0241 | 0.0041 | 0.9998 | 0.9986 | 0.9960 |
| Sao Paulo | 0.8529 | 0.1180 | 0.9930 | 0.9504 | 1.0000 |
| Tokyo | 0.0975 | 0.0192 | 0.9988 | 0.9976 | 1.0000 |
| Berlin | 0.1754 | 0.0272 | 0.9816 | 0.9148 | 0.9913 |

### 4. Cross-Frequency Generalization

Testing the model pre-trained at 14 GHz on unseen high-frequency channels (12 GHz, 20 GHz, 30 GHz) without retraining:

| Test Channel Frequency | RMSE (mm/h) | MAE (mm/h) | Correlation | R² Score | F1 Score (0.1 mm/h) |
|---|---|---|---|---|---|
| 12 GHz | 2.1962 | 1.0292 | 0.9956 | 0.8978 | 0.9967 |
| 20 GHz | 3.6022 | 1.9971 | 0.9745 | 0.7250 | 0.9994 |
| 30 GHz | 7.7491 | 4.7598 | 0.9293 | -0.2727 | 0.9993 |

### 5. Distribution Matching

To verify that the model produces physically realistic distributions rather than simply smoothing outliers:

* **Jensen-Shannon (JS) Divergence** ($P(R) \parallel P(\widehat{R})$) for Sao Paulo Stochastic Rain: **0.08297** *(where 0.0 is a perfect match)*.

### 6. Simulator Parameter Modification (Noise Shifts)

Testing robustness when evaluated against a simulator run with modified dynamics (Rain coherence time tau_c = 600s, rain scale 1.5x, and injected 2.0x nominal scintillation power):

* **RMSE**: 2.3367 mm/h
* **Correlation ($R$)**: 0.9766
* **R² Score**: 0.9384
* **F1 Score**: 0.9998

### Stage B Conclusions
- **Architecture Superiority**: The cascaded XGBoost architecture substantially outperforms analytical inversion across all evaluated stochastic-rain scenarios.
- **Geographic Generalization**: Leave-One-Station-Out validation demonstrates strong geographic generalization, indicating that the model is learning attenuation-to-rain relationships rather than station-specific climatology.
- **Parametric Robustness**: Robustness testing under modified scintillation power, rain coherence times, and rain severity shows that the model remains stable under significant simulator parameter shifts.
- **Statistical Fidelity**: Distribution matching analysis (JS divergence = 0.083) indicates that the model reproduces realistic rain-rate statistics rather than simply minimizing regression error.
- **Frequency Transferability Limit**: The primary limitation is frequency transferability. Models trained at 14 GHz exhibit significant degradation at higher frequencies, particularly 30 GHz, suggesting that attenuation-frequency coupling must be explicitly incorporated into training.


## Stage B.5: Frequency-Aware XGBoost Narrowcaster

Stage B.5 introduces explicit physical carrier frequency parameters into the feature set to solve the bottleneck of cross-frequency transferability:
1. **Multi-Frequency Training**: Models trained on simulated datasets spanning $10$, $12$, $14$, $20$, and $30\text{ GHz}$.
2. **Explicit Attenuation Coupling Features**: Features now explicitly include the carrier frequency ($f_c$), along with the ITU-R P.838 coefficients $k$ and $\alpha$, and the dynamic effective path length $L_{\text{eff}}$.

### Cross-Frequency Performance Comparison (R² and RMSE)

Evaluating the generalization of the $14\text{ GHz}$ trained model (Stage B) vs. the multi-frequency trained model (Stage B.5):

| Frequency | Stage B (Unaware) R² | Stage B.5 (Aware) R² | Stage B (Unaware) RMSE | Stage B.5 (Aware) RMSE | Stage B.5 F1 |
|---|---|---|---|---|---|
| 10 GHz | N/A | 0.9990 | N/A | 0.2105 | 0.9982 |
| 12 GHz | 0.8978 | 0.9990 | 2.1962 | 0.2087 | 0.9992 |
| 14 GHz | 0.9945 | 0.9982 | 0.4934 | 0.2847 | 0.9996 |
| 20 GHz | 0.7250 | 0.9943 | 3.6022 | 0.5046 | 0.9998 |
| 30 GHz | -0.2727 | 0.9804 | 7.7491 | 0.9349 | 0.9999 |

### Visual Validation

#### Cross-Frequency Generalization Improvement Plot
![Cross Frequency Generalization Comparison](plots/stage_b5_frequency_generalization.png)

## Scientific Findings

* **Finding 1**: Pure analytical inversion is insufficient for realistic rain-rate estimation. Tropospheric scintillation creates significant false-positive rain detections.
* **Finding 2**: Temporal attenuation statistics contain enough information to distinguish rain from scintillation.
* **Finding 3**: Rain attenuation inversion is fundamentally frequency-dependent. Single-frequency models do not generalize across communication bands.
* **Finding 4**: Embedding physical attenuation parameters directly into the learning process restores cross-frequency generalization.

---

## Real-World GPM/IMERG Validation & Simulator Analytic Flaws

### Key Findings

#### Finding 1: ITU-R P.837-7 underestimates extreme rainfall for Delhi.

| Source | $R_{0.01}$ |
| :--- | :---: |
| **ITU-R** | 42 mm/h |
| **NASA GPM** | 90 mm/h |

#### Finding 2: The original simulator under-produced extreme rain rates.

| Station | Target | Original |
| :--- | :---: | :---: |
| **Delhi** | 42 | 22 |
| **Tokyo** | 80 | 60 |
| **Berlin** | 28 | 19 |
| **Sao Paulo** | 95 | 80 |

#### Finding 3: Two independent biases were identified:
1. **Quantile fitting bias** (Static $P_{\text{rain}}$ quantiles in probit calculation)
2. **Event reset bias** (Lognormal temporal Markov chain initialization reset to the median)

#### Finding 4: Correcting the quantile fitting significantly improves tail reproduction.

| Station | Original | Corrected |
| :--- | :---: | :---: |
| **Delhi** | 22 | 28 |
| **Tokyo** | 60 | 66 |
| **Berlin** | 19 | 22 |
| **Sao Paulo** | 80 | 81 |

### 1. The NASA GPM IMERG vs. ITU-R Discrepancy
Real-world validation using NASA's Global Precipitation Measurement (GPM) Integrated Multi-satellitE Retrievals (IMERG) data indicates a significant gap between theoretical ITU-R P.837-7 climatological statistics and actual precipitation intensities. For instance, in subtropical monsoon zones like Delhi, GPM data reveals:
* **GPM $R_{0.01}$**: **$90.0\text{ mm/h}$**
* **ITU-R P.837-7 $R_{0.01}$**: **$42.0\text{ mm/h}$**

This discrepancy indicates that standard ITU-R climatology maps underestimate extreme convective rainfall tails by **over $110\%$** in monsoon-heavy areas.

### 2. Discovering Simulator Rain Rate Analytic Flaws
A rigorous comparison of the simulator's output against its database configurations revealed **two critical analytical/mathematical flaws** in the simulation's rain rate generation:

#### Flaw A: Quantile Probit Fitting Error (Static $P_{\text{rain}}$ Assumption)
* **Flaw**: The simulator fits lognormal parameters ($\mu$, $\sigma$) using static normal quantiles ($z_{0.001} = 3.0902$ and $z_{0.01} = 2.3263$). These quantiles mathematically assume that the probability of rain ($P_{\text{rain}}$) is exactly $10\%$ ($0.1$) for all locations.
* **Impact**: Ground stations with lower rain fractions (e.g., Delhi, where $P_{\text{rain}} = 0.053$) suffer from distorted mapping. By ignoring $P_{\text{rain}}$ in the probit fitting, the generator spreads rainfall across too much clear sky, underestimating the $R_{0.01}$ peak.
* **Correction**: Percentiles must be computed dynamically using the inverse standard normal cumulative distribution function (probit function $Q^{-1}$):
  $$z_{0.001} = Q^{-1}\left(\frac{0.0001}{P_{\text{rain}}}\right) \quad \text{and} \quad z_{0.01} = Q^{-1}\left(\frac{0.001}{P_{\text{rain}}}\right)$$

#### Flaw B: Temporal Markov Reset (Tail Truncation Bias)
* **Flaw**: When a rain event starts, the simulator initializes the log-normal rain rate `ln_R` to the median value $\mu$ ($z=0.0$). Because average rain event durations are short (coherence time $\tau_c = 300\text{ s}$), the AR(1) random walk gets cut off and resets to the median before it can walk up to the extreme upper tail of the distribution ($z > 3.0$).
* **Impact**: This creates a severe tail truncation bias, underestimating peak rainfall rates.

#### Quantitative Verification
By running the original vs. corrected models, we isolate the impact of both flaws:

| Ground Station | Target $R_{0.01}$ | Simulated $R_{0.01}$ (Flawed) | Simulated $R_{0.01}$ (Corrected) | Total Generator Underestimation |
| :--- | :---: | :---: | :---: | :---: |
| **Delhi** | $42.00\text{ mm/h}$ | $22.07\text{ mm/h}$ | $27.80\text{ mm/h}$ | **$47.4\%$** ($47.4\%$ from Flaw A+B, leaving $33.8\%$ to Flaw B) |
| **Tokyo** | $80.00\text{ mm/h}$ | $60.17\text{ mm/h}$ | $65.86\text{ mm/h}$ | **$24.8\%$** |
| **Berlin** | $28.00\text{ mm/h}$ | $19.40\text{ mm/h}$ | $22.08\text{ mm/h}$ | **$30.7\%$** |
| **Sao Paulo** | $95.00\text{ mm/h}$ | $80.51\text{ mm/h}$ | $81.46\text{ mm/h}$ | **$15.2\%$** *(Sao Paulo $P_{rain}=0.095 \approx 0.1$, hence low Flaw A impact)* |

### 3. Real-World GPM IMERG Online Data Sources
Programmatic and manual access to GPM/IMERG validation data is available via:
1. **NASA GES DISC**: Goddard Earth Sciences Data and Information Services Center (provides HDF5/NetCDF files).
2. **Google Earth Engine**: Programmatic access to the daily or 30-minute IMERG dataset via GEE ID: `NASA/GPM_L3/IMERG_V06`.
3. **NASA Earthdata Search**: UI portal for downloading spatial-temporal IMERG grids.

### 4. Comparative Evaluation of Simulator Flaw Corrections (Delhi Station)

To evaluate the isolated and combined impact of the corrections for **Bug A (Quantile Fitting Bias)** and **Bug B (Event Reset Bias)**, we simulated the Delhi ground station for $1,000,000$ minutes (approx. 1.9 years) under three simulator configurations:
1. **Original**: Both Bug A and Bug B present (legacy simulation engine behavior).
2. **Bug A only**: Quantile fitting corrected via dynamic standard normal probit mapping, but event onset values still reset to the median rain rate $\mu$.
3. **Bug A + Bug B**: Both quantile fitting corrected and event onset values initialized using standard normal scaling (Full Correction).

The table below compares these runs against a reference series simulated directly from the **NASA GPM Target parameters** for Delhi ($R_{0.01}=90.0\text{ mm/h}$, $R_{0.1}=35.0\text{ mm/h}$, $P_{\text{rain}}=0.065$):

| Configuration | $R_{0.01}$ (mm/h) | $R_{0.001}$ (mm/h) | Mean Rain Rate (mm/h) | Average Rain Duration (s) | JS Divergence (vs GPM Target) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NASA GPM Target** | 89.85 | 150.00* | 0.3239 | 330.0 | 0.0000 |
| **Original (Bug A + B present)** | 23.37 | 42.48 | 0.1211 | 337.6 | 0.0520 |
| **Bug A only (Bug B present)** | 32.53 | 58.54 | 0.1762 | 333.1 | 0.0441 |
| **Bug A + Bug B (Fully Corrected)** | 41.64 | 81.28 | 0.1931 | 329.5 | 0.0315 |

*\* Note: The simulator truncates peak rain rates to $150.0\text{ mm/h}$ by default.*

#### Analysis and Key Takeaways:
- **Exceedance Reproduction ($R_{0.01}$)**: The Delhi ITU target is $R_{0.01} = 42.0\text{ mm/h}$. The fully corrected simulator (**Bug A + Bug B**) achieves $R_{0.01} = 41.64\text{ mm/h}$ (almost a perfect match), whereas the Original simulator was severely under-producing extreme rainfall at $23.37\text{ mm/h}$ ($44.4\%$ underestimation).
- **Distribution Distance (JS Divergence)**: Correcting both biases continuously reduces the Jensen-Shannon divergence relative to the GPM Target from $0.0520$ down to $0.0315$. This confirms a significantly higher statistical fidelity across the entire precipitation distribution.
- **Rain Event Duration**: Average event durations remain extremely stable across configurations (ranging between $329\text{ s}$ and $337\text{ s}$), showing that the onset initialization fix successfully restores variance without distorting temporal autocorrelation.


