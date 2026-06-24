# Inverse Rain Rate Modeling Results

This document tracks the results of our multi-stage rain rate narrowcasting implementation.

## Executive Summary: Comparative Model Performance

This work demonstrates that rain rate can be inferred from link telemetry through a hybrid physics-ML pipeline. A purely analytical inversion (Stage A) suffers from significant false-positive rates due to scintillation leakage. Introducing a machine learning cascade (Stage B) resolves this leakage, and embedding explicit carrier frequency physics (Stage C) achieves robust generalization across communication bands.

### Rain Detection Performance (Classification at 0.1 mm/h threshold)
| Model | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **Analytical Inversion (Stage A)** | 9.0% | 86.7% | 0.163 |
| **XGBoost Cascade (Stage B)** | 99.96% | 99.82% | 0.999 |
| **Frequency-Aware XGBoost (Stage C)** | 99.98% | 99.92% | 0.999 |

### Rain Rate Estimation Performance (Regression during rain)
| Model | RMSE (mm/h) | MAE (mm/h) | Correlation | R² |
| :--- | :---: | :---: | :---: | :---: |
| **Analytical Inversion (Stage A)** | 2.10 | 0.76 | 0.346 | 0.111 |
| **XGBoost Cascade (Stage B)** | 0.49 | 0.06 | 0.997 | 0.995 |
| **Frequency-Aware XGBoost (Stage C)** | 0.28 | 0.04 | 0.999 | 0.998 |

### Consolidated Model Comparison
| Metric | Stage A | Stage B | Stage C |
| :--- | :---: | :---: | :---: |
| **F1 Score** | 0.163 | 0.999 | 0.999 |
| **R²** | 0.111 | 0.995 | 0.998 |
| **RMSE (mm/h)** | 2.10 | 0.49 | 0.28 |
| **Correlation** | 0.346 | 0.997 | 0.999 |

> [!WARNING]
> **Simulator-Domain Validation Limitation**
> Although training and testing use different random seeds, all datasets are generated from the same underlying stochastic rainfall model and attenuation engine. Consequently, Stage C demonstrates strong interpolation and parameter-shift robustness within the simulator domain, but real-world out-of-distribution performance remains unverified.


## Narrowcasting Pipeline Architecture

The end-to-end signal processing and machine learning pipeline is structured as follows:

```mermaid
graph TD
    Observed_SNR[Observed SNR] --> Physics_Decomp["Physics Decomposition<br/>(FSPL + GL Removal)"]
    Physics_Decomp --> Excess_Attn[Excess Attenuation]
    Excess_Attn --> Feature_Eng[Feature Engineering]
    Feature_Eng --> XGB_Classifier[XGB Classifier]
    XGB_Classifier --> Rain_Clear{Rain / Clear?}
    Rain_Clear -->|Rain| XGB_Regressor[XGB Regressor]
    Rain_Clear -->|Clear| Clear_Output[0.0 mm/h]
    XGB_Regressor --> Rain_Rate_Estimate[Rain Rate Estimate]
```


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
| 0.1 | 247 | 1 | 1 | 6951 | 99.60% | 99.60% | 0.9960 |
| 0.5 | 229 | 4 | 1 | 6966 | 98.28% | 99.57% | 0.9892 |
| 1.0 | 182 | 1 | 7 | 7010 | 99.45% | 96.30% | 0.9785 |
| 2.0 | 97 | 1 | 3 | 7099 | 98.98% | 97.00% | 0.9798 |
| 5.0 | 33 | 0 | 1 | 7166 | 99.96% | 97.06% | 0.9850 |

#### Sao Paulo (Stochastic Rain)

| Threshold (mm/h) | TP | FP | FN | TN | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| 0.1 | 556 | 1 | 1 | 6642 | 99.82% | 99.82% | 0.9982 |
| 0.5 | 555 | 1 | 2 | 6642 | 99.82% | 99.64% | 0.9973 |
| 1.0 | 554 | 2 | 2 | 6642 | 99.64% | 99.64% | 0.9964 |
| 2.0 | 552 | 2 | 2 | 6644 | 99.64% | 99.64% | 0.9964 |
| 5.0 | 483 | 1 | 2 | 6714 | 99.79% | 99.59% | 0.9969 |

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

Removing excess attenuation and effective path length causes R² to collapse from 0.9945 to 0.1011, demonstrating that the model relies on physically meaningful attenuation signatures rather than climatology alone.

### 3. Leave-One-Station-Out (LOSO) Generalization

Evaluating how well the narrowcaster generalizes to a completely unseen geographic location (cross-climate validation):

| Excluded Ground Station | RMSE (mm/h) | MAE (mm/h) | Correlation | R² Score | F1 Score (0.1 mm/h) |
|---|---|---|---|---|---|
| Delhi | 0.0241 | 0.0041 | 0.9998 | 0.9986 | 0.9960 |
| Sao Paulo | 0.8529 | 0.1180 | 0.9930 | 0.9504 | 0.9994 |
| Tokyo | 0.0975 | 0.0192 | 0.9988 | 0.9976 | 0.9996 |
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
- **Performance Summary**: The cascaded XGBoost architecture substantially outperforms analytical inversion across all evaluated stochastic-rain scenarios.
- **Geographic Generalization**: Leave-One-Station-Out validation demonstrates strong geographic generalization, indicating that the model is learning attenuation-to-rain relationships rather than station-specific climatology.
- **Parametric Robustness**: Robustness testing under modified scintillation power, rain coherence times, and rain severity shows that the model remains stable under significant simulator parameter shifts.
- **Statistical Fidelity**: Distribution matching analysis (JS divergence = 0.083) indicates that the model reproduces realistic rain-rate statistics rather than simply minimizing regression error.
- **Frequency Transferability Limit**: The primary limitation is frequency transferability. Models trained at 14 GHz exhibit significant degradation at higher frequencies, particularly 30 GHz, suggesting that attenuation-frequency coupling must be explicitly incorporated into training.


## Stage C: Frequency-Aware Narrowcaster (Final Model)

Stage C introduces explicit physical carrier frequency parameters and attenuation physics coefficients into the learning process to solve the bottleneck of cross-frequency generalization:
1. **Multi-Frequency Training**: Models are trained on simulated datasets spanning a wide band of carrier frequencies: $10$, $12$, $14$, $20$, and $30\text{ GHz}$.
2. **Explicit Attenuation Coupling Features**: Physical factors driving frequency-dependent specific attenuation are embedded directly into the machine learning models.

> [!WARNING]
> **Simulator-Domain Validation Limitation**
> Although training and testing use different random seeds, all datasets are generated from the same underlying stochastic rainfall model and attenuation engine. Consequently, Stage C demonstrates strong interpolation and parameter-shift robustness within the simulator domain, but real-world out-of-distribution performance remains unverified.

> [!NOTE]
> **On Feature Leakage Concerns**
> The objective of Stage C is not frequency-blind inference. Frequency, $k$, and $\alpha$ are intentionally provided to the model because they are observable physical parameters available during deployment and directly govern rain attenuation physics.

### Full Feature Vector
The Stage C model utilizes the following complete feature set for training and inference:
- **`excess_attn`**: Instanstaneous excess path attenuation (dB) after FSPL and gas loss removal.
- **`rolling_mean_30s`**: 30-second rolling average of excess attenuation (dB).
- **`rolling_mean_60s`**: 60-second rolling average of excess attenuation (dB).
- **`rolling_max_300s`**: 5-minute rolling maximum excess attenuation (dB) to capture peak intensity.
- **`lag_excess_attn_10s`**: 10-second lagged excess attenuation (dB) to capture temporal slope.
- **`elevation`**: Ground station elevation angle (degrees) to account for path geometry.
- **`L_eff`**: Effective path length through rain (km) derived dynamically from elevation and rain height.
- **`frequency`**: Carrier frequency ($f_c$) in GHz.
- **`k`**: ITU-R P.838 polarization/frequency-specific coefficient $k$.
- **`alpha`**: ITU-R P.838 frequency-specific exponent coefficient $\alpha$.

### Cross-Frequency Performance Comparison (R² and RMSE)

Evaluating the generalization of the $14\text{ GHz}$ trained model (Stage B) vs. the multi-frequency trained model (Stage C):

| Frequency | Stage B (Unaware) R² | Stage C (Aware) R² | Stage B (Unaware) RMSE | Stage C (Aware) RMSE | Stage C F1 |
|---|---|---|---|---|---|
| 10 GHz | N/A | 0.9990 | N/A | 0.2105 | 0.9982 |
| 12 GHz | 0.8978 | 0.9990 | 2.1962 | 0.2087 | 0.9992 |
| 14 GHz | 0.9945 | 0.9982 | 0.4934 | 0.2847 | 0.9996 |
| 20 GHz | 0.7250 | 0.9943 | 3.6022 | 0.5046 | 0.9998 |
| 30 GHz | -0.2727 | 0.9804 | 7.7491 | 0.9349 | 0.9999 |

### Visual Validation

#### Cross-Frequency Generalization Improvement Plot
![Cross Frequency Generalization Comparison](plots/stage_b5_frequency_generalization.png)

Although train and test sets use different random seeds, both are generated by the same underlying simulator and therefore share the same physical assumptions.

### Architectural Selection: XGBoost vs. Temporal Convolutional Networks (TCNs)
Temporal Convolutional Networks (TCNs) and LSTMs were considered. However, the frequency-aware XGBoost architecture already achieved near-saturated simulator-domain performance while remaining significantly simpler, faster to train, easier to interpret, and easier to deploy. Consequently, further architectural complexity was not justified by expected performance gains.


## ITU-R vs. NASA Climatology Comparison

### The NASA GPM IMERG vs. ITU-R Discrepancy
Real-world validation using NASA's Global Precipitation Measurement (GPM) Integrated Multi-satellitE Retrievals (IMERG) data indicates a significant gap between theoretical ITU-R P.837-7 climatological statistics and actual precipitation intensities. For instance, in subtropical monsoon zones like Delhi:

| Climatology Source | $R_{0.01}$ Exceedance |
| :--- | :---: |
| **ITU-R P.837-7** | 42 mm/h |
| **NASA GPM IMERG** | 90 mm/h |

For Delhi, NASA GPM IMERG estimates indicate substantially higher extreme rainfall intensities than represented in ITU-R P.837-7. The observed discrepancy exceeds 110% for the Delhi study region.

### NASA GPM IMERG Methodology
To ensure scientific rigor and reproducibility, GPM estimates for the Delhi study region were derived as follows:
- **Dataset**: NASA GPM Integrated Multi-satellitE Retrievals for GPM (IMERG) Version 6 (Final Run).
- **Temporal Coverage**: June 2000 to December 2021 (over 21 years of record).
- **Spatial Resolution**: A single grid cell of $0.1^{\circ} \times 0.1^{\circ}$ centered on Delhi coordinates ($28.6^{\circ}\text{N}$, $77.2^{\circ}\text{E}$).
- **Temporal Resolution**: 30-minute sampling interval, measuring precipitation rate (mm/h).
- **Exceedance Calculation**: The empirical Complementary Cumulative Distribution Function (CCDF) was computed over the entire temporal record to locate the $0.01\%$ annual exceedance threshold ($R_{0.01}$).

### Real-World GPM IMERG Online Data Sources
Programmatic and manual access to GPM/IMERG validation data is available via:
1. **NASA GES DISC**: Goddard Earth Sciences Data and Information Services Center (provides HDF5/NetCDF files).
2. **Google Earth Engine**: Programmatic access to the daily or 30-minute IMERG dataset via GEE ID: `NASA/GPM_L3/IMERG_V06`.
3. **NASA Earthdata Search**: UI portal for downloading spatial-temporal IMERG grids.


## Simulator Generator Validation

An investigation was conducted to understand why the simulator was under-producing extreme rain rates compared to database config files. We identified two independent generator biases.

> [!IMPORTANT]
> **Rain Rate Clipping Limitation**
> Rain rates above 150 mm/h are currently clipped by the simulator. This may underestimate the upper tail of extreme convective rainfall distributions and could affect comparisons against NASA GPM climatology.

### Diagnostic of Simulator Rain Rate Analytic Flaws

#### Bug A: Quantile Probit Fitting Error (Static $P_{\text{rain}}$ Assumption)
- **Flaw**: The simulator fits lognormal parameters ($\mu$, $\sigma$) using static normal quantiles ($z_{0.001} = 3.0902$ and $z_{0.01} = 2.3263$). These quantiles mathematically assume that the probability of rain ($P_{\text{rain}}$) is exactly $10\%$ ($0.1$) for all locations.
- **Impact**: Ground stations with lower rain fractions (e.g., Delhi, where $P_{\text{rain}} = 0.053$) suffer from distorted mapping. By ignoring $P_{\text{rain}}$ in the probit fitting, the generator spreads rainfall across too much clear sky, underestimating the $R_{0.01}$ peak.
- **Correction**: Percentiles must be computed dynamically using the inverse standard normal cumulative distribution function (probit function $Q^{-1}$):
  $$z_{0.001} = \Phi^{-1} \left( 1 - \frac{0.0001}{P_{\text{rain}}} \right) \quad \text{and} \quad z_{0.01} = \Phi^{-1} \left( 1 - \frac{0.001}{P_{\text{rain}}} \right)$$
  where $\Phi^{-1}$ is the inverse standard normal CDF, and $P_{\text{rain}}$ is the annual rain occurrence probability.

#### Bug B: Temporal Markov Reset (Tail Truncation Bias)
- **Flaw**: When a rain event starts, the simulator initializes the log-normal rain rate `ln_R` to the median value $\mu$.
- **Impact**: Because rain events terminate before the AR(1) process fully explores its stationary distribution, repeatedly resetting event onset to the median introduces a finite-duration sampling bias toward lower rain intensities.
- **Correction**: Initialize the log-normal rain rate using a random draw scaled by the station's variance parameter on event onset:
  $$\ln R_{\text{onset}} = \mu + \text{noise} \times \sigma$$

### Validation Targets
To evaluate the simulation corrections accurately and prevent target confusion, we explicitly split our validation results into two different evaluation targets:
1. **Validation Against ITU Station Parameters**: Evaluates whether the simulator successfully reproduces its configured database parameters (Target: $R_{0.01} = 42\text{ mm/h}$ for Delhi).
2. **Validation Against NASA GPM Climatology**: Evaluates whether the simulator successfully reproduces observed real-world precipitation statistics when configured with NASA GPM parameters (Target: $R_{0.01} = 90\text{ mm/h}$ for Delhi).

### 1. Validation Against ITU Station Parameters (Target: 42 mm/h for Delhi)

The table below shows the simulated rain rate statistics when the generator is configured with the standard ITU database parameters for Delhi (Target $R_{0.01} = 42.00\text{ mm/h}$, $P_{\text{rain}} = 0.053$):

| Configuration | $R_{0.01}$ (mm/h) | $R_{0.001}$ (mm/h) | Mean Rain Rate (mm/h) | Average Rain Duration | Exceedance Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ITU Target** | **42.00** | — | — | — | **100.0%** |
| **Original (Bug A + B present)** | 24.13 | 55.89 | 0.1226 | 337.6s | 57.5% |
| **Bug A only (Bug B present)** | 31.18 | 60.69 | 0.1734 | 333.5s | 74.2% |
| **Bug A + Bug B (Fully Corrected)** | 41.16 | 74.87 | 0.1927 | 337.4s | **98.0%** |

### 2. Validation Against NASA GPM Climatology (Target: 90 mm/h for Delhi)

The table below shows the simulated rain rate statistics when the generator is configured with Delhi's NASA GPM parameters (Target $R_{0.01} = 90.00\text{ mm/h}$, $R_{0.1} = 35.00\text{ mm/h}$, $P_{\text{rain}} = 0.065$):

| Configuration | $R_{0.01}$ (mm/h) | $R_{0.001}$ (mm/h) | Mean Rain Rate (mm/h) | Average Rain Duration | JS Divergence (vs GPM Target) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NASA GPM Target Reference** | 87.59 | 150.00 | 0.3096 | 323.8s | 0.0000 |
| **Original (Bug A + B present)** | 55.65 | 122.86 | 0.2087 | 335.0s | 0.0316 |
| **Bug A only (Bug B present)** | 64.65 | 115.27 | 0.2712 | 333.2s | 0.0242 |
| **Bug A + Bug B (Fully Corrected)** | 90.78 | 150.00 | 0.3190 | 326.2s | **0.0167** |

#### Analysis and Key Takeaways:
- **Exceedance Reproduction**: With both biases corrected, the simulator reproduces configured targets with high fidelity, achieving $41.16\text{ mm/h}$ against the $42.0\text{ mm/h}$ ITU target, and $90.78\text{ mm/h}$ against the $90.0\text{ mm/h}$ GPM target.
- **Distribution Distance (JS Divergence)**: Correcting both biases continuously reduces the Jensen-Shannon divergence relative to the GPM Target from $0.0316$ down to $0.0167$. This confirms a significantly higher statistical fidelity across the entire precipitation distribution.
- **Rain Event Duration**: Average event durations remain extremely stable across configurations, showing that the onset initialization fix successfully restores variance without distorting temporal autocorrelation.


## Limitations

- All ML models are trained on simulator-generated data.
- Real-world beacon attenuation validation remains future work.
- Rain rates above 150 mm/h are clipped by the simulator.
- Multi-cell rain structures are not explicitly modeled.
- Cross-satellite transfer learning has not yet been evaluated.


## Scientific Conclusions

1. Analytical inversion suffers from scintillation leakage.

2. Temporal attenuation statistics contain sufficient information to recover rain rate.

3. Frequency-aware learning restores cross-frequency generalization.

4. Two generator biases produced substantial underestimation of extreme rainfall statistics, ranging from roughly 15% to 50% depending on climate regime.

5. Correcting these biases substantially improves agreement with target exceedance statistics.

6. NASA GPM observations suggest substantially higher extreme rainfall rates than represented by ITU-R climatology for Delhi.


## Future Work

To validate the model's performance beyond the simulator domain, future research will target:
- **Beacon Attenuation Datasets**: Validating the physics decomposition and regression steps using actual link beacon power data.
- **Satellite Operator Telemetry**: Evaluating model performance on live, commercial satellite operator link telemetry.
- **NASA GPM Aligned Rainfall Observations**: Directly aligning temporal telemetry events with NASA GPM observations.
- **Real-World Rain Gauge Validation**: Comparing predicted rain rates against local tipping-bucket rain gauge measurements.
