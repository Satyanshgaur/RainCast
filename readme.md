# Satellite Link Quality Simulator

A physics-first satellite link budget simulator for GEO Ku-band uplinks, combining ITU-R propagation models, a temporally correlated rain process, and an XGBoost link quality scorer inside a Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Physics — Detailed Explanation](#physics--detailed-explanation)
   - [1. GEO Orbital Geometry](#1-geo-orbital-geometry)
   - [2. Free-Space Path Loss](#2-free-space-path-loss)
   - [3. Thermal Noise Power](#3-thermal-noise-power)
   - [4. Rain Statistics — ITU-R P.837-7](#4-rain-statistics--itu-r-p837-7)
   - [5. Rain Attenuation Coefficients — ITU-R P.838-3](#5-rain-attenuation-coefficients--itu-r-p838-3)
   - [6. Rain Height — ITU-R P.839-4](#6-rain-height--itu-r-p839-4)
   - [7. Effective Slant Path — ITU-R P.618-13](#7-effective-slant-path--itu-r-p618-13)
   - [8. Rain Attenuation on the Path](#8-rain-attenuation-on-the-path)
   - [9. Gaseous Absorption — ITU-R P.676-12](#9-gaseous-absorption--itu-r-p676-12)
   - [10. Tropospheric Scintillation — ITU-R P.618-13 §2.4](#10-tropospheric-scintillation--itu-r-p618-13-24)
   - [11. Temporally Correlated Rain — ITU-R P.1853 / Maseng-Bakken](#11-temporally-correlated-rain--itu-r-p1853--maseng-bakken)
   - [12. Doppler Shift](#12-doppler-shift)
   - [13. The Link Budget Equation](#13-the-link-budget-equation)
   - [14. Packet Loss Model](#14-packet-loss-model)
6. [The Simulation Loop](#the-simulation-loop)
7. [Machine Learning Pipeline](#machine-learning-pipeline)
8. [File Reference](#file-reference)
9. [Ground Station Parameters](#ground-station-parameters)
10. [UI Controls and What They Change](#ui-controls-and-what-they-change)
11. [Known Limitations](#known-limitations)
12. [References](#references)

---

## Project Overview

Satellite communications are degraded by a chain of physical phenomena: geometric path loss, thermal noise in the receiver, rain cells absorbing and scattering microwave energy, water vapour and oxygen absorbing the signal, and small-scale turbulent refraction causing signal amplitude to flicker. Each of these is understood well enough to be modelled from first principles using internationally standardised methods published by the ITU Radiocommunication Sector (ITU-R).

This project implements those models faithfully and uses them to:

- Compute a per-minute time series of signal-to-noise ratio (SNR) for each ground station over a configurable simulation window
- Model rain as a temporally correlated stochastic process rather than independent per-step sampling, so rain events persist realistically across multiple minutes
- Derive link quality statistics (mean SNR, 10th-percentile SNR, outage fraction, packet loss) from that time series
- Feed those statistics into a trained XGBoost model to produce a single link quality score per station
- Present everything in a Streamlit dashboard where physical parameters (frequency, power, bandwidth, rain intensity) can be varied interactively

The target use case is pre-deployment link budget analysis and what-if scenario testing for GEO Ku-band satellite uplinks.

---

## Repository Structure

```
.
├── app.py                    # Streamlit dashboard — UI, scoring, display
├── satellite_link_sim.py     # ITU-R physics engine — core simulation
├── ground_stations.py        # Single source of truth: all station parameters
├── ground_station.py         # (legacy) earlier version of station definitions
├── groundstations.py         # (legacy) earlier version
├── groundstation.py          # (legacy) earlier version
├── geometry.py               # Geometry helpers used by legacy app.py
├── physicsengine.py          # Standalone physics utilities
├── propogate.py              # Orbital propagation helpers
├── link_quality.py           # Link quality scoring utilities
├── database.py               # SQLite interface layer
├── satellite.db              # SQLite database (link records)
├── satellites.db             # SQLite database (satellite catalogue)
├── train_xgboost.py          # Model training script
├── link_training_data.csv    # Training dataset for XGBoost model
├── feature_scaler.pkl        # Fitted sklearn StandardScaler
├── xgb_link_model.pkl        # Trained XGBoost regressor
├── LICENSE
└── README.md
```

The canonical files for the current version of the project are `app.py`, `satellite_link_sim.py`, and `ground_stations.py`. The `legacy` files (`groundstation.py`, `groundstations.py`, `ground_station.py`, `geometry.py`) predate the physics upgrade and are retained for reference.

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd <repo-dir>

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install streamlit pandas numpy scikit-learn xgboost joblib
```

---

## Quick Start

**Run the Streamlit dashboard:**

```bash
streamlit run app.py
```

**Run the CLI simulator directly:**

```bash
# Clear weather, probabilistic rain onset
python3 satellite_link_sim.py

# Force all stations into rain for the entire window
python3 satellite_link_sim.py --rain
```

**Retrain the XGBoost model:**

```bash
python3 train_xgboost.py
```

---

## Physics — Detailed Explanation

The simulator computes a full link budget at each time step. A link budget is an accounting of every gain and every loss that a signal experiences from transmitter power amplifier to receiver decoder. The SNR at the receiver is:

```
SNR = EIRP − FSPL − L_gas − L_rain − L_scint + G_rx − N
```

where every term is in decibels (dB) or dBW. The sections below derive each term from first principles.

---

### 1. GEO Orbital Geometry

A geostationary satellite sits in a circular orbit 35,786 km above the equator at a fixed longitude. The distance from a ground station to the satellite, and the angle above the horizon at which the station must point its antenna, are determined purely by the station's latitude and longitude and the satellite's orbital slot longitude.

**Central angle γ** is the angular separation at Earth's centre between the ground station and the sub-satellite point (the point on the equator directly below the satellite):

```
cos γ = cos(φ) · cos(Δλ)
```

where φ is the ground station latitude and Δλ is the difference in longitude between the station and the satellite.

**Elevation angle** is the angle above the local horizontal at which the ground station sees the satellite. A low elevation means the signal travels through more atmosphere and more rain, which is why polar stations suffer more than equatorial ones:

```
sin(El) = (cos γ − R_E / R_GEO) / sqrt(1 − cos²γ)
```

where R_E = 6,371 km (Earth mean radius) and R_GEO = 42,164 km (GEO orbital radius from Earth's centre, not from the surface).

**Slant range** is the straight-line distance from ground station to satellite, derived from the law of cosines on the Earth-centre / ground-station / satellite triangle:

```
d = sqrt(R_GEO² + R_E² − 2 · R_GEO · R_E · cos γ)
```

For a station at the sub-satellite point (directly below), d = R_GEO − R_E = 35,786 km. For a station near the edge of coverage where the elevation angle falls to around 5°, d can reach 41,000–42,000 km. This 15% increase in distance corresponds to an extra 1.3 dB of free-space path loss.

Each ground station in this project uses its own GEO satellite arc:
- Delhi → INSAT-4A/SES-8 at 83.0°E, elevation ≈ 48°
- Tokyo → JCSAT-3A at 110.0°E, elevation ≈ 52°
- Berlin → Astra 1 at 19.2°E, elevation ≈ 35°
- Sao Paulo → Star One C2 at 70.0°W, elevation ≈ 65°

---

### 2. Free-Space Path Loss

Free-space path loss (FSPL) is the attenuation a radio wave experiences purely from spreading out over distance, with no absorption. It is not a loss in the material sense — no energy is absorbed — it is a consequence of the inverse-square law. A transmit antenna radiates power into an expanding sphere. At distance d, that power is spread over a sphere of area 4πd². The receive antenna with effective aperture A_eff intercepts only A_eff / 4πd² of the transmitted power.

The ITU-R formulation in dB is:

```
FSPL(dB) = 92.45 + 20·log₁₀(f_GHz) + 20·log₁₀(d_km)
```

The 20·log₁₀(f) term arises because a fixed-gain antenna has an effective aperture that decreases as frequency increases (aperture ~ λ² ~ 1/f²). The 20·log₁₀(d) term is the inverse-square spreading. The constant 92.45 is a unit conversion factor.

At 14 GHz and 40,000 km, FSPL ≈ 207 dB. This is an enormous number — the signal loses 20 orders of magnitude in power — which is why satellite ground stations use large dishes and high-power amplifiers.

The **20 log₁₀(f)** relationship means that moving from Ku-band (14 GHz) to Ka-band (20 GHz) adds 3.1 dB of FSPL alone, before accounting for the much worse rain attenuation at Ka-band. This is why the frequency slider in the UI produces a visible deterioration as you push toward 20–30 GHz.

---

### 3. Thermal Noise Power

Every receiver generates thermal noise due to the random motion of electrons in its components. The noise power in watts in a receiver with system noise temperature T_sys and bandwidth B is:

```
N = k_B · T_sys · B
```

where k_B = 1.381 × 10⁻²³ J/K is Boltzmann's constant. In dBW:

```
N(dBW) = 10·log₁₀(k_B) + 10·log₁₀(T_sys) + 10·log₁₀(B)
       = −228.6 + 10·log₁₀(T_sys) + 10·log₁₀(B)
```

The system noise temperature T_sys captures contributions from the antenna (looking at a warm Earth, or a cold sky), the low-noise amplifier (LNA), feedline losses, and subsequent receiver stages. It is typically 290–600 K for Ku-band uplink terminals.

This equation explains why the bandwidth slider matters. Doubling the bandwidth from 36 MHz to 72 MHz adds 3 dB to the noise floor, cutting 3 dB from the SNR. This is a direct, unavoidable physics trade-off — more bandwidth means more data throughput capacity, but it also admits more noise.

---

### 4. Rain Statistics — ITU-R P.837-7

The probability that rain of a given intensity is occurring at a location on Earth is highly non-uniform. Tropical regions near the equator, like Sao Paulo, experience intense convective rain cells — short-duration storms with rain rates sometimes exceeding 100 mm/h. High-latitude temperate regions like Berlin experience mostly stratiform rain — widespread, lower-intensity rainfall — that actually occurs for more hours per year but causes less severe attenuation per event. This difference is fundamental to satellite link design.

ITU-R P.837-7 provides digital map grids (latitude/longitude) of rain rate exceeded for given percentages of an average year. The key exceedance probabilities used in this project are:

| Probability | Meaning | Symbol |
|---|---|---|
| 0.01% of year | Exceeded for ~53 minutes/year | R₀.₀₁ |
| 0.1% of year | Exceeded for ~8.8 hours/year | R₀.₁ |
| 1% of year | Exceeded for ~88 hours/year | R₁ |

For each ground station, three values from the P.837-7 maps are looked up and hardcoded:

| Station | R₀.₀₁ (mm/h) | R₀.₁ (mm/h) | R₁ (mm/h) | P_rain |
|---|---|---|---|---|
| Delhi | 42 | 19 | 6 | 5.3% |
| Tokyo | 80 | 42 | 16 | 7.2% |
| Berlin | 28 | 14 | 5.5 | 6.5% |
| Sao Paulo | 95 | 55 | 22 | 9.5% |

The `P_rain` column is the fraction of a year during which rain of any intensity is occurring (above roughly 0.5 mm/h). Sao Paulo has the highest values on all dimensions — both the most intense rain (R₀.₀₁ = 95 mm/h vs Berlin's 28) and the most rain hours per year (9.5% vs Berlin's 6.5%).

These three quantiles are used to fit a **lognormal distribution** to the conditional rain rate distribution (the distribution of rain rate, given that it is raining). A lognormal is appropriate because rain rate is strictly positive and has a heavy right tail — most rain events are moderate, but a small fraction are extremely intense. Two standard normal quantiles z₀.₀₁ = 3.0902 and z₀.₁ = 2.3263 allow solving analytically for the lognormal parameters:

```
σ_ln = (ln(R₀.₀₁) − ln(R₀.₁)) / (z₀.₀₁ − z₀.₁)

μ_ln = ln(R₀.₁) − z₀.₁ · σ_ln
```

This fitted (μ_ln, σ_ln) pair parameterises the AR(1) correlated rain process described in Section 11.

---

### 5. Rain Attenuation Coefficients — ITU-R P.838-3

When a microwave signal passes through rain, two physical mechanisms cause attenuation:

**Absorption**: raindrops contain liquid water, which is a polar molecule. Oscillating electric fields in the microwave signal excite rotational and vibrational modes in the water molecules. This energy is converted to heat, removing it from the signal.

**Scattering**: raindrops are comparable in size to microwave wavelengths (a 1 mm raindrop is 1/21 of the 21 mm wavelength of 14 GHz). Mie scattering causes the incoming wave to be redirected out of the beam path.

Both effects are captured by the **specific attenuation** γ_R (dB/km), which describes how many decibels of signal are lost per kilometre of path through rain of intensity R (mm/h):

```
γ_R = k · R^α    [dB/km]
```

The coefficients k and α depend on frequency, polarization, and rain drop size distribution. ITU-R P.838-3 provides tabulated values of k_H, α_H (horizontal polarization) and k_V, α_V (vertical polarization) at frequencies from 1 to 1000 GHz, derived from Mie scattering theory applied to measured raindrop size distributions.

At 14 GHz vertical polarization: k ≈ 0.0307, α ≈ 1.190.

The relationship is a power law in R with exponent α slightly above 1.0. This means attenuation grows faster than linearly with rain rate: doubling the rain rate from 20 to 40 mm/h roughly triples the specific attenuation. The exponent α also decreases with frequency above ~30 GHz, meaning very high rain rates at Ka-band saturate attenuation more quickly than at Ku-band.

**Polarization matters**: horizontal polarization consistently has higher k values than vertical at the same frequency (for 14 GHz, k_H ≈ 0.0368 vs k_V ≈ 0.0307). This is because oblate raindrops — which are flattened by air resistance as they fall — have their major axis horizontal, so horizontally polarized waves interact with more water. The polarization selector in the UI switches between these coefficient sets.

The coefficients are implemented via log-linear interpolation (log for k, linear for α) between the tabulated frequency points. k is interpolated on a log-log scale because it varies over many orders of magnitude across the frequency range.

---

### 6. Rain Height — ITU-R P.839-4

Rain does not extend to arbitrary altitude. Above a certain height — the 0°C isotherm, the freezing level — precipitation falls as ice or snow, which interacts with microwaves very differently from liquid water (ice has much lower specific attenuation). The **rain height** h_R is the altitude above mean sea level (MSL) up to which liquid rain is found.

ITU-R P.839-4 gives h_R as a function of latitude, based on the height of the 0°C isotherm:

```
For |φ| ≤ 23°:       h_R = 5.0 km
For 23° < |φ| ≤ 36°: h_R = 5.0 − 0.075·(|φ| − 23)
For |φ| > 36°:       h_R = max(5.0 − 0.1·(|φ| − 36), 2.0)
```

This gives physically correct behaviour: near the equator the atmosphere is warmer and the freezing level is higher (~5 km), while at high latitudes the freezing level descends to ~2 km. The practical consequence is that a signal arriving at low elevation from a high-latitude station like Berlin cuts through a shorter vertical extent of rain than a signal arriving at low elevation near the equator.

---

### 7. Effective Slant Path — ITU-R P.618-13

The rain column that the signal passes through is not vertical — the satellite is at an angle above the horizon. The vertical distance from the ground station altitude h_s to the rain height h_R is (h_R − h_s). The slant distance through this rain column at elevation angle El is:

```
L_s = (h_R − h_s) / sin(El)
```

This is the fundamental reason why low-elevation stations suffer more from rain: a station at 35° elevation has L_s = (h_R − h_s)/sin(35°) = 1.74·(h_R − h_s), while a station at 65° elevation has L_s = 1.10·(h_R − h_s). The same rain column translates to 58% more path length at 35° than at 65°.

For very low elevation angles (below 10°), a path reduction factor r accounts for the horizontal inhomogeneity of rain — rain cells are typically a few kilometres wide horizontally, so the signal at grazing angles exits the rain cell horizontally before traversing the full vertical extent:

```
r = 1 / (1 + 0.78·sqrt(L_g · k) − 0.38·(1 − exp(−2·L_g)))
```

where L_g = (h_R − h_s) / tan(El) is the horizontal projection of the path, and k is the ITU-R P.838 coefficient at the current frequency and polarization. For elevation angles above 10° the reduction factor is effectively 1.0 and is omitted.

---

### 8. Rain Attenuation on the Path

Combining the specific attenuation (Section 5) with the effective path length (Section 7) gives the total rain attenuation on the signal path:

```
A_rain = k · R^α · L_eff    [dB]
```

This is ITU-R P.838-3 applied to the P.618-13 geometry. The per-step rain rate R comes from the AR(1) correlated stochastic process described in Section 11.

To give a sense of scale: at 14 GHz vertical polarization with a rain rate of 19 mm/h (Delhi's R₀.₁, the rate exceeded 8.8 hours/year) and an effective path of 7 km, the attenuation is approximately:

```
A = 0.0307 × 19^1.190 × 7 ≈ 0.0307 × 28.1 × 7 ≈ 6.0 dB
```

This 6 dB loss cuts the received signal power by 75%. At the R₀.₀₁ rate of 42 mm/h:

```
A = 0.0307 × 42^1.190 × 7 ≈ 0.0307 × 68.4 × 7 ≈ 14.7 dB
```

A 14.7 dB loss at the most extreme rain conditions expected for 53 minutes per year is a link-threatening event for a system with only 10–15 dB margin. This is the core design challenge for Ku-band satellite links in tropical regions.

---

### 9. Gaseous Absorption — ITU-R P.676-12

The atmosphere absorbs microwave energy through molecular resonances in oxygen (O₂) and water vapour (H₂O). The dominant features relevant to this project are:

**Oxygen**: a broad absorption band centred around 60 GHz (multiple overlapping O₂ rotation lines) and a weaker line at 118.75 GHz. At 14 GHz, oxygen contributes approximately 0.008 dB/km of zenith attenuation. Though individually small, it integrates over the full atmosphere column (~10 km effective for zenith).

**Water vapour**: a strong resonance line at 22.235 GHz (water rotation transition) and lines at 183.31 GHz and 325.153 GHz. At 14 GHz the water vapour contribution is off-resonance but not negligible, and it scales with the surface water vapour density (g/m³), which varies significantly between stations:

| Station | Water vapour density | Humidity |
|---|---|---|
| Delhi | 12.0 g/m³ | 70% |
| Tokyo | 9.5 g/m³ | 65% |
| Berlin | 7.0 g/m³ | 75% |
| Sao Paulo | 14.5 g/m³ | 80% |

The total zenith attenuation is:

```
A_zen = (γ_O₂ + γ_H₂O) · H_eff
```

where H_eff ≈ 10 km is the effective scale height of the atmosphere for this simplified model. The slant-path attenuation scales by 1/sin(El):

```
A_gas = A_zen / sin(El)
```

Gaseous absorption is typically 0.5–1.5 dB for Ku-band GEO links and is constant for a given station (no time variation in this model). It is relatively small compared to rain attenuation but not negligible, and it increases toward the 22 GHz water vapour line, which is one reason Ka-band (17.7–21.2 GHz downlink, 27.5–31 GHz uplink) is more challenging than Ku-band.

---

### 10. Tropospheric Scintillation — ITU-R P.618-13 §2.4

Tropospheric scintillation is rapid, small-amplitude fluctuation of the received signal caused by turbulent irregularities in the refractive index of the troposphere (the lowest ~10 km of atmosphere). Regions of air with slightly different temperature and humidity have slightly different refractive indices. As the wavefront propagates through these irregularities it is refracted slightly differently at different points across its aperture, producing constructive and destructive interference at the receiver. The result is amplitude flickering at time scales of seconds to tens of seconds with typical peak-to-peak variation of 0.5–2 dB.

The standard deviation of scintillation fade σ_s depends on:

**Wet refractivity N_wet**: proportional to the surface humidity. Higher humidity means larger refractive index variations in the turbulent atmosphere.

**Antenna aperture**: a larger antenna averages over more of the wavefront, reducing the apparent scintillation. The aperture averaging factor g(x) in ITU-R P.618-13 quantifies this, where x = 1.22 D²_eff (f/300) with D_eff = √η · D (η = aperture efficiency, D = antenna diameter in metres).

**Elevation angle**: lower elevation means the path traverses more of the turbulent troposphere. The dependence is ∝ 1/sin(El)^1.2.

In this simulator, scintillation is modelled as Gaussian noise added to the SNR at each time step:

```
L_scint[t] ~ N(0, σ_s²)
```

This is a simplification — real scintillation is correlated in time with a bandwidth of a few Hz — but it correctly captures the magnitude and Gaussian statistics of the effect for a 1-minute time step where the correlation is already weak.

---

### 11. Temporally Correlated Rain — ITU-R P.1853 / Maseng-Bakken

This is the most physically important modelling choice in the simulator. Rain is not a memoryless process. When it starts raining, it typically continues raining for tens of minutes to hours. A real rain fade event on a satellite link lasts 5–30 minutes, during which the link may be impaired or lost completely. If each time step sampled rain rate independently from the lognormal distribution, you would get rapid uncorrelated flickering that looks nothing like real rain. The temporal structure matters enormously for availability calculations.

The simulator implements the **Maseng-Bakken first-order autoregressive model**, standardised in ITU-R P.1853. It operates in two layers:

**Layer 1 — Rain occurrence (two-state Markov chain)**

The process is either in a CLEAR state or a RAINING state. At each time step:
- From CLEAR, transitions to RAINING with probability: `p_onset = 1 − exp(−Δt / τ_clear)`
- From RAINING, transitions to CLEAR with probability: `p_clear = 1 − exp(−Δt / τ_rain)`

where Δt = 60 s is the time step and τ_rain = τ_c = 300 s (5 minutes, the empirical rain cell coherence time). τ_clear is derived from the annual rain occurrence fraction P_rain:

```
τ_clear = τ_c · (1 − P_rain) / P_rain
```

For Delhi (P_rain = 5.3%), τ_clear ≈ 5,377 s ≈ 90 minutes, meaning the expected time between rain events is about 90 minutes. For Sao Paulo (P_rain = 9.5%), τ_clear ≈ 2,842 s ≈ 47 minutes — rain starts again much more frequently.

**Layer 2 — Rain rate (AR(1) lognormal process)**

While in the RAINING state, the log-rain-rate ln(R) evolves as a first-order autoregressive process:

```
ln(R[t]) = ρ · ln(R[t−1]) + sqrt(1 − ρ²) · σ_ln · N(0,1) + (1 − ρ) · μ_ln
```

The autocorrelation coefficient ρ = exp(−Δt/τ_c) = exp(−60/300) ≈ 0.819. This means adjacent 1-minute samples share about 82% correlation — the rain rate changes slowly and smoothly rather than jumping discontinuously. The innovation term sqrt(1 − ρ²) · σ_ln · N(0,1) injects new randomness each step while the (1 − ρ) · μ_ln term provides mean reversion, ensuring the process stays centred on the station's climatological mean log-rate.

The **stationary distribution** of ln(R) is exactly N(μ_ln, σ_ln²), which is the lognormal distribution fitted to the ITU-R P.837-7 quantiles in Section 4. This means the long-run statistics of the simulated rain rates match the empirical ITU statistics exactly, while the temporal evolution is physically realistic.

This model correctly reproduces:
- Rain events that persist for realistic durations (5–30 minutes typically)
- Gradual ramp-up and ramp-down within a rain event (not instantaneous)
- Station-specific severity (Sao Paulo's rates are much higher than Berlin's)
- The correct fraction of time spent raining (controlled by P_rain)

---

### 12. Doppler Shift

A satellite in a geostationary orbit is nominally stationary relative to the ground. In practice, orbital perturbations (solar radiation pressure, lunar/solar gravity, slight orbital eccentricity) cause the satellite to trace a small figure-8 path around its nominal position, giving it a non-zero radial velocity relative to each ground station.

The classical Doppler shift in Hz is:

```
f_D = (v_r / c) · f_c
```

where v_r is the radial velocity of the satellite (negative = receding), c = 2.998 × 10⁸ m/s, and f_c is the carrier frequency. At 14 GHz with a typical radial velocity of −30 m/s:

```
f_D = (−30 / 2.998×10⁸) × 14×10⁹ ≈ −1,401 Hz
```

This ~1.4 kHz Doppler offset is constant over the simulation window and is informational only — it is not incorporated into the SNR calculation but is reported in the output for completeness. Modern satellite modems track and compensate for Doppler to well within 1 Hz.

---

### 13. The Link Budget Equation

Every physical effect computed above combines into the link budget equation evaluated at each time step t:

```
SNR[t] = EIRP_eff
         − FSPL
         − A_gas
         − A_rain[t]
         − L_scint[t]
         + G_rx
         − N(T_sys, B)
```

All terms in dBW or dB:

| Term | Symbol | Source | Typical value |
|---|---|---|---|
| Effective transmit power | EIRP_eff = EIRP + offset | Hardware + UI slider | 50–54 dBW |
| Free-space path loss | FSPL | Geometry + frequency | ~207 dB at 14 GHz |
| Gaseous absorption | A_gas | ITU-R P.676 | 0.5–1.5 dB |
| Rain attenuation | A_rain[t] | ITU-R P.838 × P.618 path | 0–30+ dB |
| Scintillation | L_scint[t] | ITU-R P.618 §2.4 | ±0.5–2 dB |
| Receive antenna gain | G_rx | Hardware | 44–48 dBi |
| Thermal noise | N | k_B · T_sys · B | −130 to −125 dBW |

The time-varying terms are A_rain and L_scint. FSPL, A_gas, G_rx, and N are constant for a given station and set of UI parameters. EIRP changes with the UI offset slider.

---

### 14. Packet Loss Model

The link budget produces SNR in dB. To convert this to a meaningful quality metric for the ML model, a sigmoid function maps SNR to packet loss probability:

```
P_loss = 1 / (1 + exp(0.8 · (SNR − θ)))
```

where θ = 10 dB is the inflection point — the SNR at which 50% of packets are lost. This threshold corresponds to the practical Eb/N₀ floor for DVB-S2 modulation with typical forward error correction coding at Ku-band. The steepness coefficient 0.8 gives the following mapping:

| SNR (dB) | Packet loss |
|---|---|
| 17 | 1.6% |
| 14 | 7.5% |
| 10 (threshold) | 50% |
| 7 | 85% |
| 3 | 98% |

This sigmoid is physically motivated: real satellite link performance curves (BER vs Eb/N₀ for DVB-S2) are indeed very steep near the threshold — coding gain concentrates the transition from near-zero BER to near-100% BER in a narrow SNR range of a few dB.

---

## The Simulation Loop

For each ground station, `simulate_station()` executes the following sequence once per time step (default: 60 steps of 60 seconds each = 1 hour):

```
1.  Advance the Markov rain state (CLEAR / RAINING)
2.  If RAINING: advance the AR(1) lognormal process → rain_rate[t]
3.  Compute rain attenuation: A_rain[t] = k · rain_rate[t]^α · L_eff
4.  Sample scintillation: L_scint[t] ~ N(0, σ_s²)
5.  Compute SNR[t] = EIRP − FSPL − A_gas − A_rain[t] − L_scint[t] + G_rx − N
6.  Compute packet loss: P_loss[t] = sigmoid(SNR[t] − θ)
7.  Append all values to time series
```

After all steps, summary statistics are computed:

- `snr_mean`: arithmetic mean of SNR series — the average link condition
- `snr_min`: worst single step — the deepest fade
- `snr_std`: standard deviation — how much the link fluctuates
- `snr_p10`: 10th percentile — SNR exceeded 90% of the time (availability margin)
- `rain_fraction`: fraction of steps where rain was active
- `avg_rain_db`: mean rain attenuation over rainy steps only
- `avg_pkt_loss`: mean packet loss probability across all steps
- `outage_fraction`: fraction of steps where SNR fell below the 10 dB threshold

---

## Machine Learning Pipeline

The ML component scores each station's link quality from a single number rather than the full time series. The pipeline:

**Feature vector** (matches the schema the model was trained on):

| Column | Source | Meaning |
|---|---|---|
| `snr_db` | `r.snr_mean` | Average SNR over the simulation window |
| `packet_loss` | `r.avg_pkt_loss` | Mean packet loss probability |
| `load_factor` | UI slider | Fraction of link capacity in use |

**`feature_scaler.pkl`** — a scikit-learn `StandardScaler` fitted on the training data. It transforms each feature to zero mean and unit variance. This is critical because the raw features have very different scales (snr_db ~10–20, packet_loss ~0–1, load_factor ~0–1).

**`xgb_link_model.pkl`** — a trained XGBoost regressor that maps the scaled 3-feature vector to a link quality score. The model was trained on `link_training_data.csv` using `train_xgboost.py`.

**Important caveat**: the model was trained on a 3-feature schema. The richer statistics computed by the physics engine (`snr_p10`, `snr_std`, `outage_fraction`) are displayed in the UI but not passed to the model. Retraining on the full feature set would allow the model to differentiate between a stable link at 12 dB SNR and a highly variable link that averages 12 dB but dips to 4 dB during rain.

---

## File Reference

| File | Role |
|---|---|
| `app.py` | Streamlit UI. Reads sidebar controls → calls `simulate_station()` → scores with ML → displays charts and tables |
| `satellite_link_sim.py` | Core physics engine. Implements all ITU-R models. Public API: `simulate_station()`, `simulate_all()` |
| `ground_stations.py` | Single source of truth. All station RF, geometry, ITU rain, and climate parameters live here |
| `geometry.py` | Legacy geometry helpers (`slant_range()`, `effective_elevation()`) used by the older `app.py` version |
| `physicsengine.py` | Standalone physics utilities, predating the ITU-R upgrade |
| `propogate.py` | Orbital propagation helpers for non-GEO orbits |
| `link_quality.py` | Link quality scoring utilities used during training |
| `database.py` | SQLite read/write interface for `satellite.db` and `satellites.db` |
| `satellite.db` | SQLite database of recorded link observations |
| `satellites.db` | SQLite database of satellite catalogue entries |
| `train_xgboost.py` | Reads `link_training_data.csv`, fits `feature_scaler.pkl` and `xgb_link_model.pkl` |
| `link_training_data.csv` | Training dataset with columns `snr_db`, `packet_loss`, `load_factor`, and target label |
| `feature_scaler.pkl` | Fitted sklearn StandardScaler |
| `xgb_link_model.pkl` | Trained XGBoost regressor |

---

## Ground Station Parameters

All parameters for each ground station are defined in `ground_stations.py`. Every key used by both `satellite_link_sim.py` and `app.py` lives in that single file.

| Parameter | Type | Used by | Description |
|---|---|---|---|
| `name` | str | both | Display name |
| `eirp_dbw` | float | sim | Effective isotropic radiated power (dBW) = transmit power + antenna gain |
| `g_rx_dbi` | float | sim | Receive antenna gain (dBi) |
| `system_temp_k` | float | sim | Receiver system noise temperature (K) |
| `antenna_diam_m` | float | sim | Dish diameter for scintillation aperture averaging (m) |
| `atm_loss_db` | float | app (legacy) | Legacy atmospheric loss for simple link budget |
| `rain_severity` | str | app (legacy) | `low` / `medium` / `high` — legacy rain model label |
| `latitude` | float | sim | Ground station latitude (°, N positive) |
| `longitude` | float | sim | Ground station longitude (°, E positive) |
| `sat_lon_deg` | float | sim | GEO satellite longitude (°, W negative) |
| `altitude_km` | float | sim | Ground station altitude above MSL (km) — affects rain path length |
| `itu_rain.R001` | float | sim | Rain rate exceeded 0.01% of year (mm/h) — ITU-R P.837-7 |
| `itu_rain.R01` | float | sim | Rain rate exceeded 0.1% of year (mm/h) |
| `itu_rain.R1` | float | sim | Rain rate exceeded 1% of year (mm/h) |
| `itu_rain.P_rain` | float | sim | Fraction of year rain occurs — ITU-R P.837-7 |
| `wv_g_m3` | float | sim | Surface water vapour density (g/m³) — ITU-R P.676 |
| `humidity_pct` | float | sim | Relative humidity (%) — ITU-R P.618 scintillation |
| `v_radial_ms` | float | sim | Satellite radial velocity (m/s) — Doppler |

---

## UI Controls and What They Change

| Control | Range | Physical effect |
|---|---|---|
| Carrier frequency | 10–30 GHz | FSPL (+3.5 dB from 14→20 GHz); P.838 k/α (rain gets worse at higher freq); gaseous absorption (spikes near 22 GHz water vapour line) |
| Polarization | V / H | Switches P.838 k/α. Horizontal polarization gives ~20% higher rain attenuation due to oblate raindrop geometry |
| Bandwidth | 18–72 MHz | Raises thermal noise floor. Each doubling adds 3 dB of noise, reducing SNR by 3 dB |
| EIRP offset | −10 to +10 dB | Shifts all station SNRs up or down uniformly, simulating power amplifier headroom or degradation |
| Weather mode | Clear / Rain | Clear: probabilistic Markov onset. Rain: forces AR(1) into raining state for every step |
| Rain intensity scale | 0.5–3.0× | Multiplies ITU-R P.837-7 rain quantiles before lognormal fitting. 2.0× approximates a severe convective event |
| Simulation window | 10–180 min | Longer windows average over more rain events, stabilising mean statistics |
| Network load | 0.0–1.0 | Fed directly to ML model as `load_factor`. Does not change physics |

---

## Known Limitations

**ML model training data**: the XGBoost model was trained on `link_training_data.csv` whose provenance and labelling methodology are not documented in the repository. The link quality score it produces should be treated as a relative ranking signal rather than an absolute quality metric until the model is validated against real link performance data.

**3-feature ML schema**: the model only sees `(snr_mean, avg_pkt_loss, load)`. The physics engine computes richer statistics (`snr_p10`, `snr_std`, `outage_fraction`) that would improve discrimination between links with the same average SNR but different temporal variability. Retraining on the 5-feature schema is a clear next step.

**Single-frequency model**: the simulator uses one carrier frequency for all stations simultaneously. Real satellite systems use frequency plans, polarization reuse, and multiple transponders. The single-frequency model is adequate for relative station comparison but not for full system capacity planning.

**Static atmospheric model**: water vapour density and humidity are annual averages. In reality they vary with season and weather. A diurnal or seasonal model would improve accuracy for specific mission planning.

**GEO only**: `propogate.py` suggests LEO/MEO support was intended but the current physics engine and UI are GEO-only. The geometry functions assume a fixed slant range derived from the GEO arc.

**Independent scintillation**: scintillation is modelled as independent Gaussian noise per time step. Real scintillation has a correlation time of a few seconds to minutes. For 1-minute time steps this is a reasonable approximation but would be wrong for sub-minute analysis.

---

## References

The following ITU-R Recommendations are implemented in this project. All are freely available from the ITU website.

- **ITU-R P.618-13** (2017) — *Propagation data and prediction methods required for the design of Earth-space telecommunication systems*. Sections 2.2 (rain attenuation), 2.4 (scintillation), path reduction factor.

- **ITU-R P.676-12** (2019) — *Attenuation by atmospheric gases and related effects*. Simplified gaseous absorption model for O₂ and H₂O.

- **ITU-R P.837-7** (2017) — *Characteristics of precipitation for propagation modelling*. Digital maps of rain rate exceedance statistics. Per-station quantiles R₀.₀₁, R₀.₁, R₁ and P_rain are derived from these maps.

- **ITU-R P.838-3** (2005) — *Specific attenuation model for rain for use in prediction methods*. Tabulated k, α coefficients for horizontal and vertical polarization from 1 to 1000 GHz.

- **ITU-R P.839-4** (2013) — *Rain height model for prediction methods*. Latitude-dependent mean rain height formula.

- **ITU-R P.1853-2** (2019) — *Tropospheric attenuation time series synthesis*. Maseng-Bakken AR(1) model for temporally correlated rain.

- **ITU-R S.1066** — *Determination of the coordination area around an Earth station in the frequency bands between 100 MHz and 105 GHz*. GEO elevation angle and slant range geometry.

- **Maseng, T. and Bakken, P.M.** (1981) — *A stochastic dynamic model of rain attenuation*. IEEE Transactions on Communications, 29(5), 660–669. Original paper describing the AR(1) lognormal rain model.

- **ETSI EN 302 307** — DVB-S2 standard; defines the practical Eb/N₀ thresholds used to set the packet loss sigmoid inflection point.


## 📄 Licence

MIT Licence. See LICENSE for details.
