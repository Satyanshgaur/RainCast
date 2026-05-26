# Satellite Link Quality Simulator

A physics-first satellite link budget simulator that dynamically integrates **SGP4 orbital propagation** with ITU-R atmospheric models. It combines live geometry tracking, temporally correlated rain processes, and an XGBoost link quality scorer inside an interactive Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Physics — Detailed Explanation](#physics--detailed-explanation)
   - [1. Dynamic Orbital Geometry (SGP4)](#1-dynamic-orbital-geometry-sgp4)
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
   - [12. Dynamic Doppler Shift](#12-dynamic-doppler-shift)
   - [13. The Link Budget Equation](#13-the-link-budget-equation)
   - [14. Packet Loss Model](#14-packet-loss-model)
6. [The Simulation Loop & Propagation Pipeline](#the-simulation-loop--propagation-pipeline)
7. [Machine Learning Pipeline](#machine-learning-pipeline)
8. [Validation and Benchmarks](#validation-and-benchmarks)
   - [Validation Suite](#validation-suite)
   - [Performance Benchmarks](#performance-benchmarks)
9. [Tests](#tests)
10. [File Reference](#file-reference)
11. [Ground Station Parameters](#ground-station-parameters)
12. [UI Controls and What They Change](#ui-controls-and-what-they-change)
13. [Known Limitations](#known-limitations)
14. [References](#references)

---

## Project Overview

Satellite communications are degraded by a chain of physical phenomena: geometric path loss, thermal noise, rain attenuation, gaseous absorption, and tropospheric scintillation. While legacy simulators often assume static satellite positions, **real-world links are dynamic**. Even "geostationary" satellites drift in figure-8 patterns, and LEO/MEO constellations move rapidly across the sky.

This project implements a **Time-Aware Simulation Pipeline** that:

- Uses **SGP4 (Simplified General Perturbations)** propagation to track live ECEF position and velocity vectors for satellites stored in a SQLite database.
- Recomputes **Elevation**, **Slant Range**, and **Doppler** at every 60-second time step.
- Dynamically updates atmospheric losses (FSPL, Rain, Gas, Scintillation) as the satellite's geometric relationship to the ground station evolves.
- Models rain as a temporally correlated stochastic process (AR(1)) for realistic link fade persistence.
- Scores link quality using a pre-trained **XGBoost model** and visualises time-series data in a Streamlit dashboard.

The target use case is pre-deployment link budget analysis and what-if scenario testing for GEO Ku-band satellite uplinks.

---

## Repository Structure

```
.
├── app.py                    # Streamlit dashboard — UI, scoring, display
├── satellite_link_sim.py     # ITU-R physics engine & time-aware sim loop
├── propogate.py              # SGP4 Propagation Layer — ECEF/ENU transformations
├── ground_stations.py        # Single source of truth: all station parameters
├── database.py               # Satellite catalogue loader (TLEs to SQLite)
├── satellites.db             # SQLite database (satellite TLE catalogue)
├── geometry.py               # Geometry helpers (Legacy GEO formulas)
├── physicsengine.py          # Standalone physics utilities
├── link_quality.py           # Link quality scoring utilities
├── train_xgboost.py          # Model training script
├── link_training_data.csv    # Training dataset for XGBoost model
├── tests/                    # Pytest-based physics invariant & regression tests
│   ├── test_physics_invariants.py
│   └── test_regression.py
├── val_and_bench/            # Validation suite and Performance Benchmarks
│   ├── validation_correctness.py
│   ├── benchmarks.py         # Performance measurement script
│   ├── val_autocorr.png      # Validation plots
│   ├── val_geometry.png
│   ├── val_rain_attenuation.png
│   ├── bench_throughput.png  # Benchmark plots
│   ├── bench_scaling.png
│   └── bench_memory.png
├── LICENSE
└── README.md
```

The canonical files for the current version of the project are `app.py`, `satellite_link_sim.py`, and `ground_stations.py`. 

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
pip install streamlit pandas numpy scikit-learn xgboost joblib matplotlib sgp4
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

**Run the validation suite:**

```bash
python3 val_and_bench/validation_correctness.py
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

### 1. Dynamic Orbital Geometry (SGP4)

The project has moved beyond static GEO assumptions. While the simulator still supports fixed GEO longitudes, it now prioritises live orbital data via the `propogate.py` layer.

**SGP4 Propagation**: For any station linked to a `norad_id`, the simulator fetches the latest TLE (Two-Line Element) from `satellites.db` and propagates the orbit to the current simulation epoch.
- **ECEF State**: Returns (x, y, z) position and velocity in Earth-Centered, Earth-Fixed coordinates.
- **Topocentric Transformation**: Converts ECEF vectors into **ENU (East-North-Up)** coordinates relative to the ground station's WGS84 geodetic position.
- **Dynamic Geometry**: Recomputes Elevation (El) and Azimuth (Az) at every step.

**Why this matters**:
- **Slant Range Variation**: A satellite drifting or moving closer to the horizon increases the **Free-Space Path Loss** and **Rain Path Length** dynamically.
- **Doppler Dynamics**: Radial velocity is no longer a static guess; it is derived from the dot product of the relative velocity vector and the range vector.

If no `norad_id` is provided for a station, it falls back to the **Legacy GEO Geometry**:
- **Elevation**: `sin(El) = (cos γ − R_E / R_GEO) / sqrt(1 − cos²γ)`
- **Slant Range**: `d = sqrt(R_GEO² + R_E² − 2 · R_GEO · R_E · cos γ)`
- *Note: In legacy mode, geometry is constant for the entire simulation window.*

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

### 12. Dynamic Doppler Shift

With SGP4 integration, the Doppler shift is computed live from the satellite's state vector:

```
f_D = (v_radial / c) · f_c
```

Where `v_radial` is the projection of the relative velocity vector (Satellite - Ground Station) onto the line-of-sight vector. The ground station velocity includes the component from Earth's rotation (~460 m/s at the equator). This allows the simulator to model the rapid Doppler sweeps characteristic of non-GEO satellites.

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

## The Simulation Loop & Propagation Pipeline

The simulation in `satellite_link_sim.py` now operates as a stateful time-series engine. For each ground station:

1.  **Initialisation**: Look up `norad_id`. If found, initialise an SGP4 `Satrec` object via `propogate.py`.
2.  **Time Loop (per minute)**:
    - **Step A: Propagate.** Fetch (x, y, z) position and velocity at `T_now`.
    - **Step B: Geometry.** Compute `Elevation`, `Slant Range`, and `v_radial`.
    - **Step C: Static/Slow Losses.** Re-evaluate FSPL and Gaseous Absorption for the new distance/angle.
    - **Step D: Stochastic Losses.** Step the AR(1) Rain Process and Scintillation noise.
    - **Step E: Link Budget.** Calculate `SNR[t] = EIRP - Losses + Gain - Noise`.
    - **Step F: Clock.** Advance `T_now += 60s`.
3.  **Aggregation**: Return `StationResult` containing full time-series for SNR, Rain, Elevation, and Range.

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

## Validation and Benchmarks

The simulator includes an automated suite to ensure both physical accuracy and computational efficiency.

### Validation Suite
Located in `val_and_bench/validation_correctness.py`, this suite ensures physical models align with ITU standards and analytical references.

#### 1. Free-Space Path Loss (ITU-R P.525)
Validated against the standard formula: $L_{fs} = 92.45 + 20\log_{10}(f_{GHz}) + 20\log_{10}(d_{km})$. Accuracy within $10^{-4}$ dB.

#### 2. Rain Attenuation (ITU-R P.838 / P.839)
- **Coefficients:** Specific attenuation coefficients ($k, \alpha$) are verified via log-linear interpolation of ITU-R P.838-3 tables.
- **Rain Height:** Latitude-dependent model (P.839-4) is tested for climate zone accuracy (e.g., Delhi at 4.58 km).

![Rain Attenuation Validation](val_and_bench/val_rain_attenuation.png)

#### 3. Geometry & SGP4
- **Slant Range:** Analytical checks for Zenith ($90^\circ$) and Horizon ($0^\circ$) elevations.
- **SGP4 vs. Analytical:** Cross-validation of SGP4-propagated slant range against GEO analytical models.

![Geometry Validation](val_and_bench/val_geometry.png)

#### 4. Stochastic Rain Process (ITU-R P.1853)
- **Autocorrelation:** Verified decay constant $\rho = e^{-dt/\tau_c}$ matches the 5-minute ($\sim 300\text{s}$) correlation time.
- **Stationary Distribution:** Convergence to ITU-R P.837 lognormal mean across varying time steps.

![Autocorrelation Validation](val_and_bench/val_autocorr.png)

### Performance Benchmarks
Located in `val_and_bench/benchmarks.py`, this script measures the computational profile of the simulation loop.

#### 1. Simulation Throughput
Measures `timesteps/sec` across varying step counts. The engine achieves approximately **60,000 timesteps/sec** for large simulation windows, ensuring rapid UI responsiveness and batch processing capability.

![Throughput Benchmark](val_and_bench/bench_throughput.png)

#### 2. Runtime Scaling
Measures execution time as the number of concurrent ground stations increases. The simulation scales linearly, maintaining a profile of ~1.0 ms per 100-step station simulation.

![Scaling Benchmark](val_and_bench/bench_scaling.png)

#### 3. Memory Usage
Tracks the memory footprint of large simulations. A 500,000-step simulation consumes approximately **122 MB**, which is fully reclaimed after garbage collection.

![Memory Benchmark](val_and_bench/bench_memory.png)

#### 4. Performance Profiling
- **Propagation Latency:** The SGP4 `get_geometry` call averages **0.018 ms** per invocation.
- **CPU Profile:** The simulation effectively saturates a single core during heavy execution (100% process CPU utilization).

---

## Tests

The project uses `pytest` for comprehensive physics and regression testing. These tests ensure that any code changes do not violate fundamental physical invariants or break deterministic behavior.

### Running Tests
```bash
python3 -m pytest tests/
```

### Coverage
- **Physics Invariants:**
    - **FSPL Monotonicity:** Path loss must strictly increase with distance.
    - **Thermal Noise:** Noise power must scale correctly with bandwidth.
    - **Rain Attenuation:** Attenuation must increase with rainfall intensity (mm/h).
    - **Geometric Slant Path:** Lower elevation angles must result in longer propagation paths through the atmosphere.
- **Regression & Determinism:**
    - **Deterministic Seeds:** Using the same seed and start time must produce bit-identical SNR and rain series.
    - **Stochastic Control:** Verification that the `force_rain` flag correctly overrides probabilistic onset models.
    - **Statistical Integrity:** Verification of mean, min, and standard deviation calculations.
- **AR(1) Correlation:** 
    - Verified that the empirical autocorrelation of the Maseng-Bakken rain process matches the expected decay $\rho = e^{-\Delta t/\tau_c}$.

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
| `tests/` | Directory containing pytest-based physics and regression tests |
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

**TLE Age**: The accuracy of SGP4 propagation depends on the age of the TLEs in `satellites.db`. The initial dataset was sourced from **[CelesTrak](https://celestrak.org/)**. 

To update the satellite catalogue:
1.  Visit [CelesTrak](https://celestrak.org/NORAD/elements/) and download the latest TLE data (e.g., for GEO satellites).
2.  Open `database.py` and replace the `tle_text` string with the new TLE content.
3.  Run `python3 database.py` to rebuild the `satellites.db` with fresh orbital elements.

**ML Feature Window**: The XGBoost model currently scores based on `snr_mean` and `avg_pkt_loss`. It does not yet "see" the geometric trends (e.g., a link getting progressively worse as a satellite sets).

**Atmospheric Inhomogeneity**: While the path reduction factor `r` accounts for horizontal rain cell size, the model assumes a stratified atmosphere. Complex refraction effects at very low elevation (< 5°) are not fully modelled.

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
