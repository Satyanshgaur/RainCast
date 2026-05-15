# 🛰️ LEO/MEO Satellite Link Optimiser System

## 📑 Index

1. [Project Overview](#-project-overview)
2. [File Structure & Responsibilities](#️-file-structure--responsibilities)
3. [Script-by-Script Breakdown](#️-script-by-script-breakdown)
   - [Orbital Propagation Engine](#propogatepy--orbital-propagation-engine)
   - [Core Physics Engine](#physicsenginepy--core-physics-engine)
   - [Orbital Geometry Calculator](#geometrypy--orbital-geometry-calculator)
   - [Ground Station Class](#groundstationpy--ground-station-class)
   - [Ground Station Registry](#ground_stationspy--groundstationspy--station-registry)
   - [RF Link Budget Calculator](#link_qualitypy--rf-link-budget-calculator)
   - [Database Layer](#databasepy--database-layer)
   - [Monte Carlo Simulation Engine](#satellite_link_simpy--monte-carlo-simulation-engine)
   - [ML Training Pipeline](#train_xgboostpy--ml-training-pipeline)
   - [Streamlit Dashboard](#apppy--streamlit-dashboard)
4. [Getting Started](#-getting-started)
   - [Prerequisites](#prerequisites)
   - [Clone the Repository](#1-clone-the-repository)
   - [Create and Activate a Virtual Environment](#2-create-and-activate-a-virtual-environment)
   - [Install Dependencies](#3-install-dependencies)
   - [Initialise the Databases](#4-initialise-the-databases)
   - [Generate Training Data](#5-generate-training-data)
   - [Train the Model](#6-train-the-model)
   - [Launch the Dashboard](#7-launch-the-dashboard)
5. [Running Individual Module Benchmarks](#-running-individual-module-benchmarks)
6. [Physics-Aided ML Architecture](#-physics-aided-ml-architecture)
7. [Key Physics Features in the ML Model](#-key-physics-features-in-the-ml-model)
8. [Data & Model Files](#-data--model-files)
9. [Data Source](#-data-source)
10. [Licence](#-licence)

---

## 📌 Project Overview

This system simulates and predicts the quality of communication links between ground stations and Low Earth Orbit (LEO) / Medium Earth Orbit (MEO) satellites. It is not a lightweight ML wrapper — every feature fed into the machine learning model is derived from first-principles physics: orbital propagation, geometric visibility calculations, atmospheric signal degradation, and Doppler shift modelling.

The trained model is served through an interactive Streamlit dashboard (`app.py`) that allows operators to visualise link quality in real time across multiple ground stations and satellite passes.

---

## 🗂️ File Structure & Responsibilities

```text
.
├── app.py                    # Streamlit dashboard — main entry point
├── database.py               # Database schema setup and ORM helpers
├── feature_scaler.pkl        # Serialised sklearn scaler (output of train_xgboost.py)
├── geometry.py               # Orbital geometry: elevation, azimuth, range calculations
├── ground_stations.py        # Static ground station registry and metadata
├── groundstation.py          # Ground station class definition and pass window logic
├── groundstations.py         # Ground station collection manager / multi-station handler
├── link_quality.py           # Physics-based link budget: EIRP, path loss, SNR, C/N₀
├── link_training_data.csv    # Generated training dataset (output of satellite_link_sim.py)
├── physicsengine.py          # Core physics engine: Doppler, atmospheric loss, free-space loss
├── propogate.py              # SGP4/SDP4 TLE orbital propagator (via python-sgp4)
├── satellite_link_sim.py     # Monte Carlo simulation engine — generates link_training_data.csv
├── satellite.db              # SQLite DB for satellite metadata (output of database.py)
├── satellites.db             # SQLite DB for pass predictions and link records
├── train_xgboost.py          # XGBoost model training pipeline — outputs xgb_link_model.pkl
└── xgb_link_model.pkl        # Serialised trained XGBoost model (output of train_xgboost.py)

```

---

## ⚙️ Script-by-Script Breakdown

### `propogate.py` — Orbital Propagation Engine

Wraps the SGP4/SDP4 propagation standard to ingest Two-Line Element (TLE) data fetched from CelesTrak and compute satellite position and velocity vectors (ECI frame) at arbitrary time steps. Outputs ECEF coordinates used downstream by the geometry and physics modules.

### `physicsengine.py` — Core Physics Engine

The heart of the simulation. Computes all physics quantities that determine link viability:

* **Free-Space Path Loss (FSPL)** using the Friis transmission equation.
* **Doppler frequency shift** from satellite radial velocity.
* **Atmospheric attenuation** (tropospheric, ionospheric absorption models).
* **Rain fade margin** using ITU-R P.618 approximations.

### `geometry.py` — Orbital Geometry Calculator

Converts ECI/ECEF position vectors from `propogate.py` into observer-relative quantities:

* Elevation angle from ground station to satellite.
* Azimuth angle for antenna pointing.
* Slant range (km) for path loss input.

### `groundstation.py` — Ground Station Class

Defines the `GroundStation` object: geodetic coordinates (lat/lon/alt), antenna parameters (gain, frequency, noise temperature), and methods to compute pass windows.

### `ground_stations.py` / `groundstations.py` — Station Registry

Stores a static registry of named ground stations and provides a collection manager for multi-station scenarios (handoff points and concurrent pass overlap).

### `link_quality.py` — RF Link Budget Calculator

Assembles all physics outputs into a complete link budget:

* Transmit EIRP (dBW)
* Receive G/T (dB/K)
* Carrier-to-Noise density ratio (C/N₀)
* Binary `link_viable` label used as the ML training target.

### `database.py` — Database Layer

Defines the SQLite schema for `satellite.db` and `satellites.db` using an ORM-style interface.

### `satellite_link_sim.py` — Monte Carlo Simulation Engine

The data generation pipeline. Runs thousands of simulated satellite passes across all ground stations to generate `link_training_data.csv`.

### `train_xgboost.py` — ML Training Pipeline

Loads training data, applies feature engineering, fits a `StandardScaler`, and trains an XGBoost classifier to predict link quality margin.

### `app.py` — Streamlit Dashboard

The operator-facing interface. Fetches fresh TLE data, propagates current positions, and displays live satellite ground tracks and link quality predictions.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* pip
* Internet access (for CelesTrak TLE fetches at runtime)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/leo-meo-satellite-link-optimiser.git
cd leo-meo-satellite-link-optimiser

```

### 2. Create and Activate a Virtual Environment

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Initialise the Databases

```bash
python3 database.py

```

### 5. Generate Training Data

```bash
python3 satellite_link_sim.py

```

### 6. Train the Model

```bash
python3 train_xgboost.py

```

### 7. Launch the Dashboard

```bash
streamlit run app.py

```

---

## 🔬 Running Individual Module Benchmarks

Each module can be executed directly to print internal benchmarks:

* `python3 propogate.py` (Propagation throughput)
* `python3 physicsengine.py` (Physics computation rates)
* `python3 link_quality.py` (Link budget throughput)

---

## 🧠 Physics-Aided ML Architecture

The model learns over a rich feature space derived from physical laws rather than raw data:

1. **CelesTrak TLE** ➔ `propogate.py` (SGP4)
2. **Orbital Mechanics** ➔ `geometry.py` (Elevation/Range)
3. **RF Physics** ➔ `physicsengine.py` (Loss/Doppler)
4. **Link Budget** ➔ `link_quality.py` (C/N₀)
5. **XGBoost** ➔ `app.py` (Real-time Prediction)

---

## 📊 Key Physics Features in the ML Model

| Feature | Physical Origin |
| --- | --- |
| **Elevation angle** | Geometry: observer–satellite vector |
| **Slant range** | Geometry: Euclidean distance |
| **Free-space loss** | Friis equation |
| **Doppler shift** | Radial velocity |
| **Rain fade** | ITU-R P.618 model |

---

## 📁 Data & Model Files

| File | Created By | Description |
| --- | --- | --- |
| `satellite.db` | `database.py` | Satellite metadata |
| `link_training_data.csv` | `satellite_link_sim.py` | ML training dataset |
| `xgb_link_model.pkl` | `train_xgboost.py` | Trained XGBoost model |

---

## 📡 Data Source

Live TLE orbital elements are fetched from **CelesTrak**. Supports LEO (Starlink, OneWeb) and MEO (GPS, Galileo).

## 📄 Licence

MIT Licence. See LICENSE for details.
