import streamlit as st
import numpy as np
import pandas as pd
import math
import joblib

from geometry import slant_range, effective_elevation
from ground_stations import GROUND_STATIONS

# -----------------------------
# Load ML assets
# -----------------------------
model = joblib.load("xgb_link_model.pkl")
scaler = joblib.load("feature_scaler.pkl")

# -----------------------------
# Constants
# -----------------------------
CARRIER_FREQ = 14e9       # Hz
BANDWIDTH = 36e6          # Hz
BOLTZMANN_DB = -228.6     # dBW/K/Hz

# -----------------------------
# Helpers
# -----------------------------
def rain_fade_from_severity(severity, seed):
    rng = np.random.default_rng(seed)
    if severity == "low":
        return rng.uniform(0, 2)
    if severity == "medium":
        return rng.uniform(3, 7)
    if severity == "high":
        return rng.uniform(8, 15)
    return 0.0


def packet_loss_from_snr(snr):
    # smooth sigmoid-like curve
    return 1 / (1 + math.exp(0.8 * (snr - 3)))


# -----------------------------
# UI
# -----------------------------
st.title("Multi-Ground-Station Link Quality Simulator")

altitude = st.slider("Satellite Altitude (km)", 500, 36000, 36000)
base_elevation = st.slider("Reference Elevation Angle (deg)", 5, 90, 30)
load = st.slider("Network Load", 0.0, 1.0, 0.4)
weather = st.radio("Weather", ["Clear", "Rain"])

# -----------------------------
# Evaluate all stations
# -----------------------------
results = []

for gs in GROUND_STATIONS:
    el_eff = effective_elevation(base_elevation, gs["latitude"])
    distance = slant_range(altitude, el_eff)

    path_loss = (
        92.45
        + 20 * math.log10(CARRIER_FREQ / 1e9)
        + 20 * math.log10(distance)
    )

    rain_fade = 0.0
    if weather == "Rain":
        rain_fade = rain_fade_from_severity(
            gs["rain_severity"],
            seed=hash(gs["name"]) % 10000
        )

    noise_dbw = (
        BOLTZMANN_DB
        + 10 * math.log10(gs["system_temp_k"])
        + 10 * math.log10(BANDWIDTH)
    )

    snr = (
        gs["eirp_dbw"]
        - path_loss
        - gs["atm_loss_db"]
        - rain_fade
        + gs["g_rx_dbi"]
        - noise_dbw
    )

    packet_loss = packet_loss_from_snr(snr)

    X = pd.DataFrame(
        [[snr, packet_loss, load]],
        columns=["snr_db", "packet_loss", "load_factor"]
    )

    X_scaled = scaler.transform(X)
    score = model.predict(X_scaled)[0]

    results.append({
        "Station": gs["name"],
        "Effective Elevation (deg)": round(el_eff, 1),
        "Slant Range (km)": round(distance, 1),
        "SNR (dB)": round(snr, 2),
        "Rain Fade (dB)": round(rain_fade, 2),
        "Packet Loss": round(packet_loss, 3),
        "Link Quality": round(score, 3)
    })

# -----------------------------
# Display
# -----------------------------
df = pd.DataFrame(results).sort_values("Link Quality", ascending=False)

best_station = df.iloc[0]["Station"]

st.subheader(f"🏆 Best Ground Station: {best_station}")
st.dataframe(df, use_container_width=True)

st.subheader("Debug View (Physics + ML Inputs)")
st.dataframe(df)
