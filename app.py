import streamlit as st
import numpy as np
import pandas as pd
import math
import joblib

from geometry import slant_range, effective_elevation
from ground_stations import GROUND_STATIONS

# ── Load ML assets ───────────────────────────────────────────────────────────
model  = joblib.load("xgb_link_model.pkl")
scaler = joblib.load("feature_scaler.pkl")

# ── Constants ────────────────────────────────────────────────────────────────
CARRIER_FREQ  = 14e9        # Hz
BANDWIDTH     = 36e6        # Hz
BOLTZMANN_DB  = -228.6      # dBW/K/Hz
ITU_K         = 0.03076     # ITU-R P.838-3 at 14 GHz vertical pol
ITU_ALPHA     = 1.1903


# ── Physics helpers ───────────────────────────────────────────────────────────

def itu_rain_height(lat_deg: float) -> float:
    """ITU-R P.839-4: mean rain height above MSL (km)."""
    a = abs(lat_deg)
    if a > 23:
        return max(5.0 - 0.075 * (a - 23.0), 3.0) if a < 36 else max(5.0 - 0.1 * (a - 36.0), 2.0)
    return 5.0


def effective_rain_path_km(elevation_deg: float, rain_height_km: float,
                            station_altitude_km: float) -> float:
    """ITU-R P.618-13: effective slant path through rain layer (km)."""
    el_rad  = math.radians(max(elevation_deg, 5.0))
    h_delta = rain_height_km - station_altitude_km
    if h_delta <= 0:
        return 0.0
    return h_delta / math.sin(el_rad)


def itu_rain_fade_db(rain_rate_mmh: float, eff_path_km: float) -> float:
    """A = k * R^alpha * L_eff  [dB]  (ITU-R P.838-3 at 14 GHz vertical)."""
    if rain_rate_mmh <= 0 or eff_path_km <= 0:
        return 0.0
    return ITU_K * (rain_rate_mmh ** ITU_ALPHA) * eff_path_km


def rain_rate_from_severity(severity: str, seed: int) -> float:
    """
    Sample a physically grounded rain rate from each station's ITU-R P.837-7
    lognormal distribution, conditioned on the severity label.

    Severity bands map to exceedance quantiles:
      low    →  R  in [R_1, R_01)   — moderate rain, exceeds 1 % but not 0.1 %
      medium →  R  in [R_01, R_001) — heavy rain, exceeds 0.1 % but not 0.01 %
      high   →  R  >= R_001         — extreme, exceeds 0.01 % threshold
    """
    # Build a lookup by severity so we can call this without a full gs dict
    # when only severity is known (e.g. legacy callers).
    SEVERITY_RATES = {
        "low":    (2.0,  8.0),   # mm/h range representative of R_1 level
        "medium": (8.0,  25.0),  # mm/h range representative of R_01 level
        "high":   (25.0, 80.0),  # mm/h range representative of R_001 level
    }
    rng = np.random.default_rng(seed)
    lo, hi = SEVERITY_RATES.get(severity, (0.0, 0.0))
    return float(rng.uniform(lo, hi))


def rain_fade_from_gs(gs: dict, weather: str, seed: int) -> float:
    """
    Compute ITU-R P.838-3 rain fade (dB) for a ground station dict.

    Uses:
      - gs["itu_rain"]["R01"] / R001 to anchor the rain rate sample
      - gs["latitude"] + gs["altitude_km"] for path geometry
      - gs["rain_severity"] for the severity band sampling

    Returns 0.0 if weather is "Clear".
    """
    if weather != "Rain":
        return 0.0

    # Sample a rain rate anchored to this station's ITU quantiles
    p      = gs["itu_rain"]
    rng    = np.random.default_rng(seed)
    sev    = gs["rain_severity"]
    # Map severity to a fraction of R_01 / R_001
    if sev == "low":
        rate = float(rng.uniform(p["R1"], p["R01"]))
    elif sev == "medium":
        rate = float(rng.uniform(p["R01"], p["R001"]))
    else:   # high
        rate = float(rng.uniform(p["R001"], min(p["R001"] * 1.5, 100.0)))

    # Geometry: use effective_elevation from geometry module
    # elevation comes from the caller; we approximate with 30° here if not set
    el_deg   = gs.get("_computed_elevation", 30.0)
    rain_h   = itu_rain_height(gs["latitude"])
    eff_path = effective_rain_path_km(el_deg, rain_h, gs["altitude_km"])

    return itu_rain_fade_db(rate, eff_path)


def packet_loss_from_snr(snr: float) -> float:
    """Smooth sigmoid mapping SNR → packet loss probability."""
    return 1 / (1 + math.exp(0.8 * (snr - 3)))


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("Multi-Ground-Station Link Quality Simulator")

altitude       = st.slider("Satellite Altitude (km)", 500, 36000, 36000)
base_elevation = st.slider("Reference Elevation Angle (deg)", 5, 90, 30)
load           = st.slider("Network Load", 0.0, 1.0, 0.4)
weather        = st.radio("Weather", ["Clear", "Rain"])

# ── Evaluate all stations ─────────────────────────────────────────────────────
results = []

for gs in GROUND_STATIONS:
    el_eff   = effective_elevation(base_elevation, gs["latitude"])
    distance = slant_range(altitude, el_eff)

    path_loss = (
        92.45
        + 20 * math.log10(CARRIER_FREQ / 1e9)
        + 20 * math.log10(distance)
    )

    # Stash computed elevation so rain_fade_from_gs can use it
    gs_with_el = {**gs, "_computed_elevation": el_eff}

    rain_fade = rain_fade_from_gs(
        gs_with_el,
        weather=weather,
        seed=hash(gs["name"]) % 10_000,
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
        columns=["snr_db", "packet_loss", "load_factor"],
    )
    X_scaled = scaler.transform(X)
    score    = model.predict(X_scaled)[0]

    results.append({
        "Station":               gs["name"],
        "Effective Elevation":   round(el_eff, 1),
        "Slant Range (km)":      round(distance, 1),
        "SNR (dB)":              round(snr, 2),
        "Rain Fade (dB)":        round(rain_fade, 2),
        "Packet Loss":           round(packet_loss, 3),
        "Link Quality":          round(score, 3),
    })

# ── Display ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(results).sort_values("Link Quality", ascending=False)

best_station = df.iloc[0]["Station"]

st.subheader(f"Best Ground Station: {best_station}")
st.dataframe(df, use_container_width=True)

st.subheader("Debug View (Physics + ML Inputs)")
st.dataframe(df)
