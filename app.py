"""
app.py — Streamlit UI for the Satellite Link Quality Simulator
==============================================================
All physics are delegated to satellite_link_sim.simulate_station().
This file is responsible for:
  1. UI controls (sidebar sliders, weather mode, load)
  2. Calling simulate_station() once per ground station
  3. Building the ML feature vector from StationResult statistics
  4. Scoring with feature_scaler + xgb_link_model
  5. Displaying results: ranked table, SNR time series, per-station breakdown

ML feature vector — must match the column names seen at scaler/model fit time:
  snr_db       — mapped from snr_mean (avg SNR over simulation window)
  packet_loss  — mapped from avg_pkt_loss (mean packet loss probability)
  load_factor  — user-supplied network load slider

The richer physics statistics (snr_p10, snr_std, outage_fraction, etc.) are
displayed in the UI but are NOT passed to the model — the model was trained on
the original 3-feature schema.  Retrain on the full 5-feature schema to unlock
the stability and near-outage margin signals.
"""

import math
import streamlit as st
import numpy as np
import pandas as pd
import joblib

from ground_stations import GROUND_STATIONS
from satellite_link_sim import simulate_station, simulate_all, StationResult, DEFAULT_N_STEPS


# ── Load ML assets ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("xgb_link_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    return model, scaler

model, scaler = load_model()


# ── ML feature construction ───────────────────────────────────────────────────

# Column names MUST match exactly what feature_scaler.pkl and xgb_link_model.pkl
# were trained on.  The sim produces richer statistics; we map them here.
FEATURE_COLS = ["snr_db", "packet_loss", "load_factor"]

def build_features(r: StationResult, load: float) -> pd.DataFrame:
    """
    Map StationResult physics statistics to the 3-column schema the trained
    scaler and model expect.

      snr_mean     → snr_db       (average SNR over the simulation window)
      avg_pkt_loss → packet_loss  (mean packet loss probability)
      load                        (passed through unchanged)

    The additional statistics (snr_p10, snr_std, outage_fraction) are shown
    in the UI but not passed to the model until the model is retrained on the
    expanded feature set.
    """
    return pd.DataFrame([[
        r.snr_mean,       # snr_db
        r.avg_pkt_loss,   # packet_loss
        load,             # load_factor
    ]], columns=FEATURE_COLS)


def score_station(r: StationResult, load: float) -> float:
    """Run the ML pipeline on a StationResult and return the link quality score."""
    X        = build_features(r, load)
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Satellite Link Simulator", layout="wide")
st.title("Multi-Ground-Station Link Quality Simulator")
st.caption("Physics: ITU-R P.837/838/839/618/676/1853 · ML: XGBoost link quality scorer")

# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Simulation controls")

    weather = st.radio(
        "Weather mode",
        ["Clear", "Rain"],
        help="Rain locks the AR(1) process into its raining state for all steps.",
    )
    force_rain = (weather == "Rain")

    load = st.slider(
        "Network load",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="Fraction of link capacity utilised — fed directly to the ML model.",
    )

    n_steps = st.slider(
        "Simulation window (minutes)",
        min_value=10, max_value=120, value=60, step=10,
        help="Each step is 60 s.  Longer windows give more stable statistics.",
    )

    st.divider()
    st.caption("Satellite: Ku-band 14 GHz uplink\nPolarisation: vertical\nPer-station GEO arc")

# ── Run simulation for all stations ──────────────────────────────────────────

@st.cache_data(show_spinner="Running ITU-R physics simulation…")
def run_all(n_steps: int, force_rain: bool):
    """Cache results keyed on (n_steps, force_rain) so sliders are responsive."""
    gs_map = {gs["name"]: gs for gs in GROUND_STATIONS}
    results = []
    for gs in GROUND_STATIONS:
        r = simulate_station(
            gs,
            n_steps    = n_steps,
            force_rain = force_rain,
            seed       = hash(gs["name"]) % 100_000,
        )
        results.append((gs, r))
    return results

sim_results = run_all(n_steps=n_steps, force_rain=force_rain)

# ── Score all stations ────────────────────────────────────────────────────────

rows = []
for gs, r in sim_results:
    score = score_station(r, load)
    rows.append({
        "Station":          r.name,
        "Elevation (°)":    round(r.elevation, 1),
        "Slant range (km)": round(r.slant_km, 0),
        "SNR mean (dB)":    round(r.snr_mean, 2),
        "SNR p10 (dB)":     round(r.snr_p10, 2),
        "SNR std (dB)":     round(r.snr_std, 2),
        "Rain fraction (%)": round(r.rain_fraction * 100, 1),
        "Avg rain atten (dB)": round(r.avg_rain_db, 2),
        "Outage (%)":       round(r.outage_fraction * 100, 1),
        "Avg pkt loss":     round(r.avg_pkt_loss, 4),
        "Link quality":     round(score, 3),
        "_result":          r,   # kept for charts, dropped before display
    })

df_full = pd.DataFrame(rows).sort_values("Link quality", ascending=False)
df_display = df_full.drop(columns=["_result"]).reset_index(drop=True)

best = df_full.iloc[0]

# ── Summary header ────────────────────────────────────────────────────────────

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Best station",   best["Station"])
col_b.metric("Link quality",   f"{best['Link quality']:.3f}")
col_c.metric("SNR mean",       f"{best['SNR mean (dB)']:.1f} dB")
col_d.metric("Outage",         f"{best['Outage (%)']:.1f} %")

st.divider()

# ── Ranked results table ──────────────────────────────────────────────────────

st.subheader("Station ranking")
st.dataframe(
    df_display.style.background_gradient(subset=["Link quality"], cmap="RdYlGn")
                    .background_gradient(subset=["Outage (%)"],    cmap="RdYlGn_r")
                    .format(precision=2),
    use_container_width=True,
    hide_index=True,
)

# ── SNR time series chart ─────────────────────────────────────────────────────

st.subheader("SNR time series")

snr_df = pd.DataFrame({
    row["Station"]: row["_result"].snr_series
    for _, row in df_full.iterrows()
})
snr_df.index.name = "Step (min)"
st.line_chart(snr_df, use_container_width=True)

# ── Rain attenuation time series ──────────────────────────────────────────────

st.subheader("Rain attenuation time series")

rain_db_df = pd.DataFrame({
    row["Station"]: row["_result"].rain_db_series
    for _, row in df_full.iterrows()
})
rain_db_df.index.name = "Step (min)"
st.line_chart(rain_db_df, use_container_width=True)

# ── Per-station physics breakdown ─────────────────────────────────────────────

st.subheader("Per-station physics breakdown")

for _, row in df_full.iterrows():
    r: StationResult = row["_result"]
    with st.expander(f"{r.name}  —  link quality {row['Link quality']:.3f}"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Elevation",     f"{r.elevation:.1f} °")
        c1.metric("Slant range",   f"{r.slant_km:.0f} km")
        c1.metric("Doppler shift", f"{r.doppler_hz:+.0f} Hz")

        c2.metric("FSPL",          f"{r.path_loss:.2f} dB")
        c2.metric("Gas absorption",f"{r.gas_loss:.3f} dB")
        c2.metric("Rain height",   f"{r.rain_height:.2f} km")

        c3.metric("Eff. rain path",f"{r.eff_path:.2f} km")
        c3.metric("Scint. σ",      f"{r.scint_sig:.4f} dB")
        c3.metric("ITU k / α",     f"{r.itu_k:.4f} / {r.itu_alpha:.3f}")

        c4.metric("SNR mean",      f"{r.snr_mean:.2f} dB")
        c4.metric("SNR p10",       f"{r.snr_p10:.2f} dB")
        c4.metric("Avg pkt loss",  f"{r.avg_pkt_loss:.4f}")

        # Mini SNR chart inside the expander
        st.line_chart(
            pd.DataFrame({"SNR (dB)": r.snr_series,
                          "Rain atten (dB)": r.rain_db_series}),
            use_container_width=True,
        )

        # ML feature vector for this station
        st.caption("ML feature vector sent to XGBoost:")
        st.dataframe(build_features(r, load), use_container_width=True, hide_index=True)
