"""
app.py — Streamlit UI for the Satellite Link Quality Simulator
==============================================================
All physics are delegated to satellite_link_sim.simulate_station().
"""

import streamlit as st
import pandas as pd
import joblib
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timezone

from satlinksim.ground_stations import GROUND_STATIONS
from satlinksim.config import config
from satlinksim.domain.models import StationResult, Satellite, Constellation
from satlinksim.application.simulation_engine import (
    SimulationEngine,
    DEFAULT_CARRIER_FREQ_HZ, DEFAULT_BANDWIDTH_HZ, DEFAULT_POLARIZATION,
    DEFAULT_N_STEPS, SNR_THRESHOLD_DB,
)
from satlinksim.infrastructure.api.client import SatLinkSimClient
from satlinksim.infrastructure.api.schemas import SimulationRequest, GroundStation, ItuRain, ConstellationSchema, SatelliteSchema

# ── ML assets ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        # Use absolute paths relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ml_dir = os.path.join(os.path.dirname(current_dir), "ml")
        model_path = os.path.join(ml_dir, "xgb_link_model.pkl")
        scaler_path = os.path.join(ml_dir, "feature_scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.warning(f"ML assets not found in {ml_dir}. Run the trainer script first.")
            return None, None
            
        return joblib.load(model_path), joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Error loading ML models: {e}")
        return None, None

model, scaler = load_model()

FEATURE_COLS = ["snr_db", "packet_loss", "load_factor"]

def build_features(r: StationResult, load: float) -> pd.DataFrame:
    return pd.DataFrame(
        [[r.snr_mean, r.avg_pkt_loss, load]],
        columns=FEATURE_COLS,
    )

def score_station(r: StationResult, load: float) -> float:
    if model is None or scaler is None:
        return 0.5 # Fallback
    return float(model.predict(scaler.transform(build_features(r, load)))[0])


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Satellite Link Simulator", layout="wide")
st.title("Multi-Ground-Station Link Quality Simulator")
st.caption(
    "Physics: ITU-R P.837 / P.838 / P.839 / P.618 / P.676 / P.1853  "
    "·  ML scorer: XGBoost  ·  Handoff: Hysteresis + Dwell Time"
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Sidebar — all user controls                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with st.sidebar:
    st.header("RF / Link layer")

    freq_ghz = st.slider(
        "Carrier frequency (GHz)",
        min_value=10.0, max_value=30.0, value=14.0, step=0.5,
    )

    polarization = st.radio(
        "Polarization",
        ["vertical", "horizontal"],
    )

    bw_mhz = st.slider(
        "Transponder bandwidth (MHz)",
        min_value=18, max_value=72, value=36, step=9,
    )

    eirp_offset = st.slider(
        "EIRP offset (dB)",
        min_value=-10.0, max_value=10.0, value=0.0, step=0.5,
    )

    st.divider()
    st.header("Environment")

    weather = st.radio(
        "Weather mode",
        ["Clear (probabilistic)", "Rain (forced)"],
    )
    force_rain = weather.startswith("Rain")

    rain_scale = st.slider(
        "Rain intensity scale",
        min_value=0.5, max_value=3.0, value=1.0, step=0.25,
        disabled=not force_rain,
    )

    st.divider()
    st.header("Constellation & Handoff")

    use_constellation = st.checkbox("Enable Multi-Sat Constellation", value=True)
    
    selected_constellation = None
    handoff_policy = "highest_elevation"
    hysteresis = config.simulation.handoff.hysteresis_db
    min_dwell = config.simulation.handoff.dwell_steps

    if use_constellation:
        const_choice = st.selectbox(
            "Select Constellation",
            ["Starlink (Mock 3-Sat)", "OneWeb (Mock 2-Sat)", "Custom (IDs)"]
        )
        
        if const_choice == "Starlink (Mock 3-Sat)":
            # Using actual LEO satellites from DB so they move and trigger handoffs
            selected_constellation = Constellation.from_norad_ids("Starlink-Mock", [44057, 44059, 44061])
        elif const_choice == "OneWeb (Mock 2-Sat)":
            selected_constellation = Constellation.from_norad_ids("OneWeb-Mock", [45131, 45132])
        else:
            ids_str = st.text_input("NORAD IDs (comma separated)", "44057, 45131")
            try:
                ids = [int(i.strip()) for i in ids_str.split(",")]
                selected_constellation = Constellation.from_norad_ids("Custom", ids)
            except:
                st.error("Invalid IDs")

        handoff_policy = st.selectbox("Switching Policy", ["highest_elevation", "highest_snr"])
        hysteresis = st.slider("Hysteresis (dB/deg)", 0.0, 10.0, float(config.simulation.handoff.hysteresis_db), 0.5)
        min_dwell = st.slider("Min Dwell (steps)", 1, 10, int(config.simulation.handoff.dwell_steps))

    st.divider()
    st.header("Simulation & Parallelism")

    sim_mode = st.selectbox(
        "Execution Mode",
        ["Standard (Batched NumPy)", "Concurrent (Async Propagation)", "Monte Carlo (Multiprocessing)", "REST API (Remote)"],
    )

    api_url = "http://localhost:8000"
    if sim_mode == "REST API (Remote)":
        api_url = st.text_input("API URL", value="http://localhost:8000")
        client = SatLinkSimClient(base_url=api_url)
        if client.health_check():
            st.sidebar.success("● API Connected")
        else:
            st.sidebar.error("○ API Disconnected")
            st.info(
                "### 🛰️ Setup Required\n"
                "To use the Remote API mode, you must start the simulation service in a separate terminal:\n\n"
                "```bash\n"
                "satlinksim api\n"
                "```\n"
                "Once the server is running, the status above will turn green."
            )

    window_minutes = st.slider(
        "Window (minutes)", min_value=1, max_value=180, value=10, step=1,
    )
    # Convert minutes to number of steps (1 step = 1 second)
    n_steps = window_minutes * 60

    mc_iterations = 1
    if sim_mode == "Monte Carlo (Multiprocessing)":
        mc_iterations = st.slider("MC Iterations", 2, 20, 5)

    load = st.slider(
        "Network load", min_value=0.0, max_value=1.0, value=0.4, step=0.05,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Run simulation                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_simulation(sim_mode, mc_iterations, n_steps, force_rain, freq_hz, 
                   polarization, bandwidth_hz, eirp_offset_db, rain_rate_scale,
                   constellation=None, handoff_policy="highest_elevation",
                   hysteresis=2.0, min_dwell_steps=2, api_url="http://localhost:8000"):
    
    if sim_mode == "REST API (Remote)":
        client = SatLinkSimClient(base_url=api_url)
        
        # Prepare request
        gs_models = []
        for gs in GROUND_STATIONS:
            itu = ItuRain(**gs["itu_rain"])
            gs_models.append(GroundStation(
                name=gs["name"],
                eirp_dbw=gs["eirp_dbw"],
                g_rx_dbi=gs["g_rx_dbi"],
                system_temp_k=gs["system_temp_k"],
                antenna_diam_m=gs["antenna_diam_m"],
                latitude=gs["latitude"],
                longitude=gs["longitude"],
                altitude_km=gs["altitude_km"],
                itu_rain=itu,
                wv_g_m3=gs["wv_g_m3"],
                humidity_pct=gs["humidity_pct"],
                v_radial_ms=gs["v_radial_ms"],
                norad_id=gs.get("norad_id"),
                sat_name=gs.get("sat_name"),
                sat_lon_deg=gs.get("sat_lon_deg")
            ))

        const_schema = None
        if constellation:
            sats = [
                SatelliteSchema(norad_id=s.norad_id, name=s.name, tle_line1=s.tle_line1, tle_line2=s.tle_line2)
                for s in constellation.satellites
            ]
            const_schema = ConstellationSchema(name=constellation.name, satellites=sats)

        request = SimulationRequest(
            ground_stations=gs_models,
            n_steps=n_steps,
            dt_s=1.0,
            force_rain=force_rain,
            freq_hz=freq_hz,
            eirp_offset_db=eirp_offset_db,
            bandwidth_hz=bandwidth_hz,
            polarization=polarization,
            rain_rate_scale=rain_rate_scale,
            constellation=const_schema,
            handoff_policy=handoff_policy,
            hysteresis=hysteresis,
            min_dwell_steps=min_dwell_steps
        )
        
        try:
            response = client.simulate(request)
            # Convert StationResultSchema (Pydantic) back to StationResult (Dataclass)
            # so the rest of the app logic (plotting, scoring) works seamlessly.
            results = []
            for r in response.results:
                results.append(StationResult(
                    name=r.name,
                    elevation=r.elevation,
                    slant_km=r.slant_km,
                    doppler_hz=r.doppler_hz,
                    path_loss=r.path_loss,
                    gas_loss=r.gas_loss,
                    rain_height=r.rain_height,
                    eff_path=r.eff_path,
                    itu_k=r.itu_k,
                    itu_alpha=r.itu_alpha,
                    scint_sig=r.scint_sig,
                    noise_floor=r.noise_floor,
                    snr_series=r.snr_series,
                    rain_series=r.rain_series,
                    rain_db_series=r.rain_db_series,
                    scint_series=r.scint_series,
                    pkt_loss_series=r.pkt_loss_series,
                    elevation_series=r.elevation_series,
                    slant_range_series=r.slant_range_series,
                    doppler_series=r.doppler_series,
                    snr_mean=r.snr_mean,
                    snr_min=r.snr_min,
                    snr_std=r.snr_std,
                    snr_p10=r.snr_p10,
                    rain_fraction=r.rain_fraction,
                    avg_rain_db=r.avg_rain_db,
                    avg_pkt_loss=r.avg_pkt_loss,
                    outage_fraction=r.outage_fraction,
                    sat_name_series=r.sat_name_series,
                    handoff_events=r.handoff_events
                ))
            return [results]
        except Exception as e:
            st.error(f"API Error: {e}")
            return []

    kwargs = dict(
        n_steps         = n_steps,
        dt_s            = 1.0,
        force_rain      = force_rain,
        freq_hz         = freq_hz,
        eirp_offset_db  = eirp_offset_db,
        bandwidth_hz    = bandwidth_hz,
        polarization    = polarization,
        rain_rate_scale = rain_rate_scale,
        constellation   = constellation,
        handoff_policy  = handoff_policy,
        hysteresis      = hysteresis,
        min_dwell_steps = min_dwell_steps
    )
    
    engine = SimulationEngine()
    if sim_mode == "Monte Carlo (Multiprocessing)":
        return engine.run_monte_carlo(mc_iterations, GROUND_STATIONS, **kwargs)
    elif sim_mode == "Concurrent (Async Propagation)":
        return [asyncio.run(engine.simulate_all_concurrent(GROUND_STATIONS, **kwargs))]
    else:
        return [engine.simulate_all_batched(GROUND_STATIONS, **kwargs)]

all_iterations_results = run_simulation(
    sim_mode        = sim_mode,
    mc_iterations   = mc_iterations,
    n_steps         = n_steps,
    force_rain      = force_rain,
    freq_hz         = freq_ghz * 1e9,
    polarization    = polarization,
    bandwidth_hz    = bw_mhz * 1e6,
    eirp_offset_db  = eirp_offset,
    rain_rate_scale = rain_scale if force_rain else 1.0,
    constellation   = selected_constellation,
    handoff_policy  = handoff_policy,
    hysteresis      = hysteresis,
    min_dwell_steps = min_dwell,
    api_url         = api_url
)

sim_results = all_iterations_results[0] if all_iterations_results else []

if not sim_results:
    st.error("No simulation results available. Check your API server or settings.")
    st.stop()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Score and build display table                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

rows = []
for r in sim_results:
    score = score_station(r, load)
    rows.append({
        "Station":             r.name,
        "Elevation (°)":       round(r.elevation, 1),
        "Slant range (km)":    round(r.slant_km, 0),
        "FSPL (dB)":           round(r.path_loss, 2),
        "Gas loss (dB)":       round(r.gas_loss, 3),
        "SNR mean (dB)":       round(r.snr_mean, 2),
        "SNR p10 (dB)":        round(r.snr_p10, 2),
        "Outage (%)":          round(r.outage_fraction * 100, 1),
        "Avg pkt loss":        round(r.avg_pkt_loss, 4),
        "Handoffs":            len(r.handoff_events) if r.handoff_events else 0,
        "Link quality":        round(score, 3),
        "_result":             r,
    })

df_full    = pd.DataFrame(rows).sort_values("Link quality", ascending=False)
df_display = df_full.drop(columns=["_result"]).reset_index(drop=True)
best       = df_full.iloc[0]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Summary metrics row                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Best station",    best["Station"])
c2.metric("Link quality",    f"{best['Link quality']:.3f}")
c3.metric("SNR mean",        f"{best['SNR mean (dB)']:.1f} dB")
c4.metric("SNR p10",         f"{best['SNR p10 (dB)']:.1f} dB")
c5.metric("Outage",          f"{best['Outage (%)']:.1f} %")

st.divider()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Ranked table                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.subheader("Station ranking")
st.dataframe(
    df_display.style
        .background_gradient(subset=["Link quality"],        cmap="RdYlGn")
        .background_gradient(subset=["SNR mean (dB)"],       cmap="RdYlGn")
        .background_gradient(subset=["Outage (%)"],          cmap="RdYlGn_r")
        .format(precision=2),
    use_container_width=True,
    hide_index=True,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Time series charts                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

tab_snr, tab_rain, tab_pkt, tab_geo = st.tabs(["SNR time series", "Rain attenuation", "Packet loss", "Geometry"])

with tab_snr:
    snr_data = {row["Station"]: row["_result"].snr_series for _, row in df_full.iterrows()}
    st.line_chart(pd.DataFrame(snr_data), use_container_width=True)

with tab_rain:
    rain_data = {row["Station"]: row["_result"].rain_db_series for _, row in df_full.iterrows()}
    st.line_chart(pd.DataFrame(rain_data), use_container_width=True)

with tab_pkt:
    pkt_data = {row["Station"]: row["_result"].pkt_loss_series for _, row in df_full.iterrows()}
    st.line_chart(pd.DataFrame(pkt_data), use_container_width=True)

with tab_geo:
    geo_data = {row["Station"]: row["_result"].elevation_series for _, row in df_full.iterrows()}
    st.line_chart(pd.DataFrame(geo_data), use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Per-station physics breakdown (expandable)                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.subheader("Per-station physics breakdown")

for _, row in df_full.iterrows():
    r: StationResult = row["_result"]
    with st.expander(
        f"{r.name}  —  quality {row['Link quality']:.3f}  "
        f"|  SNR mean {r.snr_mean:.1f} dB  |  outage {r.outage_fraction*100:.0f}%"
    ):
        col_geo, col_prop, col_stat = st.columns(3)

        with col_geo:
            st.markdown("**Geometry**")
            st.metric("Elevation",     f"{r.elevation:.1f} °")
            st.metric("Slant range",   f"{r.slant_km:.0f} km")
            st.metric("Doppler shift", f"{r.doppler_hz:+.0f} Hz")
            if r.sat_name_series:
                st.caption(f"Current Sat: {r.sat_name_series[0]}")

        with col_prop:
            st.markdown("**Propagation**")
            st.metric("FSPL",           f"{r.path_loss:.2f} dB")
            st.metric("Gas absorption", f"{r.gas_loss:.3f} dB")
            st.metric("Noise floor",    f"{r.noise_floor:.1f} dBW")

        with col_stat:
            st.markdown("**Link statistics**")
            st.metric("SNR mean",  f"{r.snr_mean:.2f} dB")
            st.metric("SNR p10",   f"{r.snr_p10:.2f} dB")
            st.metric("Pkt loss",  f"{r.avg_pkt_loss:.4f}")
            st.metric("Handoffs",  len(r.handoff_events) if r.handoff_events else 0)

        if r.handoff_events:
            st.markdown("**Handoff History**")
            ho_df = pd.DataFrame([{
                "Step": e.time_step,
                "From": e.old_sat,
                "To": e.new_sat,
                "Reason": e.reason,
                "Delta": f"{e.metric_delta:+.2f}"
            } for e in r.handoff_events])
            st.dataframe(ho_df, use_container_width=True, hide_index=True)

        st.line_chart(
            pd.DataFrame({
                "SNR (dB)":           r.snr_series,
                "Rain atten (dB)":    r.rain_db_series,
            }),
            use_container_width=True,
        )
