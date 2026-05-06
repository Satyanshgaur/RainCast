"""
ground_stations.py — Single source of truth for all ground station data.
=========================================================================
Consumed by:
  - satellite_link_sim.py  (physics simulation, ITU-R models, AR(1) rain)
  - app.py                 (Streamlit UI, ML scoring, link budget)

Every key used by either file lives here. Do not duplicate or override
these values elsewhere.

Key groups per entry:
  RF hardware   : eirp_dbw, g_rx_dbi, system_temp_k, antenna_diam_m
  Legacy compat : atm_loss_db, rain_severity   ← used by app.py / ML pipeline
  Geometry      : latitude, longitude, sat_lon_deg, altitude_km
  ITU-R P.837-7 : itu_rain  {R001, R01, R1, P_rain}
  ITU-R P.676   : wv_g_m3, humidity_pct
  Dynamics      : v_radial_ms
"""

GROUND_STATIONS = [
    # ── Delhi ──────────────────────────────────────────────────────────────
    # ITU rain zone K (monsoon).  Satellite: INSAT-4A/SES-8 @ 83.0 E
    {
        "name":           "Delhi",
        # RF
        "eirp_dbw":       52,
        "g_rx_dbi":       45,
        "system_temp_k":  500,
        "antenna_diam_m": 2.4,
        # Legacy / ML-pipeline
        "atm_loss_db":    1.2,
        "rain_severity":  "medium",
        # Geometry
        "latitude":       28.6,
        "longitude":      77.2,
        "sat_lon_deg":    83.0,
        "altitude_km":    0.216,
        # ITU-R P.837-7 rain quantiles
        "itu_rain": {
            "R001":   42.0,   # mm/h at 0.01 % annual exceedance
            "R01":    19.0,   # mm/h at 0.1  %
            "R1":      6.0,   # mm/h at 1    %
            "P_rain":  0.053, # fraction of year rain occurs
        },
        # ITU-R P.676 / P.618 climate
        "wv_g_m3":        12.0,
        "humidity_pct":   70,
        "v_radial_ms":    -30.0,
    },

    # ── Tokyo ──────────────────────────────────────────────────────────────
    # ITU rain zone N (humid subtropical).  Satellite: JCSAT-3A @ 110.0 E
    {
        "name":           "Tokyo",
        "eirp_dbw":       54,
        "g_rx_dbi":       48,
        "system_temp_k":  450,
        "antenna_diam_m": 2.4,
        "atm_loss_db":    0.8,
        "rain_severity":  "low",
        "latitude":       35.6,
        "longitude":      139.6,
        "sat_lon_deg":    110.0,
        "altitude_km":    0.040,
        "itu_rain": {
            "R001":   80.0,
            "R01":    42.0,
            "R1":     16.0,
            "P_rain":  0.072,
        },
        "wv_g_m3":         9.5,
        "humidity_pct":   65,
        "v_radial_ms":    -28.0,
    },

    # ── Berlin ─────────────────────────────────────────────────────────────
    # ITU rain zone E (temperate maritime).  Satellite: Astra 1 @ 19.2 E
    {
        "name":           "Berlin",
        "eirp_dbw":       50,
        "g_rx_dbi":       44,
        "system_temp_k":  520,
        "antenna_diam_m": 2.4,
        "atm_loss_db":    1.0,
        "rain_severity":  "low",
        "latitude":       52.5,
        "longitude":      13.4,
        "sat_lon_deg":    19.2,
        "altitude_km":    0.034,
        "itu_rain": {
            "R001":   28.0,
            "R01":    14.0,
            "R1":      5.5,
            "P_rain":  0.065,
        },
        "wv_g_m3":         7.0,
        "humidity_pct":   75,
        "v_radial_ms":    -25.0,
    },

    # ── Sao Paulo ──────────────────────────────────────────────────────────
    # ITU rain zone N/P (tropical convective).  Satellite: Star One C2 @ 70.0 W
    {
        "name":           "Sao Paulo",
        "eirp_dbw":       53,
        "g_rx_dbi":       46,
        "system_temp_k":  600,
        "antenna_diam_m": 2.4,
        "atm_loss_db":    1.5,
        "rain_severity":  "high",
        "latitude":       -23.5,
        "longitude":      -46.6,
        "sat_lon_deg":    -70.0,
        "altitude_km":    0.760,
        "itu_rain": {
            "R001":   95.0,
            "R01":    55.0,
            "R1":     22.0,
            "P_rain":  0.095,
        },
        "wv_g_m3":        14.5,
        "humidity_pct":   80,
        "v_radial_ms":    -32.0,
    },
]
