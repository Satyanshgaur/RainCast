"""
Satellite Link Budget Simulator — Multi-Station, Physics-Upgraded
==================================================================
Public API (consumed by app.py):
    simulate_station(gs, *, n_steps, dt_s, force_rain, seed,
                     freq_hz, eirp_offset_db, bandwidth_hz,
                     polarization, rain_rate_scale) -> StationResult
    simulate_all(...)  -> list[StationResult]

Physics models:
  1. ITU-R P.837-7  — location-specific rain statistics
  2. ITU-R P.838-3  — rain attenuation coefficients (freq/pol aware)
  3. ITU-R P.839-4  — rain height (latitude dependent)
  4. ITU-R P.618-13 — effective path length, scintillation
  5. ITU-R P.676-12 — gaseous absorption (O2 + H2O)
  6. ITU-R P.1853   — Maseng-Bakken AR(1) correlated rain time series
  7. Geometry       — GEO elevation angle from lat/lon/sat-lon
"""

import math
import random
import statistics
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from ground_stations import GROUND_STATIONS
from propogate import Propagator

# ── Physical constants ────────────────────────────────────────────────────────
C     = 2.998e8
K_B   = 1.380649e-23
R_E   = 6371.0
R_GEO = 42_164.0

# ── System defaults (overrideable per simulate_station call) ─────────────────
DEFAULT_CARRIER_FREQ_HZ = 14e9
DEFAULT_BANDWIDTH_HZ    = 36e6
DEFAULT_POLARIZATION    = "vertical"
DEFAULT_DT_S            = 60
DEFAULT_N_STEPS         = 60

# ── Packet-loss sigmoid threshold (Ku-band DVB-S2 practical floor) ───────────
# At 14 GHz with typical coding, link fails below ~10 dB Eb/N0.
# Using 10 dB as the inflection point gives a physically meaningful
# spread: stations at 5 dB show ~88 % loss; stations at 15 dB show ~2 %.
SNR_THRESHOLD_DB = 10.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  A.  Geometry                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def geo_elevation_deg(lat_deg, lon_deg, sat_lon_deg):
    lat       = math.radians(lat_deg)
    dlon      = math.radians(lon_deg - sat_lon_deg)
    cos_gamma = math.cos(lat) * math.cos(dlon)
    d         = geo_slant_range_km(lat_deg, lon_deg, sat_lon_deg)
    sin_el    = (R_GEO * cos_gamma - R_E) / d
    return math.degrees(math.asin(max(min(sin_el, 1.0), -1.0)))

def geo_slant_range_km(lat_deg, lon_deg, sat_lon_deg):
    lat       = math.radians(lat_deg)
    dlon      = math.radians(lon_deg - sat_lon_deg)
    cos_gamma = math.cos(lat) * math.cos(dlon)
    return math.sqrt(R_GEO**2 + R_E**2 - 2 * R_GEO * R_E * cos_gamma)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  B.  ITU-R P.838-3                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def itu_rain_coefficients(freq_ghz, polarization):
    table = [
        ( 1,  0.0000259, 0.9691, 0.0000308, 0.8592),
        ( 2,  0.0000847, 1.0664, 0.0000998, 0.9490),
        ( 4,  0.0001071, 1.6009, 0.0002461, 1.2476),
        ( 6,  0.007056,  1.590,  0.004115,  1.590 ),
        ( 7,  0.001915,  1.481,  0.001128,  1.457 ),
        ( 8,  0.004115,  1.590,  0.002455,  1.584 ),
        (10,  0.01217,   1.261,  0.01129,   1.3026),
        (12,  0.02386,   1.179,  0.01731,   1.2070),
        (15,  0.04481,   1.154,  0.03979,   1.1820),
        (20,  0.09164,   1.099,  0.08084,   1.0993),
        (25,  0.1571,    1.046,  0.1378,    1.0639),
        (30,  0.2403,    1.021,  0.2101,    1.0299),
        (35,  0.3374,    0.979,  0.2991,    0.9876),
        (40,  0.4743,    0.939,  0.4285,    0.9491),
    ]
    freqs  = [r[0] for r in table]
    kH_tab = [r[1] for r in table]; aH_tab = [r[2] for r in table]
    kV_tab = [r[3] for r in table]; aV_tab = [r[4] for r in table]

    def log_interp(x, xs, ys):
        if x <= xs[0]:  return ys[0]
        if x >= xs[-1]: return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i+1]:
                t = (math.log10(x)-math.log10(xs[i])) / (math.log10(xs[i+1])-math.log10(xs[i]))
                return 10**(math.log10(ys[i]) + t*(math.log10(ys[i+1])-math.log10(ys[i])))

    def lin_interp(x, xs, ys):
        if x <= xs[0]:  return ys[0]
        if x >= xs[-1]: return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i+1]:
                return ys[i] + (x-xs[i])/(xs[i+1]-xs[i]) * (ys[i+1]-ys[i])

    if polarization.lower() == "horizontal":
        return log_interp(freq_ghz, freqs, kH_tab), lin_interp(freq_ghz, freqs, aH_tab)
    return log_interp(freq_ghz, freqs, kV_tab), lin_interp(freq_ghz, freqs, aV_tab)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  C.  ITU-R P.839-4                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def itu_rain_height(lat_deg):
    a = abs(lat_deg)
    if a > 23:
        return max(5.0 - 0.075*(a-23.0), 3.0) if a < 36 else max(5.0 - 0.1*(a-36.0), 2.0)
    return 5.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  D.  ITU-R P.618-13                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def effective_path_length(elevation_deg, rain_height_km, station_altitude_km, itu_k):
    el_rad  = math.radians(max(elevation_deg, 5.0))
    h_delta = rain_height_km - station_altitude_km
    if h_delta <= 0:
        return 0.0
    L_s = h_delta / math.sin(el_rad)
    L_g = h_delta / math.tan(el_rad)
    r   = 1.0 / (1.0 + 0.78*math.sqrt(L_g*itu_k) - 0.38*(1-math.exp(-2*L_g))) \
          if elevation_deg <= 10 else 1.0
    return L_s * r

def rain_attenuation_db(rain_rate_mmh, itu_k, itu_alpha, eff_path_km):
    if rain_rate_mmh <= 0 or eff_path_km <= 0:
        return 0.0
    return itu_k * (rain_rate_mmh ** itu_alpha) * eff_path_km


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  E.  ITU-R P.1853 — Maseng-Bakken AR(1) correlated rain                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TAU_COHERENCE_S = 300.0

class CorrelatedRainProcess:
    def __init__(self, gs, dt_s, tau_c=TAU_COHERENCE_S,
                 force_rain=False, rain_rate_scale=1.0):
        p = gs["itu_rain"]
        R001, R01, P_rain = p["R001"] * rain_rate_scale, \
                            p["R01"]  * rain_rate_scale, \
                            p["P_rain"]
        # Floor R01 so log doesn't blow up if scale is tiny
        R001 = max(R001, 0.1); R01 = max(R01, 0.05)
        _z001, _z01 = 3.0902, 2.3263
        self.sigma      = (math.log(R001) - math.log(R01)) / (_z001 - _z01)
        self.mu         = math.log(R01) - _z01 * self.sigma
        self.rho        = math.exp(-dt_s / tau_c)
        self.ln_R       = self.mu
        self.force_rain = force_rain
        self.raining    = force_rain
        mean_rain_dur_s  = tau_c
        mean_clear_dur_s = tau_c * (1 - P_rain) / (P_rain + 1e-9)
        self._p_onset = 1 - math.exp(-dt_s / mean_clear_dur_s)
        self._p_clear = 1 - math.exp(-dt_s / mean_rain_dur_s)

    def step(self):
        if not self.force_rain:
            if not self.raining:
                if random.random() < self._p_onset:
                    self.raining = True; self.ln_R = self.mu
            else:
                if random.random() < self._p_clear:
                    self.raining = False
        if not self.raining:
            return 0.0
        self.ln_R = (self.rho * self.ln_R
                     + math.sqrt(1 - self.rho**2) * self.sigma * random.gauss(0,1)
                     + (1 - self.rho) * self.mu)
        return min(math.exp(self.ln_R), 150.0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  F.  ITU-R P.676-12                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def gaseous_absorption_db(freq_ghz, elevation_deg, water_vapour_g_m3):
    f = freq_ghz
    gamma_oxy = max((7.2/(f**2+0.34) + 0.62/((54-f)**1.16+0.83))
                    * (f/22.235)**2 * 1e-3, 0.0078)
    gamma_wv  = (0.050 + 0.0021*water_vapour_g_m3
                 + 3.6  / ((f-22.235)**2 + 8.5)
                 + 10.6 / ((f-183.31)**2 + 9.0)
                 + 8.9  / ((f-325.153)**2 + 26.3)) * water_vapour_g_m3 * f**2 * 1e-4
    return (gamma_oxy + gamma_wv) * 10.0 / math.sin(math.radians(max(elevation_deg, 5.0)))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  G.  ITU-R P.618-13 §2.4 — scintillation                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def scintillation_sigma_db(freq_ghz, elevation_deg, antenna_diam_m, humidity_pct):
    el_rad    = math.radians(max(elevation_deg, 5.0))
    Nwet      = 0.75 * humidity_pct
    sigma_ref = 0.5509 * Nwet * math.sqrt(1e-3) / (math.sin(el_rad) ** 1.2)
    eta       = 0.5
    D_eff     = math.sqrt(eta) * antenna_diam_m
    x         = 1.22 * D_eff**2 * (freq_ghz / 300.0)
    g_x       = math.sqrt(max(3.86*(x**2+1)**0.116
                               * math.cos(math.atan(x)*11.0/6.0)
                               - 7.08*x**(5.0/6.0), 1e-6))
    return sigma_ref * g_x * 1e-3


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  H.  Link budget utilities                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def fspl_db(freq_hz, distance_km):
    return 92.45 + 20*math.log10(freq_hz/1e9) + 20*math.log10(distance_km)

def noise_power_dbw(T_sys_K, B_hz):
    return 10 * math.log10(K_B * T_sys_K * B_hz)

def doppler_shift_hz(v_radial_ms, freq_hz):
    return (v_radial_ms / C) * freq_hz

def packet_loss_from_snr(snr_db, threshold_db=SNR_THRESHOLD_DB):
    """
    Sigmoid packet loss curve centred at threshold_db.
    At threshold_db      → 50 % loss
    At threshold_db+7 dB → ~4  % loss  (strong link)
    At threshold_db-7 dB → ~96 % loss  (failed link)
    """
    return 1.0 / (1.0 + math.exp(0.8 * (snr_db - threshold_db)))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  I.  StationResult                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class StationResult:
    # identity
    name:          str
    # geometry (initial or mean values)
    elevation:     float
    slant_km:      float
    doppler_hz:    float
    # propagation constants
    path_loss:     float
    gas_loss:      float
    rain_height:   float
    eff_path:      float
    itu_k:         float
    itu_alpha:     float
    scint_sig:     float
    noise_floor:   float   # dBW — so UI can show noise budget
    # time series
    snr_series:        list
    rain_series:       list
    rain_db_series:    list
    scint_series:      list
    pkt_loss_series:   list
    elevation_series:  list   # New
    slant_range_series: list  # New
    doppler_series:    list   # New
    # summary statistics
    snr_mean:        float
    snr_min:         float
    snr_std:         float
    snr_p10:         float
    rain_fraction:   float
    avg_rain_db:     float
    avg_pkt_loss:    float
    outage_fraction: float   # fraction of steps with SNR < SNR_THRESHOLD_DB


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  J.  simulate_station() — public API                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def simulate_station(gs: dict, *,
                     n_steps:         int   = DEFAULT_N_STEPS,
                     dt_s:            float = DEFAULT_DT_S,
                     start_time:      datetime | None = None,
                     force_rain:      bool  = False,
                     seed:            int | None = None,
                     # ── overrideable physical knobs ──────────────────────
                     freq_hz:         float = DEFAULT_CARRIER_FREQ_HZ,
                     eirp_offset_db:  float = 0.0,   # +/- dB on station EIRP
                     bandwidth_hz:    float = DEFAULT_BANDWIDTH_HZ,
                     polarization:    str   = DEFAULT_POLARIZATION,
                     rain_rate_scale: float = 1.0,   # multiplier on ITU rain rates
                     ) -> StationResult:
    """
    Run the full ITU-R physics simulation for one ground station.
    Now supports dynamic SGP4 propagation if 'norad_id' or 'sat_name' in gs.
    """
    if seed is not None:
        random.seed(seed)

    freq_ghz = freq_hz / 1e9
    lat, lon = gs["latitude"], gs["longitude"]
    alt_km = gs["altitude_km"]
    
    # ── Initialization ────────────────────────────────────────────────────────
    propagator = Propagator()
    curr_time = start_time or datetime.now(timezone.utc)
    
    # Initial geometry (or static if no SGP4)
    sat_id = gs.get("norad_id") or gs.get("sat_name")
    if sat_id:
        geo = propagator.get_geometry(sat_id, curr_time, lat, lon, alt_km)
        if geo:
            elevation = geo.elevation_deg
            slant_km  = geo.slant_range_km
            dop_hz    = doppler_shift_hz(geo.radial_velocity_ms, freq_hz)
        else:
            # Fallback to static if propagation fails
            elevation = geo_elevation_deg(lat, lon, gs["sat_lon_deg"])
            slant_km  = geo_slant_range_km(lat, lon, gs["sat_lon_deg"])
            dop_hz    = doppler_shift_hz(gs["v_radial_ms"], freq_hz)
    else:
        elevation = geo_elevation_deg(lat, lon, gs["sat_lon_deg"])
        slant_km  = geo_slant_range_km(lat, lon, gs["sat_lon_deg"])
        dop_hz    = doppler_shift_hz(gs["v_radial_ms"], freq_hz)

    # ── Fixed propagation terms (non-geometric) ──────────────────────────────
    noise_dbw    = noise_power_dbw(gs["system_temp_k"], bandwidth_hz)
    rain_h       = itu_rain_height(lat)
    itu_k, itu_a = itu_rain_coefficients(freq_ghz, polarization)
    eirp_eff     = gs["eirp_dbw"] + eirp_offset_db

    # ── Rain process ──────────────────────────────────────────────────────────
    rain_proc = CorrelatedRainProcess(gs, dt_s=dt_s, force_rain=force_rain,
                                      rain_rate_scale=rain_rate_scale)

    snr_s = []; rain_s = []; rain_db_s = []; scint_s = []; pkt_s = []
    el_s = []; slant_s = []; dop_s = []

    for _ in range(n_steps):
        # Update Dynamic Geometry
        if sat_id:
            geo = propagator.get_geometry(sat_id, curr_time, lat, lon, alt_km)
            if geo:
                elevation = geo.elevation_deg
                slant_km  = geo.slant_range_km
                dop_hz    = doppler_shift_hz(geo.radial_velocity_ms, freq_hz)

        # Recompute geometry-dependent terms
        path_loss    = fspl_db(freq_hz, slant_km)
        gas_loss     = gaseous_absorption_db(freq_ghz, elevation, gs["wv_g_m3"])
        eff_path     = effective_path_length(elevation, rain_h, alt_km, itu_k)
        scint_sig    = scintillation_sigma_db(freq_ghz, elevation,
                                           gs["antenna_diam_m"], gs["humidity_pct"])

        # Rain and Scintillation steps
        rain_rate = rain_proc.step()
        rain_db   = rain_attenuation_db(rain_rate, itu_k, itu_a, eff_path)
        scint_db  = random.gauss(0.0, scint_sig)

        snr = (eirp_eff
               - path_loss
               - gas_loss
               - rain_db
               - scint_db
               + gs["g_rx_dbi"]
               - noise_dbw)

        snr_s.append(snr);           rain_s.append(rain_rate)
        rain_db_s.append(rain_db);   scint_s.append(scint_db)
        pkt_s.append(packet_loss_from_snr(snr))
        el_s.append(elevation);      slant_s.append(slant_km); dop_s.append(dop_hz)
        
        curr_time += timedelta(seconds=dt_s)

    rainy_dbs  = [db for db in rain_db_s if db > 0]
    sorted_snr = sorted(snr_s)
    p10_idx    = max(0, int(0.10 * n_steps) - 1)

    return StationResult(
        name=gs["name"], elevation=el_s[0], slant_km=slant_s[0], doppler_hz=dop_s[0],
        path_loss=fspl_db(freq_hz, slant_s[0]), # report initial
        gas_loss=gaseous_absorption_db(freq_ghz, el_s[0], gs["wv_g_m3"]),
        rain_height=rain_h,
        eff_path=effective_path_length(el_s[0], rain_h, alt_km, itu_k),
        itu_k=itu_k, itu_alpha=itu_a, scint_sig=scintillation_sigma_db(freq_ghz, el_s[0], gs["antenna_diam_m"], gs["humidity_pct"]),
        noise_floor=noise_dbw,
        snr_series=snr_s, rain_series=rain_s, rain_db_series=rain_db_s,
        scint_series=scint_s, pkt_loss_series=pkt_s,
        elevation_series=el_s, slant_range_series=slant_s, doppler_series=dop_s,
        snr_mean=statistics.mean(snr_s),
        snr_min=min(snr_s),
        snr_std=statistics.stdev(snr_s) if len(snr_s) > 1 else 0.0,
        snr_p10=sorted_snr[p10_idx],
        rain_fraction=sum(1 for r in rain_s if r > 0) / n_steps,
        avg_rain_db=statistics.mean(rainy_dbs) if rainy_dbs else 0.0,
        avg_pkt_loss=statistics.mean(pkt_s),
        outage_fraction=sum(1 for s in snr_s if s < SNR_THRESHOLD_DB) / n_steps,
    )


def simulate_all(n_steps=DEFAULT_N_STEPS, dt_s=DEFAULT_DT_S,
                 force_rain=False, **kwargs) -> list:
    return [simulate_station(gs, n_steps=n_steps, dt_s=dt_s,
                             force_rain=force_rain,
                             seed=hash(gs["name"]) % 100_000, **kwargs)
            for gs in GROUND_STATIONS]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  K.  CLI entry point                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _print_station(gs, r):
    W = 92; p = gs["itu_rain"]
    sat_info = f"NORAD:{gs['norad_id']}" if "norad_id" in gs else f"SatLon:{gs['sat_lon_deg']:+.1f}"
    print("=" * W)
    print(f"  {r.name}  ({gs['latitude']:+.1f}, {gs['longitude']:+.1f})  "
          f"{sat_info}  El:{r.elevation:.1f}°  Range:{r.slant_km:.0f}km")
    print(f"  FSPL:{r.path_loss:.2f}dB  Gas:{r.gas_loss:.3f}dB  "
          f"RainH:{r.rain_height:.2f}km  EffPath:{r.eff_path:.2f}km  "
          f"Noise:{r.noise_floor:.1f}dBW")
    print(f"  k={r.itu_k:.5f} α={r.itu_alpha:.4f}  Scint σ={r.scint_sig:.4f}dB  "
          f"Doppler:{r.doppler_hz:+.0f}Hz")
    print(f"  P.837: R001={p['R001']} R01={p['R01']} R1={p['R1']} "
          f"P_rain={p['P_rain']*100:.1f}%")
    print("-" * W)
    print(f"  SNR mean={r.snr_mean:.2f} min={r.snr_min:.2f} std={r.snr_std:.2f} "
          f"p10={r.snr_p10:.2f} dB  |  rain={r.rain_fraction*100:.0f}%  "
          f"outage={r.outage_fraction*100:.0f}%  pkt_loss={r.avg_pkt_loss:.4f}")
    print()

if __name__ == "__main__":
    import sys
    force = "--rain" in sys.argv
    print(f"\n14 GHz Ku-band  |  {'RAIN forced' if force else 'Clear/probabilistic'}\n")
    gs_map = {gs["name"]: gs for gs in GROUND_STATIONS}
    results = simulate_all(force_rain=force)
    for r in results:
        _print_station(gs_map[r.name], r)
