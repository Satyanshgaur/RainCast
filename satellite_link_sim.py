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
import numpy as np
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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
SNR_THRESHOLD_DB = 10.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  A.  Geometry                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def geo_elevation_deg(lat_deg, lon_deg, sat_lon_deg):
    lat       = np.radians(lat_deg)
    dlon      = np.radians(lon_deg - sat_lon_deg)
    cos_gamma = np.cos(lat) * np.cos(dlon)
    d         = geo_slant_range_km(lat_deg, lon_deg, sat_lon_deg)
    sin_el    = (R_GEO * cos_gamma - R_E) / d
    return np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))

def geo_slant_range_km(lat_deg, lon_deg, sat_lon_deg):
    lat       = np.radians(lat_deg)
    dlon      = np.radians(lon_deg - sat_lon_deg)
    cos_gamma = np.cos(lat) * np.cos(dlon)
    return np.sqrt(R_GEO**2 + R_E**2 - 2 * R_GEO * R_E * cos_gamma)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  B.  ITU-R P.838-3                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def itu_rain_coefficients(freq_ghz, polarization):
    table = np.array([
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
    ])
    freqs  = table[:, 0]
    kH_tab = table[:, 1]; aH_tab = table[:, 2]
    kV_tab = table[:, 3]; aV_tab = table[:, 4]

    if polarization.lower() == "horizontal":
        k = 10**np.interp(np.log10(freq_ghz), np.log10(freqs), np.log10(kH_tab))
        alpha = np.interp(freq_ghz, freqs, aH_tab)
    else:
        k = 10**np.interp(np.log10(freq_ghz), np.log10(freqs), np.log10(kV_tab))
        alpha = np.interp(freq_ghz, freqs, aV_tab)
    return k, alpha


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  C.  ITU-R P.839-4                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def itu_rain_height(lat_deg):
    a = np.abs(lat_deg)
    h = np.full_like(a, 5.0, dtype=float)
    mask1 = (a > 23) & (a < 36)
    h[mask1] = np.maximum(5.0 - 0.075 * (a[mask1] - 23.0), 3.0)
    mask2 = (a >= 36)
    h[mask2] = np.maximum(5.0 - 0.1 * (a[mask2] - 36.0), 2.0)
    return h


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  D.  ITU-R P.618-13                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def effective_path_length(elevation_deg, rain_height_km, station_altitude_km, itu_k):
    el_rad  = np.radians(np.maximum(elevation_deg, 5.0))
    h_delta = rain_height_km - station_altitude_km
    L_s = np.where(h_delta > 0, h_delta / np.sin(el_rad), 0.0)
    L_g = np.where(h_delta > 0, h_delta / np.tan(el_rad), 0.0)
    r = np.where(elevation_deg <= 10,
                 1.0 / (1.0 + 0.78 * np.sqrt(L_g * itu_k) - 0.38 * (1 - np.exp(-2 * L_g))),
                 1.0)
    return L_s * r

def rain_attenuation_db(rain_rate_mmh, itu_k, itu_alpha, eff_path_km):
    return np.where((rain_rate_mmh > 0) & (eff_path_km > 0),
                    itu_k * (rain_rate_mmh ** itu_alpha) * eff_path_km,
                    0.0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  E.  ITU-R P.1853 — Maseng-Bakken AR(1) correlated rain                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TAU_COHERENCE_S = 300.0

class CorrelatedRainProcess:
    def __init__(self, gs, dt_s, tau_c=TAU_COHERENCE_S,
                 force_rain=False, rain_rate_scale=1.0):
        if isinstance(gs, dict):
            gs = [gs]
        
        self.n_stations = len(gs)
        self.dt_s = dt_s
        self.force_rain = force_rain
        
        self.sigma = np.zeros(self.n_stations)
        self.mu = np.zeros(self.n_stations)
        self.rho = np.zeros(self.n_stations)
        self._p_onset = np.zeros(self.n_stations)
        self._p_clear = np.zeros(self.n_stations)
        
        for i, station in enumerate(gs):
            p = station["itu_rain"]
            R001 = max(p["R001"] * rain_rate_scale, 0.1)
            R01  = max(p["R01"]  * rain_rate_scale, 0.05)
            P_rain = p["P_rain"]
            
            _z001, _z01 = 3.0902, 2.3263
            self.sigma[i] = (np.log(R001) - np.log(R01)) / (_z001 - _z01)
            self.mu[i]    = np.log(R01) - _z01 * self.sigma[i]
            self.rho[i]   = np.exp(-dt_s / tau_c)
            
            mean_rain_dur_s  = tau_c
            mean_clear_dur_s = tau_c * (1 - P_rain) / (P_rain + 1e-9)
            self._p_onset[i] = 1 - np.exp(-dt_s / mean_clear_dur_s)
            self._p_clear[i] = 1 - np.exp(-dt_s / mean_rain_dur_s)

        self.ln_R = self.mu.copy()
        self.raining = np.full(self.n_stations, force_rain)

    def step(self):
        if not self.force_rain:
            onset_mask = (~self.raining) & (np.random.rand(self.n_stations) < self._p_onset)
            self.raining[onset_mask] = True
            self.ln_R[onset_mask] = self.mu[onset_mask]
            
            clear_mask = self.raining & (np.random.rand(self.n_stations) < self._p_clear)
            self.raining[clear_mask] = False
            
        noise = np.random.normal(0, 1, self.n_stations)
        self.ln_R = (self.rho * self.ln_R
                     + np.sqrt(1 - self.rho**2) * self.sigma * noise
                     + (1 - self.rho) * self.mu)
        
        rates = np.where(self.raining, np.minimum(np.exp(self.ln_R), 150.0), 0.0)
        return rates


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  F.  ITU-R P.676-12                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def gaseous_absorption_db(freq_ghz, elevation_deg, water_vapour_g_m3):
    f = freq_ghz
    gamma_oxy = np.maximum((7.2/(f**2+0.34) + 0.62/((54-f)**1.16+0.83))
                    * (f/22.235)**2 * 1e-3, 0.0078)
    gamma_wv  = (0.050 + 0.0021*water_vapour_g_m3
                 + 3.6  / ((f-22.235)**2 + 8.5)
                 + 10.6 / ((f-183.31)**2 + 9.0)
                 + 8.9  / ((f-325.153)**2 + 26.3)) * water_vapour_g_m3 * f**2 * 1e-4
    return (gamma_oxy + gamma_wv) * 10.0 / np.sin(np.radians(np.maximum(elevation_deg, 5.0)))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  G.  ITU-R P.618-13 §2.4 — scintillation                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def scintillation_sigma_db(freq_ghz, elevation_deg, antenna_diam_m, humidity_pct):
    el_rad    = np.radians(np.maximum(elevation_deg, 5.0))
    Nwet      = 0.75 * humidity_pct
    sigma_ref = 0.5509 * Nwet * np.sqrt(1e-3) / (np.sin(el_rad) ** 1.2)
    eta       = 0.5
    D_eff     = np.sqrt(eta) * antenna_diam_m
    x         = 1.22 * D_eff**2 * (freq_ghz / 300.0)
    g_x       = np.sqrt(np.maximum(3.86*(x**2+1)**0.116
                               * np.cos(np.arctan(x)*11.0/6.0)
                               - 7.08*x**(5.0/6.0), 1e-6))
    return sigma_ref * g_x * 1e-3


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  H.  Link budget utilities                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def fspl_db(freq_hz, distance_km):
    return 92.45 + 20*np.log10(freq_hz/1e9) + 20*np.log10(distance_km)

def noise_power_dbw(T_sys_K, B_hz):
    return 10 * np.log10(K_B * T_sys_K * B_hz)

def doppler_shift_hz(v_radial_ms, freq_hz):
    return (v_radial_ms / C) * freq_hz

def packet_loss_from_snr(snr_db, threshold_db=SNR_THRESHOLD_DB):
    return 1.0 / (1.0 + np.exp(0.8 * (snr_db - threshold_db)))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  I.  StationResult                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class StationResult:
    name:          str
    elevation:     float
    slant_km:      float
    doppler_hz:    float
    path_loss:     float
    gas_loss:      float
    rain_height:   float
    eff_path:      float
    itu_k:         float
    itu_alpha:     float
    scint_sig:     float
    noise_floor:   float
    snr_series:        list
    rain_series:       list
    rain_db_series:    list
    scint_series:      list
    pkt_loss_series:   list
    elevation_series:  list
    slant_range_series: list
    doppler_series:    list
    snr_mean:        float
    snr_min:         float
    snr_std:         float
    snr_p10:         float
    rain_fraction:   float
    avg_rain_db:     float
    avg_pkt_loss:    float
    outage_fraction: float


def simulate_all_batched(ground_stations: list[dict],
                         n_steps:         int   = DEFAULT_N_STEPS,
                         dt_s:            float = DEFAULT_DT_S,
                         start_time:      datetime | None = None,
                         force_rain:      bool  = False,
                         seed:            int | None = None,
                         freq_hz:         float = DEFAULT_CARRIER_FREQ_HZ,
                         eirp_offset_db:  float = 0.0,
                         bandwidth_hz:    float = DEFAULT_BANDWIDTH_HZ,
                         polarization:    str   = DEFAULT_POLARIZATION,
                         rain_rate_scale: float = 1.0,
                         ) -> list[StationResult]:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n_stations = len(ground_stations)
    curr_time = start_time or datetime.now(timezone.utc)
    times = [curr_time + timedelta(seconds=i*dt_s) for i in range(n_steps)]
    freq_ghz = freq_hz / 1e9
    
    # 1. Batched Geometry
    propagator = Propagator()
    el_matrix    = np.zeros((n_stations, n_steps))
    slant_matrix = np.zeros((n_stations, n_steps))
    dop_matrix   = np.zeros((n_stations, n_steps))
    
    for i, gs in enumerate(ground_stations):
        sat_id = gs.get("norad_id") or gs.get("sat_name")
        if sat_id:
            geo = propagator.get_geometry_batch(sat_id, times, gs["latitude"], gs["longitude"], gs["altitude_km"])
            if geo:
                el_matrix[i]    = geo.elevation_deg
                slant_matrix[i] = geo.slant_range_km
                dop_matrix[i]   = doppler_shift_hz(geo.radial_velocity_ms, freq_hz)
            else:
                el_matrix[i]    = geo_elevation_deg(gs["latitude"], gs["longitude"], gs["sat_lon_deg"])
                slant_matrix[i] = geo_slant_range_km(gs["latitude"], gs["longitude"], gs["sat_lon_deg"])
                dop_matrix[i]   = doppler_shift_hz(gs["v_radial_ms"], freq_hz)
        else:
            el_matrix[i]    = geo_elevation_deg(gs["latitude"], gs["longitude"], gs["sat_lon_deg"])
            slant_matrix[i] = geo_slant_range_km(gs["latitude"], gs["longitude"], gs["sat_lon_deg"])
            dop_matrix[i]   = doppler_shift_hz(gs["v_radial_ms"], freq_hz)

    # 2. Vectorized ITU Constants & Noise
    lats = np.array([gs["latitude"] for gs in ground_stations])
    alts = np.array([gs["altitude_km"] for gs in ground_stations])
    wvs  = np.array([gs["wv_g_m3"] for gs in ground_stations])
    hums = np.array([gs["humidity_pct"] for gs in ground_stations])
    diams = np.array([gs["antenna_diam_m"] for gs in ground_stations])
    g_rxs = np.array([gs["g_rx_dbi"] for gs in ground_stations])
    temps = np.array([gs["system_temp_k"] for gs in ground_stations])
    eirps = np.array([gs["eirp_dbw"] + eirp_offset_db for gs in ground_stations])
    
    rain_h = itu_rain_height(lats)
    itu_k, itu_a = itu_rain_coefficients(freq_ghz, polarization)
    noise_dbw = noise_power_dbw(temps, bandwidth_hz)

    # 3. Time-Series Calculations
    path_loss_matrix = fspl_db(freq_hz, slant_matrix)
    gas_loss_matrix  = gaseous_absorption_db(freq_ghz, el_matrix, wvs[:, None])
    eff_path_matrix  = effective_path_length(el_matrix, rain_h[:, None], alts[:, None], itu_k)
    scint_sig_matrix = scintillation_sigma_db(freq_ghz, el_matrix, diams[:, None], hums[:, None])
    
    rain_proc = CorrelatedRainProcess(ground_stations, dt_s=dt_s, force_rain=force_rain, 
                                      rain_rate_scale=rain_rate_scale)
    rain_rate_matrix = np.zeros((n_stations, n_steps))
    scint_db_matrix  = np.random.normal(0, scint_sig_matrix)
    
    for t in range(n_steps):
        rain_rate_matrix[:, t] = rain_proc.step()
        
    rain_db_matrix = rain_attenuation_db(rain_rate_matrix, itu_k, itu_a, eff_path_matrix)
    
    snr_matrix = (eirps[:, None]
                  - path_loss_matrix
                  - gas_loss_matrix
                  - rain_db_matrix
                  - scint_db_matrix
                  + g_rxs[:, None]
                  - noise_dbw[:, None])
    
    pkt_loss_matrix = packet_loss_from_snr(snr_matrix)
    
    results = []
    for i, gs in enumerate(ground_stations):
        snr_s = snr_matrix[i].tolist()
        rain_s = rain_rate_matrix[i].tolist()
        rain_db_s = rain_db_matrix[i].tolist()
        scint_s = scint_db_matrix[i].tolist()
        pkt_s = pkt_loss_matrix[i].tolist()
        el_s = el_matrix[i].tolist()
        slant_s = slant_matrix[i].tolist()
        dop_s = dop_matrix[i].tolist()
        
        rainy_dbs = [db for db in rain_db_s if db > 0]
        sorted_snr = sorted(snr_s)
        p10_idx = max(0, int(0.10 * n_steps) - 1)
        
        results.append(StationResult(
            name=gs["name"], elevation=el_s[0], slant_km=slant_s[0], doppler_hz=dop_s[0],
            path_loss=path_loss_matrix[i, 0],
            gas_loss=gas_loss_matrix[i, 0],
            rain_height=rain_h[i],
            eff_path=eff_path_matrix[i, 0],
            itu_k=itu_k, itu_alpha=itu_a, scint_sig=scint_sig_matrix[i, 0],
            noise_floor=noise_dbw[i],
            snr_series=snr_s, rain_series=rain_s, rain_db_series=rain_db_s,
            scint_series=scint_s, pkt_loss_series=pkt_s,
            elevation_series=el_s, slant_range_series=slant_s, doppler_series=dop_s,
            snr_mean=float(np.mean(snr_s)),
            snr_min=float(np.min(snr_s)),
            snr_std=float(np.std(snr_s, ddof=1)) if len(snr_s) > 1 else 0.0,
            snr_p10=float(sorted_snr[p10_idx]),
            rain_fraction=float(np.sum(np.array(rain_s) > 0) / n_steps),
            avg_rain_db=float(np.mean(rainy_dbs)) if rainy_dbs else 0.0,
            avg_pkt_loss=float(np.mean(pkt_s)),
            outage_fraction=float(np.mean(pkt_s)),
        ))
    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  J.  Concurrency & Parallelism                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

async def simulate_all_concurrent(ground_stations: List[Dict], **kwargs) -> List[StationResult]:
    """
    Concurrent station simulation using asyncio for overlapping propagation.
    """
    propagator = Propagator()
    n_steps = kwargs.get("n_steps", DEFAULT_N_STEPS)
    dt_s = kwargs.get("dt_s", DEFAULT_DT_S)
    start_time = kwargs.get("start_time") or datetime.now(timezone.utc)
    times = [start_time + timedelta(seconds=i*dt_s) for i in range(n_steps)]
    
    # Run all propagation tasks concurrently
    tasks = []
    for gs in ground_stations:
        sat_id = gs.get("norad_id") or gs.get("sat_name")
        if sat_id:
            tasks.append(propagator.get_geometry_batch_async(
                sat_id, times, gs["latitude"], gs["longitude"], gs["altitude_km"]
            ))
        else:
            tasks.append(asyncio.sleep(0, result=None))
            
    geometries = await asyncio.gather(*tasks)
    
    # For now, we still use simulate_all_batched for the link budget 
    # as it's highly optimized with NumPy. This demonstrates overlapping I/O/CPU.
    return simulate_all_batched(ground_stations, **kwargs)


def run_monte_carlo(n_iterations: int, ground_stations: List[Dict], **kwargs) -> List[List[StationResult]]:
    """
    Monte Carlo parallelism using multiprocessing.
    Runs multiple full simulation iterations in parallel.
    """
    # Create seeds for each iteration to ensure different rain patterns
    base_seed = kwargs.pop("seed", 42) or 42
    seeds = [base_seed + i for i in range(n_iterations)]
    
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(simulate_all_batched, ground_stations, seed=s, **kwargs)
            for s in seeds
        ]
        results = [f.result() for f in futures]
    return results


def simulate_station(gs: dict, **kwargs) -> StationResult:
    results = simulate_all_batched([gs], **kwargs)
    return results[0]


def simulate_all(n_steps=DEFAULT_N_STEPS, dt_s=DEFAULT_DT_S,
                 force_rain=False, **kwargs) -> list:
    return simulate_all_batched(GROUND_STATIONS, n_steps=n_steps, dt_s=dt_s, force_rain=force_rain, **kwargs)



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
