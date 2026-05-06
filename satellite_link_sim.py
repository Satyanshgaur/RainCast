"""
Satellite Link Budget Simulator — Multi-Station, Physics-Upgraded
==================================================================
Ground station data is imported from ground_stations.py (single source of
truth shared with app.py).

Public API (used by app.py):
    simulate_station(gs, n_steps, dt_s, force_rain) -> StationResult
    simulate_all(n_steps, dt_s, force_rain)          -> list[StationResult]

StationResult carries both the full per-step time series AND pre-computed
summary statistics that app.py feeds directly to the ML feature vector.

Physics models:
  1. ITU-R P.837-7  — location-specific rain statistics (from ground_stations)
  2. ITU-R P.838-3  — specific rain attenuation coefficients (freq/pol aware)
  3. ITU-R P.839-4  — rain height (latitude dependent)
  4. ITU-R P.618-13 — effective path length, scintillation model
  5. ITU-R P.676-12 — gaseous absorption (O2 + H2O, humidity aware)
  6. ITU-R P.1853   — Maseng-Bakken AR(1) correlated rain time series
  7. Geometry       — GEO elevation angle computed from lat/lon/sat-lon
"""

import math
import random
import statistics
from dataclasses import dataclass, field

from ground_stations import GROUND_STATIONS

# ── Physical constants ────────────────────────────────────────────────────────
C     = 2.998e8        # speed of light, m/s
K_B   = 1.380649e-23   # Boltzmann constant, J/K
R_E   = 6371.0         # Earth mean radius, km
R_GEO = 42_164.0       # GEO orbital radius from Earth centre, km

# ── System-wide RF parameters ─────────────────────────────────────────────────
CARRIER_FREQ_HZ = 14e9        # Ku-band uplink
BANDWIDTH_HZ    = 36e6
POLARIZATION    = "vertical"  # "vertical" | "horizontal"

# ── Default simulation timing ─────────────────────────────────────────────────
DEFAULT_DT_S    = 60    # 1-minute time step
DEFAULT_N_STEPS = 60    # 60 minutes


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  A.  Geometry                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def geo_elevation_deg(lat_deg: float, lon_deg: float, sat_lon_deg: float) -> float:
    """Elevation angle to GEO satellite (ITU-R S.1066)."""
    lat       = math.radians(lat_deg)
    dlon      = math.radians(lon_deg - sat_lon_deg)
    cos_gamma = math.cos(lat) * math.cos(dlon)
    sin_el    = (cos_gamma - R_E / R_GEO) / math.sqrt(1 - cos_gamma**2 + 1e-12)
    return math.degrees(math.asin(max(min(sin_el, 1.0), -1.0)))


def geo_slant_range_km(lat_deg: float, lon_deg: float, sat_lon_deg: float) -> float:
    """Slant range to GEO satellite (km)."""
    lat       = math.radians(lat_deg)
    dlon      = math.radians(lon_deg - sat_lon_deg)
    cos_gamma = math.cos(lat) * math.cos(dlon)
    return math.sqrt(R_GEO**2 + R_E**2 - 2 * R_GEO * R_E * cos_gamma)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  B.  ITU-R P.838-3 — rain attenuation coefficients                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def itu_rain_coefficients(freq_ghz: float, polarization: str) -> tuple[float, float]:
    """Return (k, alpha) from ITU-R P.838-3 Table 1."""
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
                t = (math.log10(x) - math.log10(xs[i])) / \
                    (math.log10(xs[i+1]) - math.log10(xs[i]))
                return 10 ** (math.log10(ys[i]) + t*(math.log10(ys[i+1]) - math.log10(ys[i])))

    def lin_interp(x, xs, ys):
        if x <= xs[0]:  return ys[0]
        if x >= xs[-1]: return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i+1]:
                t = (x - xs[i]) / (xs[i+1] - xs[i])
                return ys[i] + t * (ys[i+1] - ys[i])

    if polarization.lower() == "horizontal":
        return log_interp(freq_ghz, freqs, kH_tab), lin_interp(freq_ghz, freqs, aH_tab)
    return log_interp(freq_ghz, freqs, kV_tab), lin_interp(freq_ghz, freqs, aV_tab)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  C.  ITU-R P.839-4 — rain height                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def itu_rain_height(lat_deg: float) -> float:
    """Mean rain height above MSL (km) — ITU-R P.839-4."""
    a = abs(lat_deg)
    if a > 23:
        return max(5.0 - 0.075 * (a - 23.0), 3.0) if a < 36 else max(5.0 - 0.1 * (a - 36.0), 2.0)
    return 5.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  D.  ITU-R P.618-13 — effective rain path & attenuation                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def effective_path_length(elevation_deg: float, rain_height_km: float,
                           station_altitude_km: float, itu_k: float) -> float:
    """Effective slant path through rain layer (km) — ITU-R P.618-13 §2.2.1."""
    el_rad  = math.radians(max(elevation_deg, 5.0))
    h_delta = rain_height_km - station_altitude_km
    if h_delta <= 0:
        return 0.0
    L_s = h_delta / math.sin(el_rad)
    L_g = h_delta / math.tan(el_rad)
    r   = 1.0 / (1.0 + 0.78 * math.sqrt(L_g * itu_k) - 0.38 * (1 - math.exp(-2 * L_g))) \
          if elevation_deg <= 10 else 1.0
    return L_s * r


def rain_attenuation_db(rain_rate_mmh: float, itu_k: float, itu_alpha: float,
                         eff_path_km: float) -> float:
    """A = k * R^alpha * L_eff  [dB]  (ITU-R P.838-3)."""
    if rain_rate_mmh <= 0 or eff_path_km <= 0:
        return 0.0
    return itu_k * (rain_rate_mmh ** itu_alpha) * eff_path_km


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  E.  ITU-R P.1853 — Maseng-Bakken AR(1) correlated rain process        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TAU_COHERENCE_S = 300.0   # rain-cell coherence time ~5 min

class CorrelatedRainProcess:
    """
    First-order log-normal AR(1) rain time series anchored to station's
    ITU-R P.837-7 quantiles.  force_rain=True locks the process into its
    raining state for the entire run (used by app.py "Rain" mode).
    """
    def __init__(self, gs: dict, dt_s: float,
                 tau_c: float = TAU_COHERENCE_S,
                 force_rain: bool = False):
        p = gs["itu_rain"]
        R001, R01, P_rain = p["R001"], p["R01"], p["P_rain"]
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

    def step(self) -> float:
        """Return instantaneous rain rate (mm/h) for this time step."""
        if not self.force_rain:
            if not self.raining:
                if random.random() < self._p_onset:
                    self.raining = True
                    self.ln_R = self.mu
            else:
                if random.random() < self._p_clear:
                    self.raining = False

        if not self.raining:
            return 0.0

        self.ln_R = (self.rho * self.ln_R
                     + math.sqrt(1 - self.rho**2) * self.sigma * random.gauss(0.0, 1.0)
                     + (1 - self.rho) * self.mu)
        return min(math.exp(self.ln_R), 100.0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  F.  ITU-R P.676-12 — gaseous absorption                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def gaseous_absorption_db(freq_ghz: float, elevation_deg: float,
                           water_vapour_g_m3: float) -> float:
    """Simplified ITU-R P.676 slant-path attenuation (O2 + H2O)."""
    f = freq_ghz
    gamma_oxy = max((7.2 / (f**2 + 0.34) + 0.62 / ((54 - f)**1.16 + 0.83))
                    * (f / 22.235)**2 * 1e-3, 0.0078)
    gamma_wv  = (0.050 + 0.0021 * water_vapour_g_m3
                 + 3.6  / ((f - 22.235)**2 + 8.5)
                 + 10.6 / ((f - 183.31)**2 + 9.0)
                 + 8.9  / ((f - 325.153)**2 + 26.3)) * water_vapour_g_m3 * f**2 * 1e-4
    zenith = (gamma_oxy + gamma_wv) * 10.0
    return zenith / math.sin(math.radians(max(elevation_deg, 5.0)))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  G.  ITU-R P.618-13 §2.4 — tropospheric scintillation                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def scintillation_sigma_db(freq_ghz: float, elevation_deg: float,
                            antenna_diam_m: float, humidity_pct: float) -> float:
    """Standard deviation of scintillation fade (dB)."""
    el_rad    = math.radians(max(elevation_deg, 5.0))
    Nwet      = 0.75 * humidity_pct
    sigma_ref = 0.5509 * Nwet * math.sqrt(1e-3) / (math.sin(el_rad) ** 1.2)
    eta       = 0.5
    D_eff     = math.sqrt(eta) * antenna_diam_m
    x         = 1.22 * D_eff**2 * (freq_ghz / 300.0)
    g_x       = math.sqrt(max(3.86 * (x**2 + 1)**0.116
                               * math.cos(math.atan(x) * 11.0 / 6.0)
                               - 7.08 * x**(5.0 / 6.0), 1e-6))
    return sigma_ref * g_x * 1e-3


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  H.  Link budget utilities                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def fspl_db(freq_hz: float, distance_km: float) -> float:
    return 92.45 + 20 * math.log10(freq_hz / 1e9) + 20 * math.log10(distance_km)

def noise_power_dbw(T_sys_K: float, B_hz: float) -> float:
    return 10 * math.log10(K_B * T_sys_K * B_hz)

def doppler_shift_hz(v_radial_ms: float, freq_hz: float) -> float:
    return (v_radial_ms / C) * freq_hz

def packet_loss_from_snr(snr_db: float) -> float:
    """Sigmoid mapping from SNR to packet loss probability."""
    return 1.0 / (1.0 + math.exp(0.8 * (snr_db - 3.0)))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  I.  StationResult — full output returned to app.py                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class StationResult:
    # ── identity ──────────────────────────────────────────────────────────
    name:          str

    # ── geometry (fixed per station) ──────────────────────────────────────
    elevation:     float   # deg
    slant_km:      float   # km
    doppler_hz:    float   # Hz

    # ── propagation constants (fixed per station) ──────────────────────────
    path_loss:     float   # dB  FSPL
    gas_loss:      float   # dB  ITU-R P.676
    rain_height:   float   # km  ITU-R P.839
    eff_path:      float   # km  ITU-R P.618
    itu_k:         float
    itu_alpha:     float
    scint_sig:     float   # dB  1-sigma scintillation

    # ── per-step time series ────────────────────────────────────────────────
    snr_series:    list    # dB,  length = n_steps
    rain_series:   list    # mm/h
    rain_db_series: list   # dB
    scint_series:  list    # dB
    pkt_loss_series: list  # [0,1]

    # ── summary statistics (derived from time series) ──────────────────────
    snr_mean:      float   # dB  — primary ML feature
    snr_min:       float   # dB  — worst-case quality indicator
    snr_std:       float   # dB  — variability / stability
    snr_p10:       float   # dB  — 10th-percentile (near-outage margin)
    rain_fraction: float   # fraction of steps where rain occurred
    avg_rain_db:   float   # dB  mean rain attenuation over rainy steps
    avg_pkt_loss:  float   # [0,1] mean packet loss over run
    outage_fraction: float # fraction of steps with SNR < 3 dB (threshold)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  J.  simulate_station() — public API consumed by app.py                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def simulate_station(gs: dict,
                     n_steps: int  = DEFAULT_N_STEPS,
                     dt_s: float   = DEFAULT_DT_S,
                     force_rain: bool = False,
                     seed: int | None = None) -> StationResult:
    """
    Run the full ITU-R physics simulation for one ground station.

    Parameters
    ----------
    gs          : ground station dict from ground_stations.py
    n_steps     : number of time steps to simulate
    dt_s        : time step in seconds
    force_rain  : if True, lock the rain process into its raining state
                  (maps to app.py "Rain" weather mode)
    seed        : random seed for reproducibility (None = random)

    Returns
    -------
    StationResult with full time series + summary statistics
    """
    if seed is not None:
        random.seed(seed)

    freq_ghz = CARRIER_FREQ_HZ / 1e9
    lat, lon = gs["latitude"], gs["longitude"]

    # ── Fixed geometric / propagation terms ──────────────────────────────────
    elevation    = geo_elevation_deg(lat, lon, gs["sat_lon_deg"])
    slant_km     = geo_slant_range_km(lat, lon, gs["sat_lon_deg"])
    path_loss    = fspl_db(CARRIER_FREQ_HZ, slant_km)
    noise_dbw    = noise_power_dbw(gs["system_temp_k"], BANDWIDTH_HZ)
    gas_loss     = gaseous_absorption_db(freq_ghz, elevation, gs["wv_g_m3"])
    rain_h       = itu_rain_height(lat)
    itu_k, itu_a = itu_rain_coefficients(freq_ghz, POLARIZATION)
    eff_path     = effective_path_length(elevation, rain_h, gs["altitude_km"], itu_k)
    scint_sig    = scintillation_sigma_db(freq_ghz, elevation,
                                           gs["antenna_diam_m"], gs["humidity_pct"])
    dop_hz       = doppler_shift_hz(gs["v_radial_ms"], CARRIER_FREQ_HZ)

    # ── Time-varying rain process ─────────────────────────────────────────────
    rain_proc = CorrelatedRainProcess(gs, dt_s=dt_s, force_rain=force_rain)

    snr_series      = []
    rain_series     = []
    rain_db_series  = []
    scint_series    = []
    pkt_loss_series = []

    for _ in range(n_steps):
        rain_rate = rain_proc.step()
        rain_db   = rain_attenuation_db(rain_rate, itu_k, itu_a, eff_path)
        scint_db  = random.gauss(0.0, scint_sig)

        snr = (gs["eirp_dbw"]
               - path_loss
               - gas_loss
               - rain_db
               - scint_db
               + gs["g_rx_dbi"]
               - noise_dbw)

        pkt_loss = packet_loss_from_snr(snr)

        snr_series.append(snr)
        rain_series.append(rain_rate)
        rain_db_series.append(rain_db)
        scint_series.append(scint_db)
        pkt_loss_series.append(pkt_loss)

    # ── Derived summary statistics ────────────────────────────────────────────
    rainy_steps   = [db for db in rain_db_series if db > 0]
    rain_fraction = sum(1 for r in rain_series if r > 0) / n_steps
    sorted_snr    = sorted(snr_series)
    p10_idx       = max(0, int(0.10 * n_steps) - 1)

    return StationResult(
        name          = gs["name"],
        elevation     = elevation,
        slant_km      = slant_km,
        doppler_hz    = dop_hz,
        path_loss     = path_loss,
        gas_loss      = gas_loss,
        rain_height   = rain_h,
        eff_path      = eff_path,
        itu_k         = itu_k,
        itu_alpha     = itu_a,
        scint_sig     = scint_sig,
        snr_series    = snr_series,
        rain_series   = rain_series,
        rain_db_series = rain_db_series,
        scint_series  = scint_series,
        pkt_loss_series = pkt_loss_series,
        snr_mean      = statistics.mean(snr_series),
        snr_min       = min(snr_series),
        snr_std       = statistics.stdev(snr_series),
        snr_p10       = sorted_snr[p10_idx],
        rain_fraction = rain_fraction,
        avg_rain_db   = statistics.mean(rainy_steps) if rainy_steps else 0.0,
        avg_pkt_loss  = statistics.mean(pkt_loss_series),
        outage_fraction = sum(1 for s in snr_series if s < 3.0) / n_steps,
    )


def simulate_all(n_steps: int = DEFAULT_N_STEPS,
                 dt_s: float  = DEFAULT_DT_S,
                 force_rain: bool = False) -> list[StationResult]:
    """Run simulate_station() for every entry in GROUND_STATIONS."""
    return [simulate_station(gs, n_steps=n_steps, dt_s=dt_s,
                             force_rain=force_rain,
                             seed=hash(gs["name"]) % 100_000)
            for gs in GROUND_STATIONS]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  K.  CLI entry point (unchanged behaviour)                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _print_station(gs: dict, r: StationResult) -> None:
    W = 92
    p = gs["itu_rain"]
    print("=" * W)
    print(f"  STATION : {r.name}  ({gs['latitude']:+.1f}, {gs['longitude']:+.1f})  |  "
          f"Sat: {gs['sat_lon_deg']:+.1f}  |  El: {r.elevation:.1f}  |  "
          f"Range: {r.slant_km:.0f} km")
    print(f"  FSPL    : {r.path_loss:.2f} dB  |  Gas: {r.gas_loss:.3f} dB  "
          f"|  Rain height: {r.rain_height:.2f} km  |  Eff path: {r.eff_path:.2f} km")
    print(f"  P.838   : k={r.itu_k:.5f}, a={r.itu_alpha:.4f}  "
          f"|  Scint s={r.scint_sig:.4f} dB  |  Doppler: {r.doppler_hz:+.1f} Hz")
    print(f"  P.837-7 : R001={p['R001']} mm/h  R01={p['R01']} mm/h  "
          f"R1={p['R1']} mm/h  P_rain={p['P_rain']*100:.1f}%")
    print("-" * W)
    hdr = (f"{'t':>4} | {'R mm/h':>7} | {'Rain dB':>8} | "
           f"{'Gas dB':>7} | {'Scint dB':>9} | {'SNR dB':>8} | {'PktLoss':>8}")
    print(hdr)
    print("-" * W)
    for t, (snr, rain, rain_db, scint, pkt) in enumerate(
            zip(r.snr_series, r.rain_series, r.rain_db_series,
                r.scint_series, r.pkt_loss_series)):
        print(f"{t:>4} | {rain:>7.2f} | {rain_db:>8.3f} | "
              f"{r.gas_loss:>7.3f} | {scint:>+9.4f} | {snr:>8.2f} | {pkt:>8.4f}")
    print("=" * W)
    print(f"  SNR   mean={r.snr_mean:.2f}  min={r.snr_min:.2f}  "
          f"std={r.snr_std:.2f}  p10={r.snr_p10:.2f} dB")
    print(f"  Rain  {r.rain_fraction*100:.1f}% of steps  |  "
          f"avg rain atten={r.avg_rain_db:.2f} dB  |  "
          f"outage frac={r.outage_fraction*100:.1f}%")
    print(f"  Pkt loss mean={r.avg_pkt_loss:.4f}")
    print()


def _print_summary(results: list[StationResult]) -> None:
    W = 92
    print("=" * W)
    print("  CROSS-STATION SUMMARY")
    print("=" * W)
    hdr = (f"{'Station':<12} | {'El':>5} | {'SNR mean':>9} | {'SNR min':>8} | "
           f"{'SNR p10':>8} | {'Rain %':>7} | {'Outage %':>9} | {'PktLoss':>8}")
    print(hdr)
    print("-" * W)
    for r in results:
        print(f"{r.name:<12} | {r.elevation:>5.1f} | {r.snr_mean:>9.2f} | "
              f"{r.snr_min:>8.2f} | {r.snr_p10:>8.2f} | "
              f"{r.rain_fraction*100:>7.1f} | {r.outage_fraction*100:>9.1f} | "
              f"{r.avg_pkt_loss:>8.4f}")
    print("=" * W)


if __name__ == "__main__":
    freq_ghz = CARRIER_FREQ_HZ / 1e9
    print(f"\nSATELLITE LINK BUDGET SIMULATOR  --  {freq_ghz:.0f} GHz Ku-band  "
          f"|  Pol: {POLARIZATION}  |  Per-station GEO arc\n")
    gs_map = {gs["name"]: gs for gs in GROUND_STATIONS}
    results = simulate_all(n_steps=DEFAULT_N_STEPS, dt_s=DEFAULT_DT_S)
    for r in results:
        _print_station(gs_map[r.name], r)
    _print_summary(results)
