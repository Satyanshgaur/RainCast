"""
Satellite Link Budget Simulator — Multi-Station, Physics-Upgraded
==================================================================
Physics models used:
  1. ITU-R P.837-7  — location-specific rain statistics (per station)
  2. ITU-R P.838-3  — specific rain attenuation coefficients (freq/pol aware)
  3. ITU-R P.839-4  — rain height (latitude dependent)
  4. ITU-R P.618-13 — effective path length, scintillation model
  5. ITU-R P.676-12 — gaseous absorption (O₂ + H₂O, humidity aware)
  6. ITU-R P.1853   — Maseng-Bakken AR(1) correlated rain time series
  7. Geometry       — GEO elevation angle computed from lat/lon/sat-lon
"""

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Optional

# ── Physical constants ───────────────────────────────────────────────────────
C   = 2.998e8          # speed of light, m/s
K_B = 1.380649e-23     # Boltzmann constant, J/K
R_E = 6371.0           # Earth radius, km
R_GEO = 42_164.0       # GEO orbit radius from Earth centre, km

# ── System-wide RF parameters ────────────────────────────────────────────────
CARRIER_FREQ_HZ = 14e9        # Ku-band uplink
BANDWIDTH_HZ    = 36e6
POLARIZATION    = "vertical"  # "vertical" | "horizontal"
# Each station points to its most commonly used regional GEO arc:
#   Delhi     → INSAT-4A / SES-8 at 83.0°E
#   Tokyo     → N-SAT-110 / JCSAT at 110.0°E
#   Berlin    → Astra 19.2°E (most popular European Ku slot)
#   Sao Paulo → Star One C2 at 70.0°W (typical Brazilian Ku slot)
STATION_SAT_LON: dict[str, float] = {
    "Delhi":      83.0,
    "Tokyo":     110.0,
    "Berlin":     19.2,
    "Sao Paulo":  -70.0,
}

# ── Simulation timing ────────────────────────────────────────────────────────
DT_SECONDS = 60    # 1-minute time step
N_STEPS    = 60    # simulate 60 minutes

# ── ITU-R P.837-7 rain statistics per location ──────────────────────────────
# Columns: R_001 (mm/h), R_01 (mm/h), R_1 (mm/h), P_rain (fraction of year)
# Values derived from P.837-7 digital maps for each station's lat/lon.
#
#   Delhi    (28.6°N,  77.2°E) — ITU zone K, monsoon climate
#   Tokyo    (35.6°N, 139.6°E) — ITU zone N, humid subtropical
#   Berlin   (52.5°N,  13.4°E) — ITU zone E, temperate maritime
#   Sao Paulo(-23.5°S, 46.6°W) — ITU zone N/P, tropical
#
ITU_RAIN_PARAMS: dict[str, dict] = {
    "Delhi":     {"R001": 42.0, "R01": 19.0, "R1":  6.0, "P_rain": 0.053},
    "Tokyo":     {"R001": 80.0, "R01": 42.0, "R1": 16.0, "P_rain": 0.072},
    "Berlin":    {"R001": 28.0, "R01": 14.0, "R1":  5.5, "P_rain": 0.065},
    "Sao Paulo": {"R001": 95.0, "R01": 55.0, "R1": 22.0, "P_rain": 0.095},
}

# ── Per-station climate parameters ──────────────────────────────────────────
# water_vapour_g_m3: surface absolute humidity (annual avg)
# humidity_pct:      relative humidity (for scintillation model)
# altitude_km:       station altitude above MSL
# antenna_diam_m:    ground dish diameter
# v_radial_ms:       satellite radial velocity (m/s, – = recession)
STATION_CLIMATE: dict[str, dict] = {
    "Delhi":     {"wv_g_m3": 12.0, "humidity_pct": 70, "altitude_km": 0.216, "antenna_diam_m": 2.4, "v_radial_ms": -30.0},
    "Tokyo":     {"wv_g_m3":  9.5, "humidity_pct": 65, "altitude_km": 0.040, "antenna_diam_m": 2.4, "v_radial_ms": -28.0},
    "Berlin":    {"wv_g_m3":  7.0, "humidity_pct": 75, "altitude_km": 0.034, "antenna_diam_m": 2.4, "v_radial_ms": -25.0},
    "Sao Paulo": {"wv_g_m3": 14.5, "humidity_pct": 80, "altitude_km": 0.760, "antenna_diam_m": 2.4, "v_radial_ms": -32.0},
}

# ── Ground stationdefinitions ───────────────────────────────────────────────
GROUND_STATIONS = [
    {
        "name":          "Delhi",
        "eirp_dbw":      52,
        "g_rx_dbi":      45,
        "system_temp_k": 500,
        "latitude":      28.6,
        "longitude":     77.2,
    },
    {
        "name":          "Tokyo",
        "eirp_dbw":      54,
        "g_rx_dbi":      48,
        "system_temp_k": 450,
        "latitude":      35.6,
        "longitude":     139.6,
    },
    {
        "name":          "Berlin",
        "eirp_dbw":      50,
        "g_rx_dbi":      44,
        "system_temp_k": 520,
        "latitude":      52.5,
        "longitude":     13.4,
    },
    {
        "name":          "Sao Paulo",
        "eirp_dbw":      53,
        "g_rx_dbi":      46,
        "system_temp_k": 600,
        "latitude":      -23.5,
        "longitude":     -46.6,
    },
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  A.  Geometry — GEO elevation angle                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def geo_elevation_deg(lat_deg: float, lon_deg: float, sat_lon_deg: float) -> float:
    """
    Compute elevation angle from a ground station to a GEO satellite.
    Uses the standard spherical Earth formula (ITU-R S.1066).
    """
    lat  = math.radians(lat_deg)
    dlon = math.radians(lon_deg - sat_lon_deg)
    # Central angle γ between station and sub-satellite point
    cos_gamma = math.cos(lat) * math.cos(dlon)
    # Elevation
    sin_el = (cos_gamma - R_E / R_GEO) / math.sqrt(1 - cos_gamma**2 + 1e-12)
    el_deg = math.degrees(math.asin(max(min(sin_el, 1.0), -1.0)))
    return el_deg


def geo_slant_range_km(lat_deg: float, lon_deg: float, sat_lon_deg: float) -> float:
    """Slant range to GEO satellite in km."""
    lat  = math.radians(lat_deg)
    dlon = math.radians(lon_deg - sat_lon_deg)
    cos_gamma = math.cos(lat) * math.cos(dlon)
    return math.sqrt(R_GEO**2 + R_E**2 - 2 * R_GEO * R_E * cos_gamma)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  B.  ITU-R P.838-3 — rain attenuation coefficients                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def itu_rain_coefficients(freq_ghz: float, polarization: str) -> tuple[float, float]:
    """Return (k, alpha) from ITU-R P.838-3 for given frequency and polarization."""
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
                return 10 ** (math.log10(ys[i]) + t * (math.log10(ys[i+1]) - math.log10(ys[i])))

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
    """Mean rain height above MSL (km) per ITU-R P.839-4."""
    a = abs(lat_deg)
    if a < 36:
        return max(5.0 - 0.075 * (a - 23.0), 3.0) if a > 23 else 5.0
    elif a < 70:
        return 5.0 - 0.1 * (a - 36.0)
    else:
        return 2.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  D.  ITU-R P.618-13 — effective rain path & rain attenuation           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def effective_path_length(elevation_deg: float, rain_height_km: float,
                           station_altitude_km: float, itu_k: float) -> float:
    """Effective slant path through rain layer (km), ITU-R P.618-13 §2.2.1."""
    el_rad   = math.radians(max(elevation_deg, 5.0))
    h_delta  = rain_height_km - station_altitude_km
    if h_delta <= 0:
        return 0.0
    L_s = h_delta / math.sin(el_rad)
    L_g = h_delta / math.tan(el_rad)
    # Path reduction factor: simplified (valid for el > 10°)
    r = 1.0 / (1.0 + 0.78 * math.sqrt(L_g * itu_k) - 0.38 * (1 - math.exp(-2 * L_g))) \
        if elevation_deg <= 10 else 1.0
    return L_s * r


def rain_attenuation_db(rain_rate_mmh: float, itu_k: float, itu_alpha: float,
                         eff_path_km: float) -> float:
    """A = k · R^α · L_eff  [dB]  (ITU-R P.838-3)."""
    if rain_rate_mmh <= 0 or eff_path_km <= 0:
        return 0.0
    return itu_k * (rain_rate_mmh ** itu_alpha) * eff_path_km


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  E.  ITU-R P.1853 — Maseng-Bakken AR(1) correlated rain process        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TAU_COHERENCE_S = 300.0   # rain cell coherence time ~5 min

class CorrelatedRainProcess:
    """
    First-order log-normal AR(1) rain time series (Maseng-Bakken).
    Parameters are derived from per-station ITU-R P.837-7 quantiles.
    """
    def __init__(self, station_name: str, dt_s: float,
                 tau_c: float = TAU_COHERENCE_S):
        p = ITU_RAIN_PARAMS[station_name]
        R001, R01, P_rain = p["R001"], p["R01"], p["P_rain"]
        _z001 = 3.0902; _z01 = 2.3263
        self.sigma  = (math.log(R001) - math.log(R01)) / (_z001 - _z01)
        self.mu     = math.log(R01) - _z01 * self.sigma
        self.rho    = math.exp(-dt_s / tau_c)
        self.ln_R   = self.mu
        self.raining = False
        # Transition probabilities derived from P_rain and tau_c
        mean_rain_dur_s = tau_c
        mean_clear_dur_s = tau_c * (1 - P_rain) / (P_rain + 1e-9)
        self._p_onset = 1 - math.exp(-dt_s / mean_clear_dur_s)
        self._p_clear = 1 - math.exp(-dt_s / mean_rain_dur_s)

    def step(self) -> float:
        """Advance one time step; return instantaneous rain rate (mm/h)."""
        if not self.raining:
            if random.random() < self._p_onset:
                self.raining = True
                self.ln_R = self.mu
        else:
            if random.random() < self._p_clear:
                self.raining = False

        if not self.raining:
            return 0.0

        # AR(1) update in log domain — preserves lognormal marginal
        self.ln_R = (self.rho * self.ln_R
                     + math.sqrt(1 - self.rho**2) * self.sigma * random.gauss(0.0, 1.0)
                     + (1 - self.rho) * self.mu)
        return min(math.exp(self.ln_R), 100.0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  F.  ITU-R P.676-12 — gaseous absorption                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def gaseous_absorption_db(freq_ghz: float, elevation_deg: float,
                           water_vapour_g_m3: float) -> float:
    """
    Simplified ITU-R P.676 slant-path attenuation (O₂ + H₂O).
    Valid ±10 % at 14 GHz for standard atmosphere.
    """
    f = freq_ghz
    gamma_oxy = max((7.2 / (f**2 + 0.34) + 0.62 / ((54 - f)**1.16 + 0.83))
                    * (f / 22.235)**2 * 1e-3, 0.0078)
    gamma_wv  = (0.050 + 0.0021 * water_vapour_g_m3
                 + 3.6 / ((f - 22.235)**2 + 8.5)
                 + 10.6 / ((f - 183.31)**2 + 9.0)
                 + 8.9 / ((f - 325.153)**2 + 26.3)) * water_vapour_g_m3 * f**2 * 1e-4
    zenith    = (gamma_oxy + gamma_wv) * 10.0   # effective zenith column ~10 km
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
                               - 7.08 * x**(5.0/6.0), 1e-6))
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


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  I.  Per-station simulation                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@dataclass
class StationResult:
    name:       str
    snr_log:    list[float]
    rain_steps: int
    elevation:  float
    slant_km:   float
    gas_loss:   float
    rain_height: float
    eff_path:   float
    itu_k:      float
    itu_alpha:  float
    scint_sig:  float
    doppler_hz: float


def run_station(gs: dict) -> StationResult:
    name    = gs["name"]
    lat     = gs["latitude"]
    lon     = gs["longitude"]
    climate = STATION_CLIMATE[name]
    freq_ghz = CARRIER_FREQ_HZ / 1e9

    # ── Geometry ────────────────────────────────────────────────────────────
    sat_lon     = STATION_SAT_LON[name]
    elevation   = geo_elevation_deg(lat, lon, sat_lon)
    slant_km    = geo_slant_range_km(lat, lon, sat_lon)

    # ── Fixed propagation terms ─────────────────────────────────────────────
    path_loss   = fspl_db(CARRIER_FREQ_HZ, slant_km)
    noise_dbw   = noise_power_dbw(gs["system_temp_k"], BANDWIDTH_HZ)
    gas_loss    = gaseous_absorption_db(freq_ghz, elevation, climate["wv_g_m3"])
    rain_h      = itu_rain_height(lat)
    itu_k, itu_a = itu_rain_coefficients(freq_ghz, POLARIZATION)
    eff_path    = effective_path_length(elevation, rain_h, climate["altitude_km"], itu_k)
    scint_sig   = scintillation_sigma_db(freq_ghz, elevation,
                                          climate["antenna_diam_m"],
                                          climate["humidity_pct"])
    dop_hz      = doppler_shift_hz(climate["v_radial_ms"], CARRIER_FREQ_HZ)

    # ── Correlated rain process ─────────────────────────────────────────────
    rain_proc = CorrelatedRainProcess(name, dt_s=DT_SECONDS)

    # ── Simulation loop ─────────────────────────────────────────────────────
    W = 90  # column width
    print("=" * W)
    print(f"  STATION : {name}  ({lat:+.1f}°, {lon:+.1f}°)  →  "
          f"Sat lon: {sat_lon:+.1f}°  |  Elevation: {elevation:.1f}°  |  Slant range: {slant_km:.0f} km")
    print(f"  FSPL    : {path_loss:.2f} dB  |  Gas absorption: {gas_loss:.3f} dB  "
          f"|  Rain height (P.839): {rain_h:.2f} km  |  Eff path: {eff_path:.2f} km")
    print(f"  P.838   : k={itu_k:.5f}, α={itu_a:.4f}  "
          f"|  Scint σ={scint_sig:.4f} dB  |  Doppler: {dop_hz:+.1f} Hz")
    print(f"  P.837-7 : R₀.₀₁={ITU_RAIN_PARAMS[name]['R001']} mm/h  "
          f"R₀.₁={ITU_RAIN_PARAMS[name]['R01']} mm/h  "
          f"R₁={ITU_RAIN_PARAMS[name]['R1']} mm/h  "
          f"P_rain={ITU_RAIN_PARAMS[name]['P_rain']*100:.1f}%")
    print("-" * W)
    hdr = (f"{'t':>4} | {'State':^6} | {'R mm/h':>7} | "
           f"{'Rain dB':>8} | {'Gas dB':>7} | {'Scint dB':>9} | {'SNR dB':>8}")
    print(hdr)
    print("-" * W)

    snr_log    = []
    rain_steps = 0

    for t in range(N_STEPS):
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

        snr_log.append(snr)
        if rain_proc.raining:
            rain_steps += 1

        state = "RAIN" if rain_proc.raining else "CLEAR"
        print(f"{t:>4} | {state:^6} | {rain_rate:>7.2f} | "
              f"{rain_db:>8.3f} | {gas_loss:>7.3f} | {scint_db:>+9.4f} | {snr:>8.2f}")

    print("=" * W)
    print(f"  SNR   mean={statistics.mean(snr_log):.2f} dB  "
          f"median={statistics.median(snr_log):.2f} dB  "
          f"std={statistics.stdev(snr_log):.2f} dB  "
          f"min={min(snr_log):.2f} dB  "
          f"max={max(snr_log):.2f} dB")
    print(f"  Rain fraction this run: {rain_steps}/{N_STEPS} steps "
          f"({100*rain_steps/N_STEPS:.1f}%)  "
          f"|  P.837 annual expectation: {ITU_RAIN_PARAMS[name]['P_rain']*100:.1f}%")
    print()

    return StationResult(
        name=name, snr_log=snr_log, rain_steps=rain_steps,
        elevation=elevation, slant_km=slant_km, gas_loss=gas_loss,
        rain_height=rain_h, eff_path=eff_path, itu_k=itu_k,
        itu_alpha=itu_a, scint_sig=scint_sig, doppler_hz=dop_hz,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  J.  Summary comparison table                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def print_summary(results: list[StationResult]) -> None:
    W = 90
    print("=" * W)
    print("  CROSS-STATION SUMMARY")
    print("=" * W)
    hdr = (f"{'Station':<12} | {'El °':>5} | {'Slant km':>9} | "
           f"{'Gas dB':>7} | {'Eff path':>9} | {'SNR mean':>9} | "
           f"{'SNR min':>8} | {'Doppler Hz':>11}")
    print(hdr)
    print("-" * W)
    for r in results:
        print(f"{r.name:<12} | {r.elevation:>5.1f} | {r.slant_km:>9.0f} | "
              f"{r.gas_loss:>7.3f} | {r.eff_path:>9.2f} | "
              f"{statistics.mean(r.snr_log):>9.2f} | "
              f"{min(r.snr_log):>8.2f} | {r.doppler_hz:>+11.1f}")
    print("=" * W)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  K.  Entry point                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    random.seed(42)
    freq_ghz = CARRIER_FREQ_HZ / 1e9
    print(f"\nSATELLITE LINK BUDGET SIMULATOR  —  {freq_ghz:.0f} GHz Ku-band uplink  "
          f"|  Pol: {POLARIZATION}  |  Per-station GEO slot\n")

    results = []
    for gs in GROUND_STATIONS:
        results.append(run_station(gs))

    print_summary(results) 
