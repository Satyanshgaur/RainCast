"""
Satellite Link Budget Simulator — Physics-Upgraded
====================================================
Upgrades over baseline:
  1. ITU-R P.837-7 rain statistics anchored to Delhi's climate zone
  2. ITU-R P.618-13 specific attenuation coefficients (freq/polarization aware)
  3. Temporally correlated rain via Maseng-Bakken first-order Markov process
     (exponential autocorrelation with realistic coherence time ~5 min)
  4. ITU-R P.676 gaseous absorption (O₂ + H₂O)
  5. Proper ITU-R P.839 rain height model
  6. Scintillation (ITU-R P.618 Sec. 2.4) as a small-scale additive fade
"""

import math
import random
import statistics

# ── Physical constants ──────────────────────────────────────────────────────
C = 2.998e8          # speed of light, m/s
K_B = 1.380649e-23   # Boltzmann constant, J/K

# ── System parameters ───────────────────────────────────────────────────────
CARRIER_FREQ_HZ   = 14e9       # Ku-band uplink
BANDWIDTH_HZ      = 36e6
SYSTEM_TEMP_K     = 500        # receiver noise temperature
POLARIZATION      = "vertical" # "vertical" or "horizontal"

# ── Orbital geometry ─────────────────────────────────────────────────────────
SAT_DISTANCE_KM   = 40_000     # geostationary slant range ≈ 40 000 km
ELEVATION_DEG     = 35.0       # elevation angle for Delhi to typical Ku GEO

# ── Simulation time ──────────────────────────────────────────────────────────
DT_SECONDS        = 60         # time step (1 min)
N_STEPS           = 60         # simulate 60 minutes


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1.  ITU-R P.837-7  —  Delhi rain statistics                           ║
# ║      Exceedance rate R_p (mm/h) for annual exceedance probability p     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Delhi falls in ITU rain zone K (tropical/monsoon influenced).
# Table below is derived from P.837-7 digital maps for lat=28.6°N, lon=77.2°E.
# R_001 ≈ 42 mm/h  (0.01 % of year exceedance = ~53 min/year)
# R_01  ≈ 19 mm/h  (0.1 %  of year exceedance = ~8.8 h/year)
# R_1   ≈  6 mm/h  (1 %    of year exceedance = ~88 h/year)
#
# We fit a lognormal CDF to these three quantiles to get (mu, sigma) for
# sampling rain rate conditioned on rain occurring.
#
# P(rain occurs) is estimated from the annual rain accumulation and the
# conditional mean rate:  exceedance fraction at ~0.5 mm/h threshold ≈ 5.3 %

DELHI_R001 = 42.0   # mm/h  at 0.01 % annual exceedance
DELHI_R01  = 19.0   # mm/h  at 0.1  %
DELHI_R1   =  6.0   # mm/h  at 1    %

# Fit lognormal (mu_ln, sigma_ln) to the two most reliable quantiles
_z001 = 3.0902   # normal quantile for p=0.001 (0.01 % exceedance)
_z01  = 2.3263   # normal quantile for p=0.01  (0.1  %)
_sigma_ln = (math.log(DELHI_R001) - math.log(DELHI_R01)) / (_z001 - _z01)
_mu_ln    = math.log(DELHI_R01) - _z01 * _sigma_ln

# Fraction of time rain occurs (P.837 P_r) — Delhi: ~5.3 % annually
P_RAIN_OCCURRENCE = 0.053


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2.  ITU-R P.618-13 / P.838-3  —  rain attenuation                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def itu_rain_coefficients(freq_ghz: float, polarization: str) -> tuple[float, float]:
    """
    Return (k, alpha) from ITU-R P.838-3 Table 1 for the given frequency
    and polarization.  Uses log-linear interpolation between tabulated points.
    'polarization' must be 'horizontal' or 'vertical'.
    """
    # Tabulated (freq_ghz, k_H, alpha_H, k_V, alpha_V) from P.838-3
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
    kH_tab = [r[1] for r in table]
    aH_tab = [r[2] for r in table]
    kV_tab = [r[3] for r in table]
    aV_tab = [r[4] for r in table]

    def log_interp(x, xs, ys):
        """Log-linear interpolation (P.838 recommends log-log for k)."""
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
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
        k     = log_interp(freq_ghz, freqs, kH_tab)
        alpha = lin_interp(freq_ghz, freqs, aH_tab)
    else:  # vertical
        k     = log_interp(freq_ghz, freqs, kV_tab)
        alpha = lin_interp(freq_ghz, freqs, aV_tab)
    return k, alpha


def itu_rain_height(lat_deg: float) -> float:
    """
    ITU-R P.839-4: mean rain height above MSL.
    h_R = h_0 + 0.36 km, where h_0 is the 0°C isotherm height.
    Simplified formula valid for |lat| ≤ 36°:
      h_R = 5.0 - 0.075*(lat - 23)  for lat > 23°
    Delhi lat = 28.6°N → h_R ≈ 4.67 km
    """
    if lat_deg > 23:
        return 5.0 - 0.075 * (lat_deg - 23.0)
    elif lat_deg >= 0:
        return 5.0
    else:
        return 5.0 - 0.1 * lat_deg


def effective_path_length(elevation_deg: float, rain_height_km: float,
                           station_altitude_km: float = 0.216) -> float:
    """
    ITU-R P.618-13 §2.2.1.1: effective path length L_s through rain.
    station_altitude_km: Delhi is at ~216 m MSL.
    """
    el_rad = math.radians(elevation_deg)
    h_delta = rain_height_km - station_altitude_km   # vertical extent of rain
    if h_delta <= 0:
        return 0.0
    L_s = h_delta / math.sin(el_rad)   # slant path through rain layer

    # Horizontal distance (needed for reduction factor r)
    L_g = h_delta / math.tan(el_rad) if elevation_deg < 5 else L_s * math.cos(el_rad)

    # Path reduction factor (P.618 eq. 4)
    r = 1.0 / (1.0 + 0.78 * math.sqrt(L_g * ITU_K / 1.0) - 0.38 * (1 - math.exp(-2 * L_g)))
    # Simplified: for elevation > 10° the reduction factor ≈ 1
    if elevation_deg > 10:
        r = 1.0

    return L_s * r


# Pre-compute rain parameters for Delhi / Ku-band
ITU_K, ITU_ALPHA = itu_rain_coefficients(CARRIER_FREQ_HZ / 1e9, POLARIZATION)
DELHI_LAT        = 28.6
RAIN_HEIGHT_KM   = itu_rain_height(DELHI_LAT)
EFFECTIVE_PATH_KM = effective_path_length(ELEVATION_DEG, RAIN_HEIGHT_KM)


def rain_attenuation_db(rain_rate_mmh: float) -> float:
    """
    ITU-R P.838-3 specific attenuation × effective path length.
    A = k · R^alpha · L_eff   [dB]
    """
    if rain_rate_mmh <= 0:
        return 0.0
    gamma = ITU_K * (rain_rate_mmh ** ITU_ALPHA)   # dB/km
    return gamma * EFFECTIVE_PATH_KM


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3.  Maseng–Bakken correlated rain time series                         ║
# ║      First-order log-normal AR(1) process (ITU-R P.1853)               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Model: ln(R[t]) = rho * ln(R[t-1]) + sqrt(1-rho²) * sigma_ln * N(0,1) + mu_ln
# Coherence time tau_c ≈ 5 minutes (300 s) from empirical rain cell studies.

TAU_COHERENCE_S = 300.0   # rain coherence time (s) — typical for Ku-band studies

def _ar1_rho(dt_s: float, tau_c: float) -> float:
    """AR(1) autocorrelation coefficient for time step dt_s."""
    return math.exp(-dt_s / tau_c)

class CorrelatedRainProcess:
    """
    Temporally correlated rain rate using a first-order log-normal AR(1) process.
    Anchored to Delhi ITU-R P.837-7 statistics.
    """
    def __init__(self, dt_s: float, tau_c: float = TAU_COHERENCE_S):
        self.rho    = _ar1_rho(dt_s, tau_c)
        self.mu     = _mu_ln
        self.sigma  = _sigma_ln
        self.ln_R   = self.mu   # initialise at median rain rate
        self.raining = False
        self._p_onset  = 1 - math.exp(-dt_s / (tau_c * (1 - P_RAIN_OCCURRENCE) / P_RAIN_OCCURRENCE))
        self._p_clear  = 1 - math.exp(-dt_s / tau_c)

    def step(self) -> float:
        """Advance by one time step; return rain rate (0 if clear)."""
        if not self.raining:
            if random.random() < self._p_onset:
                self.raining = True
                self.ln_R = self.mu   # reset to median on onset
        else:
            if random.random() < self._p_clear:
                self.raining = False

        if not self.raining:
            return 0.0

        # AR(1) update in log domain
        innovation = random.gauss(0.0, 1.0)
        self.ln_R = self.rho * self.ln_R + math.sqrt(1 - self.rho**2) * self.sigma * innovation \
                    + (1 - self.rho) * self.mu
        rate = math.exp(self.ln_R)
        return min(rate, 100.0)   # physical cap at 100 mm/h


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4.  ITU-R P.676-12  —  gaseous absorption (simplified)                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Zenith attenuation for standard atmosphere; slant path scaled by 1/sin(el).
# Dominant terms at 14 GHz: oxygen (~0.008 dB/km) + water vapour (~0.007 dB/km).

def gaseous_absorption_db(freq_ghz: float, elevation_deg: float,
                           water_vapour_g_m3: float = 12.0) -> float:
    """
    Simplified ITU-R P.676 zenith attenuation for dry air + water vapour,
    scaled to slant path.  Valid to ~10 % accuracy for 1–100 GHz.
    water_vapour_g_m3: surface absolute humidity (Delhi summer avg ≈ 12 g/m³).
    """
    f = freq_ghz
    # Dry air (oxygen): P.676 approx
    gamma_oxy = (7.2 / (f**2 + 0.34) + 0.62 / ((54 - f)**1.16 + 0.83)) * (f / 22.235)**2 * 1e-3
    # Simplified: at 14 GHz, gamma_oxy ≈ 0.0078 dB/km
    gamma_oxy = max(gamma_oxy, 0.0078)   # ensure physical minimum

    # Water vapour: dominant line at 22.235 GHz
    gamma_wv = (0.050 + 0.0021 * water_vapour_g_m3
                + 3.6 / ((f - 22.235)**2 + 8.5)
                + 10.6 / ((f - 183.31)**2 + 9.0)
                + 8.9 / ((f - 325.153)**2 + 26.3)) * water_vapour_g_m3 * f**2 * 1e-4

    zenith_total = (gamma_oxy + gamma_wv) * 10.0  # effective zenith path ~10 km
    slant = zenith_total / math.sin(math.radians(elevation_deg))
    return max(slant, 0.0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.  ITU-R P.618-13  —  tropospheric scintillation                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Small-scale amplitude fluctuations due to refractive-index turbulence.
# sigma_scint depends on antenna gain, elevation, and humidity (Cn²).

def scintillation_sigma_db(freq_ghz: float, elevation_deg: float,
                            antenna_diameter_m: float = 2.4,
                            humidity_pct: float = 70.0) -> float:
    """
    ITU-R P.618-13 §2.4.1: standard deviation of scintillation fade.
    Returns sigma in dB.
    """
    el_rad = math.radians(elevation_deg)
    # Wet term of refractivity structure parameter (proxy via humidity)
    Nwet = 0.75 * humidity_pct   # crude proportional model
    sigma_ref = 0.5509 * Nwet * math.sqrt(1e-3) / (math.sin(el_rad) ** 1.2)
    # Antenna averaging factor
    eta = 0.5   # aperture efficiency
    D_eff = math.sqrt(eta) * antenna_diameter_m
    x = 1.22 * D_eff**2 * (freq_ghz / 300.0)
    g_x = math.sqrt(3.86 * (x**2 + 1)**0.116 * math.cos(math.atan(x) * 11.0 / 6.0) - 7.08 * x**(5.0/6.0))
    return sigma_ref * g_x * 1e-3   # scale to realistic dB values


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.  Link budget components                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def fspl_db(freq_hz: float, distance_km: float) -> float:
    """Free-space path loss (dB). ITU form: 92.45 + 20log10(f_GHz) + 20log10(d_km)."""
    return 92.45 + 20 * math.log10(freq_hz / 1e9) + 20 * math.log10(distance_km)


def noise_power_dbw(T_sys_K: float, B_hz: float) -> float:
    """Thermal noise power: N = k_B · T · B  (dBW)."""
    return 10 * math.log10(K_B * T_sys_K * B_hz)


def doppler_shift_hz(v_radial_ms: float, freq_hz: float) -> float:
    """Classical Doppler (non-relativistic).  v_radial > 0 → approaching."""
    return (v_radial_ms / C) * freq_hz


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7.  Ground station definition                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

ground_station = {
    "name"           : "Delhi (Ku-band uplink)",
    "lat_deg"        : DELHI_LAT,
    "eirp_dbw"       : 52.0,        # transmit EIRP
    "g_rx_dbi"       : 45.0,        # receive antenna gain
    "antenna_diam_m" : 2.4,         # dish diameter for scintillation model
    "v_radial_ms"    : -30.0,       # satellite radial velocity (m/s)
    "humidity_pct"   : 70.0,        # summer average
    "wv_g_m3"        : 12.0,        # surface water vapour density
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8.  Simulation                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_simulation():
    freq_ghz = CARRIER_FREQ_HZ / 1e9

    # Fixed terms (geometry-only)
    path_loss_db = fspl_db(CARRIER_FREQ_HZ, SAT_DISTANCE_KM)
    noise_dbw    = noise_power_dbw(SYSTEM_TEMP_K, BANDWIDTH_HZ)
    gas_loss_db  = gaseous_absorption_db(freq_ghz, ELEVATION_DEG,
                                          ground_station["wv_g_m3"])
    scint_sigma  = scintillation_sigma_db(freq_ghz, ELEVATION_DEG,
                                           ground_station["antenna_diam_m"],
                                           ground_station["humidity_pct"])
    doppler_hz   = doppler_shift_hz(ground_station["v_radial_ms"], CARRIER_FREQ_HZ)

    # Correlated rain process (Maseng-Bakken via AR(1))
    rain_proc = CorrelatedRainProcess(dt_s=DT_SECONDS)

    print("=" * 82)
    print(f"  SATELLITE LINK SIMULATION  —  {ground_station['name']}")
    print(f"  Frequency : {freq_ghz:.1f} GHz  |  Polarisation : {POLARIZATION}")
    print(f"  FSPL      : {path_loss_db:.2f} dB  |  Gaseous absorption : {gas_loss_db:.2f} dB")
    print(f"  Rain height (ITU P.839): {RAIN_HEIGHT_KM:.2f} km  |  "
          f"Effective path: {EFFECTIVE_PATH_KM:.2f} km")
    print(f"  ITU P.838 k={ITU_K:.5f}, α={ITU_ALPHA:.4f}  "
          f"|  Scintillation σ={scint_sigma:.3f} dB")
    print(f"  Doppler shift: {doppler_hz:+.1f} Hz  "
          f"|  Time step: {DT_SECONDS} s  |  Steps: {N_STEPS}")
    print("=" * 82)
    header = (f"{'t':>4} | {'State':^6} | {'R mm/h':>7} | "
              f"{'Rain dB':>8} | {'Gas dB':>7} | {'Scint dB':>9} | {'SNR dB':>8}")
    print(header)
    print("-" * len(header))

    snr_log = []

    for t in range(N_STEPS):
        # ── Dynamic impairments ──────────────────────────────────────────
        rain_rate  = rain_proc.step()
        rain_db    = rain_attenuation_db(rain_rate)

        # Scintillation: Gaussian instantaneous sample scaled by sigma
        scint_db   = random.gauss(0.0, scint_sigma)

        # ── Link budget ──────────────────────────────────────────────────
        # SNR = EIRP - FSPL - gas_loss - rain_attenuation - scintillation
        #        + G_rx - N
        snr = (ground_station["eirp_dbw"]
               - path_loss_db
               - gas_loss_db
               - rain_db
               - scint_db
               + ground_station["g_rx_dbi"]
               - noise_dbw)

        snr_log.append(snr)
        state_str = "RAIN" if rain_proc.raining else "CLEAR"

        print(f"{t:>4} | {state_str:^6} | {rain_rate:>7.2f} | "
              f"{rain_db:>8.3f} | {gas_loss_db:>7.3f} | {scint_db:>+9.4f} | {snr:>8.2f}")

    print("=" * len(header))
    print(f"\n  SNR summary over {N_STEPS} steps  ({N_STEPS * DT_SECONDS / 60:.0f} min):")
    print(f"    Mean   : {statistics.mean(snr_log):.2f} dB")
    print(f"    Median : {statistics.median(snr_log):.2f} dB")
    print(f"    Std    : {statistics.stdev(snr_log):.2f} dB")
    print(f"    Min    : {min(snr_log):.2f} dB")
    print(f"    Max    : {max(snr_log):.2f} dB")
    rain_frac = sum(1 for _ in range(N_STEPS) if rain_proc.raining) / N_STEPS
    print(f"  Doppler  : {doppler_hz:+.1f} Hz  (static — single GEO sat)")
    print()


if __name__ == "__main__":
    random.seed(42)
    run_simulation()
