
import time
import numpy as np
from datetime import datetime, timezone, timedelta
from propogate import Propagator, Satellite, Constellation, geodetic_to_ecef
from satellite_link_sim import (
    simulate_all_batched, run_monte_carlo, GROUND_STATIONS,
    fspl_db, itu_rain_coefficients, itu_rain_height,
    gaseous_absorption_db, effective_path_length,
    rain_attenuation_db, scintillation_sigma_db,
    HandoffManager, doppler_shift_hz, CorrelatedRainProcess
)

def validate_physics():
    print("--- 1. Physical Validation ---")
    
    # 1.1 FSPL
    # Formula: 92.45 + 20log10(f_ghz) + 20log10(d_km)
    dist = 40000.0
    freq = 14e9
    calc = fspl_db(freq, dist)
    ref = 92.45 + 20*np.log10(14.0) + 20*np.log10(dist)
    error = abs(calc - ref)
    print(f"FSPL: Max error {error:.2e} dB")

    # 1.2 Rain (P.838)
    # We'll compare against a fixed reference point
    k, alpha = itu_rain_coefficients(14.0, "vertical")
    # Reference values for 14GHz V (ITU-R P.838-3): k ~ 0.030, alpha ~ 1.19
    # Our implementation uses interpolation.
    print(f"Rain Coeffs (14GHz V): k={k:.4f}, alpha={alpha:.4f}")
    
    # 1.3 Geometry
    # Zenith case: lat=0, lon=0, alt=0; sat at 0 lon, alt=35786
    # Slant range should be exactly 35786
    gs_pos = geodetic_to_ecef(0, 0, 0)
    sat_pos = geodetic_to_ecef(0, 0, 35786.0) # GEO height
    dist = np.linalg.norm(sat_pos - gs_pos)
    rel_error = abs(dist - 35786.0) / 35786.0 * 100
    print(f"Geometry: Slant range relative error {rel_error:.4f}%")

    # 1.4 AR(1) Autocorr
    # 이론: rho = exp(-dt/tau)
    dt = 60.0
    tau = 300.0
    theo_rho = np.exp(-dt/tau)
    # Run a long process to measure
    proc = CorrelatedRainProcess(GROUND_STATIONS[0], dt_s=dt)
    samples = [proc.step()[0] for _ in range(10000)]
    samples = [s for s in samples if s > 0] # only rainy steps for log-normal corr
    if len(samples) > 100:
        log_samples = np.log(samples)
        corr = np.corrcoef(log_samples[:-1], log_samples[1:])[0,1]
        print(f"AR(1) Autocorr: Measured {corr:.3f} vs Theo {theo_rho:.3f}")

def benchmark_parallelism():
    print("\n--- 2. Performance Validation ---")
    n_steps = 20000 # Larger steps to amortize process overhead
    gs = GROUND_STATIONS
    
    # Speedup Table
    print("| Workers | Speedup | Efficiency |")
    print("| ------- | ------- | ---------- |")
    
    # Serial baseline
    t0 = time.perf_counter()
    _ = simulate_all_batched(gs, n_steps=n_steps)
    t_serial = time.perf_counter() - t0
    print(f"| 1       | 1.0     | 100%       |")
    
    for workers in [2, 4, 8, 12]:
        t0 = time.perf_counter()
        _ = run_monte_carlo(workers, gs, n_steps=n_steps)
        t_par = time.perf_counter() - t0
        # Seq baseline should be n iterations for fair speedup.
        t_seq_n = t_serial * workers
        speedup = t_seq_n / t_par
        efficiency = (speedup / workers) * 100
        print(f"| {workers:2d}      | {speedup:7.1f} | {efficiency:9.0f}%    |")

def profile_runtime():
    print("\n--- 3. Runtime Breakdown ---")
    n_steps = 50000
    gs = GROUND_STATIONS[0]
    propagator = Propagator()
    const = Constellation.from_norad_ids("Profile", [26766, 26900, 27380])
    times = [datetime.now(timezone.utc) + timedelta(seconds=i*60) for i in range(n_steps)]
    
    # 1. Propagation (SGP4 + Relative Vectors)
    t0 = time.perf_counter()
    geos = []
    for sat in const.satellites:
        geos.append(propagator.get_geometry_batch(sat, times, gs["latitude"], gs["longitude"], gs["altitude_km"]))
    t_prop = time.perf_counter() - t0
    
    # 2. Link Budget Components
    itu_k, itu_a = itu_rain_coefficients(14.0, "vertical")
    rain_h = itu_rain_height(gs["latitude"])
    rain_rates = np.zeros(n_steps) # for profiling link budget calc speed
    
    # Profile Link Budget (Gas + Rain Atten + Scint)
    t0 = time.perf_counter()
    for geo in geos:
        _ = fspl_db(14e9, geo.slant_range_km)
        _ = gaseous_absorption_db(14.0, geo.elevation_deg, gs["wv_g_m3"])
        ep = effective_path_length(geo.elevation_deg, rain_h, gs["altitude_km"], itu_k)
        _ = rain_attenuation_db(rain_rates, itu_k, itu_a, ep)
        _ = scintillation_sigma_db(14.0, geo.elevation_deg, gs["antenna_diam_m"], gs["humidity_pct"])
    t_link = time.perf_counter() - t0
    
    # 3. Rain Process (MB Process)
    t0 = time.perf_counter()
    rain_proc = CorrelatedRainProcess([gs], dt_s=60)
    for _ in range(n_steps):
        _ = rain_proc.step()
    t_rain = time.perf_counter() - t0

    # 4. Handoff
    t0 = time.perf_counter()
    hm = HandoffManager()
    metrics = np.zeros((3, n_steps))
    for t in range(n_steps):
        _ = hm.select(t, ["A", "B", "C"], metrics[:, t])
    t_handoff = time.perf_counter() - t0
    
    total = t_prop + t_link + t_rain + t_handoff
    
    print("| Component    | Runtime Share |")
    print("| ------------ | ------------- |")
    print(f"| SGP4/Geometry| {t_prop/total*100:11.1f}%   |")
    print(f"| Link Budget  | {t_link/total*100:11.1f}%   |")
    print(f"| Rain Process | {t_rain/total*100:11.1f}%   |")
    print(f"| Handoff      | {t_handoff/total*100:11.1f}%   |")

def validate_network():
    print("\n--- 4. Network Validation ---")
    # Serial vs Parallel matching
    n_iter = 10
    n_steps = 100
    gs = [GROUND_STATIONS[0]]
    base_seed = 42

    # Serial (matching the seed sequence of run_monte_carlo)
    res_ser = []
    for i in range(n_iter):
        res_ser.append(simulate_all_batched(gs, n_steps=n_steps, seed=base_seed + i))
    ser_mean = np.mean([r[0].snr_mean for r in res_ser])

    # Parallel
    res_par = run_monte_carlo(n_iter, gs, n_steps=n_steps, seed=base_seed)
    par_mean = np.mean([r[0].snr_mean for r in res_par])

    diff = abs(ser_mean - par_mean) / abs(ser_mean) * 100
    print(f"Monte Carlo Parallel Match: Aggregate metric difference {diff:.8f}%")


if __name__ == "__main__":
    validate_physics()
    benchmark_parallelism()
    profile_runtime()
    validate_network()
