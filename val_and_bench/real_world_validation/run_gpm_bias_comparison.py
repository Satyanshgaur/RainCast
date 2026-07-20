import os
import sys
import numpy as np
from scipy.spatial.distance import jensenshannon

# Ensure we can import satlinksim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from satlinksim.domain.rain.engine import CorrelatedRainProcess

def calculate_average_rain_duration(series, dt_s=60):
    raining = series > 0.0
    padded = np.pad(raining, (1, 1), 'minimum')
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    durations = ends - starts
    if len(durations) == 0:
        return 0.0
    return np.mean(durations) * dt_s

def run_simulation_config(gs_config, n_steps, dt_s, use_aq, use_bq):
    proc = CorrelatedRainProcess(
        gs_config,
        dt_s=dt_s,
        use_corrected_quantiles=use_aq,
        use_random_onset_init=use_bq
    )
    return proc.generate_batch(n_steps)[:, 0]

def main():
    print("Running Double Validation simulation...")
    n_steps = 1000000
    dt_s = 60
    
    # 1. ITU Config Parameters
    delhi_itu_gs = {
        "name": "Delhi_ITU",
        "itu_rain": {
            "R001": 42.0,
            "R01": 19.0,
            "R1": 6.0,
            "P_rain": 0.053
        }
    }
    
    # 2. GPM Config Parameters
    delhi_gpm_gs = {
        "name": "Delhi_GPM",
        "itu_rain": {
            "R001": 90.0,
            "R01": 35.0,
            "R1": 12.0,
            "P_rain": 0.065
        }
    }
    
    # Standard bins for JS divergence (relative to GPM target series)
    bins = np.linspace(0.0, 150.0, 1501)
    
    # Generate GPM Target (Corrected) series to act as reference for JS divergence
    gpm_target_series = run_simulation_config(delhi_gpm_gs, n_steps, dt_s, use_aq=True, use_bq=True)
    p_gpm_target, _ = np.histogram(gpm_target_series, bins=bins, density=True)
    p_gpm_target = p_gpm_target + 1e-12
    p_gpm_target /= np.sum(p_gpm_target)
    
    configs = [
        ("Original (Bug A + B present)", False, False),
        ("Bug A only (Bug B present)", True, False),
        ("Bug A + Bug B (Fully Corrected)", True, True)
    ]
    
    print("\n--- Validation Against ITU Parameters (Target R0.01 = 42 mm/h) ---")
    for name, use_aq, use_bq in configs:
        series = run_simulation_config(delhi_itu_gs, n_steps, dt_s, use_aq, use_bq)
        r001 = np.percentile(series, 99.99)
        r0001 = np.percentile(series, 99.999)
        mean_r = np.mean(series)
        dur = calculate_average_rain_duration(series, dt_s)
        print(f"{name:<35} | R0.01: {r001:5.2f} | R0.001: {r0001:6.2f} | Mean: {mean_r:6.4f} | Dur: {dur:5.1f}s")
        
    print("\n--- Validation Against NASA GPM Parameters (Target R0.01 = 90 mm/h) ---")
    # Also print GPM Target stats
    gpm_r001 = np.percentile(gpm_target_series, 99.99)
    gpm_r0001 = np.percentile(gpm_target_series, 99.999)
    gpm_mean_r = np.mean(gpm_target_series)
    gpm_dur = calculate_average_rain_duration(gpm_target_series, dt_s)
    print(f"{'NASA GPM Target Reference':<35} | R0.01: {gpm_r001:5.2f} | R0.001: {gpm_r0001:6.2f} | Mean: {gpm_mean_r:6.4f} | Dur: {gpm_dur:5.1f}s")
    
    for name, use_aq, use_bq in configs:
        series = run_simulation_config(delhi_gpm_gs, n_steps, dt_s, use_aq, use_bq)
        r001 = np.percentile(series, 99.99)
        r0001 = np.percentile(series, 99.999)
        mean_r = np.mean(series)
        dur = calculate_average_rain_duration(series, dt_s)
        
        # JS Divergence
        p_sim, _ = np.histogram(series, bins=bins, density=True)
        p_sim = p_sim + 1e-12
        p_sim /= np.sum(p_sim)
        js_div = jensenshannon(p_gpm_target, p_sim)
        
        print(f"{name:<35} | R0.01: {r001:5.2f} | R0.001: {r0001:6.2f} | Mean: {mean_r:6.4f} | Dur: {dur:5.1f}s | JS Div: {js_div:.4f}")

if __name__ == "__main__":
    main()
