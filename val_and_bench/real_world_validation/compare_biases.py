import os
import sys
import numpy as np
from scipy.spatial.distance import jensenshannon

# Ensure we can import satlinksim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from satlinksim.domain.rain.engine import CorrelatedRainProcess

def calculate_average_rain_duration(series):
    raining = series > 0.0
    padded = np.pad(raining, (1, 1), 'minimum')
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    durations = ends - starts
    if len(durations) == 0:
        return 0.0
    return np.mean(durations)  # in steps (minutes)

def main():
    print("Running Bias Comparison (Original vs Bug A vs Bug A + Bug B)...")
    
    n_steps = 1000000
    dt_s = 60
    
    # Delhi Ground Station Config
    delhi_gs = {
        "name": "Delhi",
        "itu_rain": {
            "R001": 42.0,   # mm/h at 0.01% exceedance
            "R01": 19.0,    # mm/h at 0.1% exceedance
            "R1": 6.0,      # mm/h at 1% exceedance
            "P_rain": 0.053
        }
    }
    
    # Delhi GPM Reference Config (target)
    gpm_gs = {
        "name": "Delhi_GPM",
        "itu_rain": {
            "R001": 90.0,   # GPM R0.01
            "R01": 35.0,    # GPM R0.1
            "R1": 12.0,     # GPM R1
            "P_rain": 0.065 # GPM Pr
        }
    }
    
    # 1. Reference GPM Series
    print("Generating NASA GPM reference series...")
    gpm_proc = CorrelatedRainProcess(gpm_gs, dt_s=dt_s, use_corrected_quantiles=True, use_random_onset_init=True)
    gpm_series = gpm_proc.generate_batch(n_steps)[:, 0]
    
    # Define common bins for JS divergence
    bins = np.linspace(0.0, 150.0, 1501)
    p_gpm, _ = np.histogram(gpm_series, bins=bins, density=True)
    p_gpm = p_gpm + 1e-12
    p_gpm /= np.sum(p_gpm)
    
    # Compute metrics for GPM reference
    gpm_r001 = np.percentile(gpm_series, 99.99)
    gpm_r0001 = np.percentile(gpm_series, 99.999)
    gpm_mean = np.mean(gpm_series)
    gpm_dur = calculate_average_rain_duration(gpm_series) * dt_s # duration in seconds
    
    results = []
    
    configs = [
        ("Original (Bug A + B present)", False, False),
        ("Bug A only (Bug B present)", True, False),
        ("Bug A + Bug B (Fully Corrected)", True, True)
    ]
    
    for label, use_aq, use_bq in configs:
        print(f"Simulating config: {label}...")
        proc = CorrelatedRainProcess(
            delhi_gs, 
            dt_s=dt_s, 
            use_corrected_quantiles=use_aq, 
            use_random_onset_init=use_bq
        )
        series = proc.generate_batch(n_steps)[:, 0]
        
        # Calculate percentiles
        r001 = np.percentile(series, 99.99)
        r0001 = np.percentile(series, 99.999)
        mean_rate = np.mean(series)
        duration_s = calculate_average_rain_duration(series) * dt_s
        
        # JS Divergence
        p_sim, _ = np.histogram(series, bins=bins, density=True)
        p_sim = p_sim + 1e-12
        p_sim /= np.sum(p_sim)
        js_div = jensenshannon(p_gpm, p_sim)
        
        results.append({
            "config": label,
            "R0.01": r001,
            "R0.001": r0001,
            "mean_rate": mean_rate,
            "duration": duration_s,
            "js_div": js_div
        })
        
    print("\n" + "="*80)
    print(f"{'Configuration':<35} | {'R0.01':<8} | {'R0.001':<8} | {'Mean Rate':<10} | {'Duration (s)':<12} | {'JS Div':<8}")
    print("-" * 88)
    # Print GPM target first
    print(f"{'NASA GPM Target':<35} | {gpm_r001:8.2f} | {gpm_r0001:8.2f} | {gpm_mean:10.4f} | {gpm_dur:12.1f} | {'0.0000':<8}")
    for res in results:
        print(f"{res['config']:<35} | {res['R0.01']:8.2f} | {res['R0.001']:8.2f} | {res['mean_rate']:10.4f} | {res['duration']:12.1f} | {res['js_div']:.4f}")
    print("="*80)
    
if __name__ == "__main__":
    main()
