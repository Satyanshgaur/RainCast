import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Ensure we can import satlinksim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from satlinksim.ground_stations import GROUND_STATIONS

GPM_REFERENCES = {
    "Delhi": {"R001": 90.0, "R01": 35.0, "R1": 12.0, "P_r": 0.065},
    "Tokyo": {"R001": 75.0, "R01": 40.0, "R1": 15.0, "P_r": 0.075},
    "Berlin": {"R001": 28.0, "R01": 14.0, "R1": 5.5, "P_r": 0.065},
    "Sao Paulo": {"R001": 100.0, "R01": 58.0, "R1": 25.0, "P_r": 0.100}
}

def get_flawed_params(r001, r01):
    _z001 = 3.0902
    _z01  = 2.3263
    sigma_ln = (np.log(r001) - np.log(r01)) / (_z001 - _z01)
    mu_ln    = np.log(r01) - _z01 * sigma_ln
    return mu_ln, sigma_ln

def get_corrected_params(r001, r01, p_rain):
    # Dynamic normal quantiles based on p_rain
    z0001 = norm.ppf(1.0 - 0.0001 / p_rain)
    z001  = norm.ppf(1.0 - 0.001 / p_rain)
    sigma_ln = (np.log(r001) - np.log(r01)) / (z0001 - z001)
    mu_ln    = np.log(r01) - z001 * sigma_ln
    return mu_ln, sigma_ln

def generate_series(n_steps, mu_ln, sigma_ln, p_rain, seed=42):
    np.random.seed(seed)
    rho = np.exp(-60 / 300)
    p_onset = 1 - np.exp(-60 / (300 * (1 - p_rain) / p_rain))
    p_clear = 1 - np.exp(-60 / 300)
    
    ln_R = mu_ln
    raining = False
    series = np.zeros(n_steps)
    
    for t in range(n_steps):
        if not raining:
            if np.random.random() < p_onset:
                raining = True
                ln_R = mu_ln
        else:
            if np.random.random() < p_clear:
                raining = False
        
        if raining:
            noise = np.random.normal(0.0, 1.0)
            ln_R = rho * ln_R + np.sqrt(1 - rho**2) * sigma_ln * noise + (1 - rho) * mu_ln
            series[t] = min(np.exp(ln_R), 150.0)
            
    return series

def run_analysis():
    print("Analyzing Rain Generator Flaws against NASA GPM Reference Data...")
    n_steps = 100000
    
    for gs in GROUND_STATIONS:
        name = gs["name"]
        itu = gs["itu_rain"]
        gpm = GPM_REFERENCES[name]
        
        # Original (flawed) parameter calculation
        fl_mu, fl_sigma = get_flawed_params(itu["R001"], itu["R01"])
        fl_series = generate_series(n_steps, fl_mu, fl_sigma, itu["P_rain"])
        
        # Corrected parameter calculation
        co_mu, co_sigma = get_corrected_params(itu["R001"], itu["R01"], itu["P_rain"])
        co_series = generate_series(n_steps, co_mu, co_sigma, itu["P_rain"])
        
        # Percentiles
        fl_r001 = np.percentile(fl_series, 99.99)
        co_r001 = np.percentile(co_series, 99.99)
        
        print(f"\nGround Station: {name}")
        print(f"  Target R0.01:            {itu['R001']:.2f} mm/h")
        print(f"  Simulated R0.01 (Flawed):  {fl_r001:.2f} mm/h (Underestimation: {((itu['R001'] - fl_r001)/itu['R001'])*100:.1f}%)")
        print(f"  Simulated R0.01 (Correct): {co_r001:.2f} mm/h (Underestimation: {((itu['R001'] - co_r001)/itu['R001'])*100:.1f}%)")
        print(f"  Corrected Sigma/Mu:      {co_sigma:.4f} / {co_mu:.4f}")
        
    print("\nVerification Complete.")

if __name__ == "__main__":
    run_analysis()
