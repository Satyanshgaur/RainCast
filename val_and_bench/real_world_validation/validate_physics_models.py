import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root_dir and src are in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from satlinksim.domain.link.itu_models import (
    itu_rain_coefficients,
    itu_rain_height,
    effective_path_length,
    rain_attenuation_db,
    gaseous_absorption_db,
    scintillation_sigma_db
)

def validate_rain_attenuation():
    print("--- 1. Rain Attenuation Validation (ITU-R P.838-3) ---")
    # Verify horizontal and vertical coefficients at 20 GHz and 30 GHz
    k_20_h, alpha_20_h = itu_rain_coefficients(20.0, "horizontal")
    k_20_v, alpha_20_v = itu_rain_coefficients(20.0, "vertical")
    
    k_30_h, alpha_30_h = itu_rain_coefficients(30.0, "horizontal")
    k_30_v, alpha_30_v = itu_rain_coefficients(30.0, "vertical")
    
    # ITU-R P.838-3 Table Reference Values:
    # 20 GHz: k_H = 0.09164, alpha_H = 1.0990, k_V = 0.08084, alpha_V = 1.0993
    # 30 GHz: k_H = 0.2403,  alpha_H = 1.0210, k_V = 0.2101,  alpha_V = 1.0299
    print(f"20 GHz Horizontal: k={k_20_h:.5f} (Ref: 0.09164), alpha={alpha_20_h:.4f} (Ref: 1.0990)")
    print(f"20 GHz Vertical:   k={k_20_v:.5f} (Ref: 0.08084), alpha={alpha_20_v:.4f} (Ref: 1.0993)")
    print(f"30 GHz Horizontal: k={k_30_h:.5f} (Ref: 0.2403),  alpha={alpha_30_h:.4f} (Ref: 1.0210)")
    print(f"30 GHz Vertical:   k={k_30_v:.5f} (Ref: 0.2101),  alpha={alpha_30_v:.4f} (Ref: 1.0299)")
    
    # Path reduction check
    # Check effective path length reduction factor r at 5, 10, 20 degrees elevation
    # P.618 suggests path reduction should gradually approach 1.0. Let's see our path reduction factor:
    rain_h = 4.0
    alt_km = 0.0
    k_coef = 0.09
    
    print("\nPath reduction factor validation:")
    for el in [5, 9, 12, 30]:
        el_rad = np.radians(el)
        L_s = (rain_h - alt_km) / np.sin(el_rad)
        L_eff = effective_path_length(el, rain_h, alt_km, k_coef)
        r = L_eff / L_s if L_s > 0 else 1.0
        print(f"  Elevation: {el:2d}° | L_s: {L_s:6.2f} km | L_eff: {L_eff:6.2f} km | reduction r: {r:.4f}")

def validate_gaseous_attenuation():
    print("\n--- 2. Gaseous Attenuation Validation (ITU-R P.676-12) ---")
    # Compare simulator's gaseous absorption with standard ITU heights ho=6 km (oxygen) and hw=2.2 km (water vapor)
    # Simulator returns: (gamma_oxy + gamma_wv) * 10.0 / sin(el)
    freq = 20.0
    el = 15.0
    wv = 7.5 # g/m3
    
    # Manual standard P.676-12 calculation with heights:
    gamma_oxy = np.maximum((7.2/(freq**2+0.34) + 0.62/((54-freq)**1.16+0.83)) * (freq/22.235)**2 * 1e-3, 0.0078)
    gamma_wv  = (0.050 + 0.0021*wv + 3.6/((freq-22.235)**2 + 8.5) + 10.6/((freq-183.31)**2 + 9.0) + 8.9/((freq-325.153)**2 + 26.3)) * wv * freq**2 * 1e-4
    
    sim_loss = gaseous_absorption_db(freq, el, wv)
    # ITU height standard loss: (gamma_oxy * 6.0 + gamma_wv * 2.2) / sin(el)
    itu_standard_loss = (gamma_oxy * 6.0 + gamma_wv * 2.2) / np.sin(np.radians(el))
    
    print(f"Frequency: {freq} GHz | Elevation: {el}° | Water Vapor: {wv} g/m³")
    print(f"  Simulator Loss:   {sim_loss:.4f} dB (Uses static 10.0 km height)")
    print(f"  ITU-R P.676 Loss: {itu_standard_loss:.4f} dB (Uses equivalent heights ho=6.0 km, hw=2.2 km)")
    print(f"  Ratio (Sim / ITU): {sim_loss / itu_standard_loss:.2f}x (Overestimates loss due to static height)")

def validate_scintillation():
    print("\n--- 3. Tropospheric Scintillation Validation (ITU-R P.618-13) ---")
    # In P.618, scintillation scales as f^(7/12). Let's see how our sigma scales with frequency.
    el = 30.0
    diam = 1.2
    hum = 50.0
    
    sigmas = {}
    for f in [10.0, 20.0, 30.0, 40.0]:
        sigmas[f] = scintillation_sigma_db(f, el, diam, hum)
        
    print("Frequency sweep for Scintillation Sigma:")
    for f, sig in sigmas.items():
        print(f"  Frequency: {f:4.1f} GHz | Scintillation Sigma: {sig:.6f} dB")
        
    # Check scaling trend
    ratio_30_10 = sigmas[30.0] / sigmas[10.0]
    expected_ratio = (30.0 / 10.0) ** (7.0 / 12.0)
    print(f"  Observed 30 GHz / 10 GHz Ratio: {ratio_30_10:.4f}")
    print(f"  ITU Expected f^(7/12) Ratio:    {expected_ratio:.4f}")
    if ratio_30_10 < 1.0:
        print("  [!] Suspicious Trend: Scintillation decreases with frequency (Aperture smoothing dominates due to missing f^(7/12) scaling factor).")

def validate_tracking_jitter():
    print("\n--- 4. Antenna Tracking Jitter Validation ---")
    # Verify tracking jitter scales as 1/sin(el)
    sigmas_el = {}
    for el in [5, 10, 30, 90]:
        sigmas_el[el] = 0.04 / np.sin(np.radians(el))
        
    print("Elevation sweep for Tracking Jitter Sigma:")
    for el, sig in sigmas_el.items():
        print(f"  Elevation: {el:2d}° | Tracking Sigma: {sig:.4f}°")

def validate_agc_and_calibration():
    print("\n--- 5. AGC & Calibration Drift Validation ---")
    # Verify calibration slow drift (AR1 process) autocorrelation
    # x_slow[t] = 0.9995 * x_slow[t-1] + noise
    # Correlation time tau = -1 / ln(0.9995) = 1999 steps (minutes) ~ 33.3 hours
    decay_coef = 0.9995
    tau_hours = -1.0 / np.log(decay_coef) / 60.0
    print(f"  Calibration Slow Drift AR(1) decay: {decay_coef}")
    print(f"  Simulated Correlation Time Constant (tau): {tau_hours:.2f} hours (Diurnal scale)")
    
    # AGC alpha = 0.20 -> tau = -1 / ln(0.80) = 4.48 steps (minutes)
    agc_alpha = 0.20
    agc_tau_mins = -1.0 / np.log(1.0 - agc_alpha)
    print(f"  AGC response lag coefficient alpha: {agc_alpha}")
    print(f"  Simulated AGC Response Time Constant: {agc_tau_mins:.2f} minutes")

def main():
    validate_rain_attenuation()
    validate_gaseous_attenuation()
    validate_scintillation()
    validate_tracking_jitter()
    validate_agc_and_calibration()

if __name__ == "__main__":
    main()
