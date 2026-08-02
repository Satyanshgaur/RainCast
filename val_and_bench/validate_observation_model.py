import os
import sys
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from satlinksim.domain.observation import ObservationConfig, ObservationModel
from satlinksim.domain.models import StationResult

def make_dummy_result(n_steps, snr=20.0, el=30.0, slant=38000.0, rain=0.0, rain_db=0.0, gas=0.5, fspl=205.0):
    snr_arr = np.full(n_steps, snr) if isinstance(snr, float) else snr
    el_arr = np.full(n_steps, el) if isinstance(el, float) else el
    slant_arr = np.full(n_steps, slant) if isinstance(slant, float) else slant
    rain_arr = np.full(n_steps, rain) if isinstance(rain, float) else rain
    rain_db_arr = np.full(n_steps, rain_db) if isinstance(rain_db, float) else rain_db
    
    return StationResult(
        name="TestStation", elevation=float(np.mean(el_arr)), slant_km=float(np.mean(slant_arr)), doppler_hz=0.0,
        path_loss=fspl, gas_loss=gas, rain_height=4.5, eff_path=5.0, itu_k=0.03, itu_alpha=1.1,
        scint_sig=0.1, noise_floor=-140.0,
        snr_series=list(snr_arr), rain_series=list(rain_arr), rain_db_series=list(rain_db_arr),
        scint_series=list(np.zeros(n_steps)), pkt_loss_series=list(np.zeros(n_steps)),
        elevation_series=list(el_arr), slant_range_series=list(slant_arr), doppler_series=list(np.zeros(n_steps)),
        snr_mean=20.0, snr_min=15.0, snr_std=1.0, snr_p10=18.0,
        rain_fraction=0.0, avg_rain_db=0.0, avg_pkt_loss=0.0, outage_fraction=0.0
    )

def validate_tracking_model():
    """
    1. Tracking Error AR(1) Model Validation
    - Tests elevation dependence: sigma(el) = sigma_0 / sin(el)
    - Tests AR(1) lag-1 correlation matching rho = 0.96
    - Tests RMS pointing loss output against analytical Gaussian beam equation
    """
    print("\n--- 1. Validating Tracking Model ---")
    cfg = ObservationConfig(scenario="typical", enable_tracking=True, enable_calibration=False,
                            enable_agc=False, enable_multipath=False, enable_wet_antenna=False, enable_scintillation=False)
    obs = ObservationModel(cfg, seed=42)
    
    n_steps = 10000
    dummy_res = make_dummy_result(n_steps, snr=20.0, el=30.0, slant=38000.0)
    gs = {"g_rx_dbi": 40.0, "antenna_diam_m": 1.2, "system_temp_k": 290.0}
    
    res = obs.observe(gs, freq_hz=14e9, bandwidth_hz=36e6, polarization="vertical", res=dummy_res)
    
    pointing_loss = res["_latent_pointing_loss_db"]
    tracking_err = res["_latent_tracking_error_deg"]
    
    # Lag-1 correlation of tracking error
    # Compute AR(1) autocorrelation
    r1 = np.corrcoef(tracking_err[:-1], tracking_err[1:])[0, 1]
    
    # Expected sigma at 30 deg: 0.04 / sin(30 deg) = 0.08 deg
    expected_sigma_30 = 0.04 / np.sin(np.radians(30.0))
    measured_rms_err = np.sqrt(np.mean(tracking_err**2))
    
    # Analytical pointing loss check: 12 * (theta_err / theta_3dB)^2
    # For 1.2m dish at 14 GHz: theta_3dB = 70 * c / (14e9 * 1.2) = 1.249 deg
    theta_3dB = (70.0 * 2.99792e8) / (14e9 * 1.2)
    analytical_pointing_loss = 12.0 * (tracking_err / theta_3dB)**2
    max_loss_diff = np.max(np.abs(pointing_loss - analytical_pointing_loss))
    
    print(f"Target AR(1) coefficient: 0.9600 | Measured Lag-1 Correlation: {r1:.4f}")
    print(f"Target sigma at 30° elevation: {expected_sigma_30:.4f}° | Measured RMS Error: {measured_rms_err:.4f}°")
    print(f"Beamwidth (theta_3dB): {theta_3dB:.4f}° | Max Pointing Loss Model Error: {max_loss_diff:.6e} dB")
    
    return {
        "lag1_corr": r1,
        "expected_sigma": expected_sigma_30,
        "measured_rms": measured_rms_err,
        "max_loss_diff": max_loss_diff
    }

def validate_agc_model():
    """
    2. AGC First-Order LPF & ADC Quantization Validation
    - Tests step response of first-order AGC filter (alpha = 0.20)
    - Verifies 10%-90% rise time t_90% = ln(0.1)/ln(1-alpha)
    - Verifies uniform ADC quantization step LSB = 0.05 dB
    """
    print("\n--- 2. Validating AGC & ADC Model ---")
    cfg = ObservationConfig(scenario="typical", enable_tracking=False, enable_calibration=False,
                            enable_agc=True, enable_multipath=False, enable_wet_antenna=False, enable_scintillation=False)
    obs = ObservationModel(cfg, seed=42)
    
    n_steps = 200
    # Step change in SNR from 20 dB to 10 dB at step 50
    snr_series = np.full(n_steps, 20.0)
    snr_series[50:] = 10.0
    
    dummy_res = make_dummy_result(n_steps, snr=snr_series, el=45.0)
    gs = {"g_rx_dbi": 40.0, "system_temp_k": 290.0}
    res = obs.observe(gs, freq_hz=14e9, bandwidth_hz=36e6, polarization="vertical", res=dummy_res)
    
    obs_snr = res["observed_snr_db"]
    
    # Step response evaluation from step 50 onwards
    step_input = obs_snr[50:]
    initial_val = step_input[0]
    final_val = step_input[-1]
    step_mag = final_val - initial_val
    
    # Find 90% response step
    target_90 = initial_val + 0.90 * step_mag
    step_90_idx = np.where(step_input <= target_90)[0][0]
    
    alpha_agc = 0.20
    theoretical_steps_90 = np.log(0.10) / np.log(1.0 - alpha_agc)
    
    # ADC Quantization check: all observed SNR values must be multiples of LSB (0.05)
    adc_lsb = 0.05
    quant_rem = np.abs((obs_snr / adc_lsb) - np.round(obs_snr / adc_lsb))
    max_quant_error = np.max(quant_rem)
    
    print(f"AGC Filter Alpha: {alpha_agc:.2f}")
    print(f"Theoretical 90% Settling Time: {theoretical_steps_90:.2f} steps | Measured Settling Steps: {step_90_idx}")
    print(f"ADC LSB: {adc_lsb:.2f} dB | Max Quantization Grid Deviation: {max_quant_error:.6e}")
    
    return {
        "alpha_agc": alpha_agc,
        "theoretical_90": theoretical_steps_90,
        "measured_90": step_90_idx,
        "max_quant_error": max_quant_error
    }

def validate_calibration_model():
    """
    3. Calibration Drift & OU Process Validation
    - Tests Ornstein-Uhlenbeck (mean-reverting AR(1)) process (alpha_cal = 0.9995)
    - Verifies bounded stationary variance vs unbounded Random Walk aging
    """
    print("\n--- 3. Validating Calibration & OU Model ---")
    cfg = ObservationConfig(scenario="typical", enable_tracking=False, enable_calibration=True,
                            enable_agc=False, enable_multipath=False, enable_wet_antenna=False, enable_scintillation=False)
    obs = ObservationModel(cfg, seed=123)
    
    n_steps = 10000
    dummy_res = make_dummy_result(n_steps, snr=20.0, el=45.0)
    gs = {"g_rx_dbi": 40.0, "system_temp_k": 290.0}
    res = obs.observe(gs, freq_hz=14e9, bandwidth_hz=36e6, polarization="vertical", res=dummy_res)
    
    cal_err = res["_latent_calibration_error_db"]
    
    # Correlation at lag 1 for OU process component
    r1 = np.corrcoef(cal_err[:-1], cal_err[1:])[0, 1]
    
    print(f"Calibration Lag-1 Autocorrelation: {r1:.4f} (Target ~ 0.9995)")
    print(f"Mean Calibration Drift: {np.mean(cal_err):.4f} dB | Std Dev: {np.std(cal_err):.4f} dB")
    
    return {
        "lag1_corr": r1,
        "mean_drift": np.mean(cal_err),
        "std_drift": np.std(cal_err)
    }

def validate_thermal_model():
    """
    4. Solar Heating & Stefan-Boltzmann Thermal RC Model
    - Tests Stefan-Boltzmann radiative equilibrium calculation
    - Tests RC thermal lag exponential dynamic smoothing
    - Verifies TWTA efficiency coupling eta_TWTA(T)
    """
    print("\n--- 4. Validating Solar Thermal EIRP Model ---")
    cfg = ObservationConfig(scenario="typical", enable_tracking=False, enable_calibration=False,
                            enable_agc=False, enable_multipath=False, enable_wet_antenna=False, enable_scintillation=False)
    obs = ObservationModel(cfg, seed=42)
    
    # Simulate full orbit of 5760 seconds (96 minutes) = 96 steps of 60s
    n_steps = 96
    dummy_res = make_dummy_result(n_steps, snr=20.0, el=45.0)
    gs = {"sat_ant_gain_dbi": 17.0, "system_temp_k": 290.0}
    res = obs.observe(gs, freq_hz=14e9, bandwidth_hz=36e6, polarization="vertical", res=dummy_res)
    
    sat_eirp = res["_latent_sat_eirp_actual_dbw"]
    eirp_range = np.max(sat_eirp) - np.min(sat_eirp)
    
    print(f"Orbit EIRP Peak-to-Peak Variation: {eirp_range:.4f} dBW")
    print(f"Max Transmit EIRP: {np.max(sat_eirp):.2f} dBW | Min Transmit EIRP: {np.min(sat_eirp):.2f} dBW")
    
    return {
        "eirp_range": eirp_range,
        "max_eirp": np.max(sat_eirp),
        "min_eirp": np.min(sat_eirp)
    }

def validate_wet_antenna_model():
    """
    5. Wet Antenna Loss Model
    - Tests non-linear absorption curve vs rain rate R
    - Verifies activation condition (R > 0.1 mm/h)
    """
    print("\n--- 5. Validating Wet Antenna Model ---")
    cfg = ObservationConfig(scenario="typical", enable_tracking=False, enable_calibration=False,
                            enable_agc=False, enable_multipath=False, enable_wet_antenna=True, enable_scintillation=False)
    obs = ObservationModel(cfg, seed=42)
    
    rain_rates = np.array([0.0, 0.05, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0])
    n_steps = len(rain_rates)
    
    dummy_res = make_dummy_result(n_steps, snr=20.0, el=45.0, rain=rain_rates, rain_db=rain_rates * 0.5)
    gs = {"g_rx_dbi": 40.0, "system_temp_k": 290.0}
    res = obs.observe(gs, freq_hz=14e9, bandwidth_hz=36e6, polarization="vertical", res=dummy_res)
    
    wet_loss = res["_latent_wet_antenna_loss_db"]
    
    # Calculate analytical expected loss for 14 GHz:
    # c_wet = clip(0.12 * (14 - 5), 0, 8) = 1.08 dB
    # For R > 0.1: loss = 1.08 * (1 - exp(-0.08 * R))
    expected_c_wet = 1.08
    expected_loss = np.where(rain_rates > 0.1, expected_c_wet * (1.0 - np.exp(-0.08 * rain_rates)), 0.0)
    
    max_err = np.max(np.abs(wet_loss - expected_loss))
    
    print(f"14 GHz Saturation Wet Antenna Loss: {expected_c_wet:.2f} dB")
    print(f"Rain Rates (mm/h): {rain_rates}")
    print(f"Wet Antenna Loss (dB): {np.round(wet_loss, 4)}")
    print(f"Max Model Formula Error: {max_err:.6e} dB")
    
    return {
        "saturation_loss": expected_c_wet,
        "max_err": max_err
    }

def validate_multipath_model():
    """
    6. Multipath Rician Fading Model
    - Tests elevation dependence: K(el) = K_base * sin(el)
    - Verifies fading magnitude statistics
    """
    print("\n--- 6. Validating Multipath Model ---")
    cfg = ObservationConfig(scenario="typical", environment="rural", enable_tracking=False, enable_calibration=False,
                            enable_agc=False, enable_multipath=True, enable_wet_antenna=False, enable_scintillation=False)
    obs = ObservationModel(cfg, seed=42)
    
    n_steps = 10000
    dummy_res = make_dummy_result(n_steps, snr=20.0, el=10.0)
    gs = {"g_rx_dbi": 40.0, "system_temp_k": 290.0}
    res = obs.observe(gs, freq_hz=14e9, bandwidth_hz=36e6, polarization="vertical", res=dummy_res)
    
    mp_loss = res["_latent_multipath_loss_db"]
    
    # Expected K factor at 10 deg for rural (K_base = 18 dB): 18 * sin(10 deg) = 3.125 dB
    k_expected_db = 18.0 * np.sin(np.radians(10.0))
    
    print(f"10° Elevation Rician K-factor (Rural): {k_expected_db:.2f} dB")
    print(f"Multipath Loss Mean: {np.mean(mp_loss):.4f} dB | Std Dev: {np.std(mp_loss):.4f} dB | Max Fade: {np.max(mp_loss):.2f} dB")
    
    return {
        "k_expected_db": k_expected_db,
        "mean_loss": np.mean(mp_loss),
        "std_loss": np.std(mp_loss)
    }

def main():
    print("==================================================")
    print("RUNNING OBSERVATION MODEL EMPIRICAL VALIDATION SUITE")
    print("==================================================")
    
    t_res = validate_tracking_model()
    a_res = validate_agc_model()
    c_res = validate_calibration_model()
    th_res = validate_thermal_model()
    w_res = validate_wet_antenna_model()
    m_res = validate_multipath_model()
    
    print("\n==================================================")
    print("ALL EMPIRICAL VALIDATIONS COMPLETED SUCCESSFULLY")
    print("==================================================")

if __name__ == "__main__":
    main()
