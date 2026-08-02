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
    snr_arr = np.full(n_steps, snr) if isinstance(snr, (float, int)) else snr
    el_arr = np.full(n_steps, el) if isinstance(el, (float, int)) else el
    slant_arr = np.full(n_steps, slant) if isinstance(slant, (float, int)) else slant
    rain_arr = np.full(n_steps, rain) if isinstance(rain, (float, int)) else rain
    rain_db_arr = np.full(n_steps, rain_db) if isinstance(rain_db, (float, int)) else rain_db
    
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

def generate_all_plots():
    plots_dir = os.path.join(root_dir, "docs", "plots", "obs_validation")
    os.makedirs(plots_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Tracking Model: Autocorrelation & Elevation Jitter
    # -------------------------------------------------------------
    print("Generating 1. Tracking Plot...")
    cfg = ObservationConfig(scenario="typical", enable_tracking=True, enable_calibration=False,
                            enable_agc=False, enable_multipath=False, enable_wet_antenna=False, enable_scintillation=False)
    obs = ObservationModel(cfg, seed=42)
    n_steps = 5000
    res_obj = make_dummy_result(n_steps, el=30.0)
    gs = {"g_rx_dbi": 40.0, "antenna_diam_m": 1.2, "system_temp_k": 290.0}
    res = obs.observe(gs, freq_hz=14e9, bandwidth_hz=36e6, polarization="vertical", res=res_obj)
    
    track_err = res["_latent_tracking_error_deg"]
    
    # Empirical autocorrelation
    max_lag = 50
    autocorr = [np.corrcoef(track_err[:-lag], track_err[lag:])[0, 1] for lag in range(1, max_lag)]
    autocorr = [1.0] + autocorr
    theoretical_ac = [0.96**l for l in range(max_lag)]
    
    elevations = np.linspace(3, 90, 100)
    sigma_elev = 0.04 / np.sin(np.radians(elevations))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    ax1.plot(range(max_lag), autocorr, 'b-o', markersize=4, label='Empirical AR(1) Lag')
    ax1.plot(range(max_lag), theoretical_ac, 'r--', label='Theoretical Decay (ρ=0.96)')
    ax1.set_title('Servo Tracking Autocorrelation Decay')
    ax1.set_xlabel('Lag Steps (minutes)')
    ax1.set_ylabel('Autocorrelation coefficient')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(elevations, sigma_elev, 'g-', linewidth=2, label=r'$\sigma(\theta) = \frac{\sigma_0}{\sin\theta}$')
    ax2.set_title('Elevation-Dependent Pointing Noise Std Dev')
    ax2.set_xlabel('Elevation Angle (deg)')
    ax2.set_ylabel('Pointing Jitter σ (deg)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "val_tracking_pointing.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. AGC Step Response & ADC Quantization
    # -------------------------------------------------------------
    print("Generating 2. AGC & ADC Plot...")
    cfg = ObservationConfig(scenario="typical", enable_tracking=False, enable_calibration=False,
                            enable_agc=True, enable_multipath=False, enable_wet_antenna=False, enable_scintillation=False)
    obs = ObservationModel(cfg, seed=42)
    n_steps = 80
    
    # Create step input in physical power: 0 to 40 steps at -110 dBW, 40 to 80 steps at -125 dBW
    snr_step = np.full(n_steps, 25.0)
    snr_step[40:] = 10.0
    res_obj = make_dummy_result(n_steps, snr=snr_step, el=45.0)
    gs = {"g_rx_dbi": 40.0, "system_temp_k": 290.0}
    res = obs.observe(gs, freq_hz=14e9, bandwidth_hz=36e6, polarization="vertical", res=res_obj)
    
    obs_snr = res["observed_snr_db"]
    
    # Calculate continuous LPF curve without quantization
    alpha = 0.20
    lpf_cont = np.zeros(n_steps)
    lpf_cont[0] = snr_step[0]
    for idx in range(1, n_steps):
        lpf_cont[idx] = lpf_cont[idx-1] * (1.0 - alpha) + snr_step[idx] * alpha
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    ax1.plot(snr_step, 'k--', label='Physical Input SNR Step', alpha=0.7)
    ax1.plot(lpf_cont, 'r-', label=r'First-order LPF ($\alpha_{AGC}=0.20$)', linewidth=2)
    ax1.set_title('AGC Variable Gain Settling Step Response')
    ax1.set_xlabel('Time Step (minutes)')
    ax1.set_ylabel('SNR (dB)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(obs_snr[35:60], 'b-s', label='Quantized Output (LSB=0.05 dB)', markersize=4)
    ax2.plot(lpf_cont[35:60], 'r--', label='Continuous LPF Level', alpha=0.8)
    ax2.set_title('ADC Uniform Power Quantization Steps')
    ax2.set_xlabel('Time Step Index')
    ax2.set_ylabel('Observed SNR (dB)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "val_agc_step_quant.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Calibration Drift & PSD
    # -------------------------------------------------------------
    print("Generating 3. Calibration Plot...")
    cfg = ObservationConfig(scenario="typical", enable_tracking=False, enable_calibration=True,
                            enable_agc=False, enable_multipath=False, enable_wet_antenna=False, enable_scintillation=False)
    obs = ObservationModel(cfg, seed=123)
    n_steps = 10000
    res_obj = make_dummy_result(n_steps, el=45.0)
    gs = {"g_rx_dbi": 40.0, "system_temp_k": 290.0}
    res = obs.observe(gs, freq_hz=14e9, bandwidth_hz=36e6, polarization="vertical", res=res_obj)
    
    cal_err = res["_latent_calibration_error_db"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    t_hours = np.arange(n_steps) / 60.0
    ax1.plot(t_hours, cal_err, 'purple', linewidth=1, alpha=0.85)
    ax1.set_title('LNB Gain Calibration Drift Trajectory (OU + Aging)')
    ax1.set_xlabel('Time (Hours)')
    ax1.set_ylabel('Calibration Bias / Drift (dB)')
    ax1.grid(True, alpha=0.3)
    
    freqs, psd = scipy.signal.welch(cal_err, fs=1/60.0, nperseg=1024)
    ax2.loglog(freqs[1:], psd[1:], color='darkgreen', linewidth=1.5, label='Empirical Welch PSD')
    ax2.set_title('Calibration Drift Power Spectral Density (PSD)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density (dB²/Hz)')
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "val_calibration_ou_drift.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Thermal EIRP Orbital Dynamics
    # -------------------------------------------------------------
    print("Generating 4. Solar Thermal Plot...")
    cfg = ObservationConfig(scenario="typical", enable_tracking=False, enable_calibration=False,
                            enable_agc=False, enable_multipath=False, enable_wet_antenna=False, enable_scintillation=False)
    obs = ObservationModel(cfg, seed=42)
    n_steps = 192 # 2 full orbits (96 min each)
    res_obj = make_dummy_result(n_steps, el=45.0)
    gs = {"sat_ant_gain_dbi": 17.0, "system_temp_k": 290.0}
    res = obs.observe(gs, freq_hz=14e9, bandwidth_hz=36e6, polarization="vertical", res=res_obj)
    
    sat_eirp = res["_latent_sat_eirp_actual_dbw"]
    
    dt = 60.0
    t = np.arange(n_steps)
    solar_angle = 2.0 * np.pi * (t * dt) / 5760.0
    is_illuminated = np.sin(solar_angle) >= -0.1
    solar_flux = np.where(is_illuminated, 1361.0, 0.0)
    
    sigma_sb = 5.670374419e-8
    t_steady = ((0.70 * solar_flux) / (2.0 * 0.85 * sigma_sb) + 3.0**4)**0.25
    t_panel = np.zeros(n_steps)
    t_panel[0] = t_steady[0]
    alpha_thermal = dt / 300.0
    for idx in range(1, n_steps):
        t_panel[idx] = t_panel[idx-1] * (1.0 - alpha_thermal) + t_steady[idx] * alpha_thermal
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    t_min = t * dt / 60.0
    ax1.plot(t_min, t_panel, 'r-', label='Panel Temp T_panel(t)', linewidth=2)
    ax1.plot(t_min, t_steady, 'k:', label='Radiative Equilibrium T_eq', alpha=0.5)
    ax1.set_title('Spacecraft Solar Array Temperature Orbital Cycle')
    ax1.set_xlabel('Orbit Time (Minutes)')
    ax1.set_ylabel('Temperature (K)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(t_min, sat_eirp, 'b-', label='Actual EIRP (dBW)', linewidth=2)
    ax2.set_title('TWTA Temperature-Coupled Transmit EIRP Ripple')
    ax2.set_xlabel('Orbit Time (Minutes)')
    ax2.set_ylabel('Satellite EIRP (dBW)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "val_solar_thermal_eirp.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Wet Antenna Attenuation Curves
    # -------------------------------------------------------------
    print("Generating 5. Wet Antenna Plot...")
    r_grid = np.linspace(0, 100, 200)
    
    # c_wet for 14, 20, 30 GHz
    c_14 = np.clip(0.12 * (14.0 - 5.0), 0.0, 8.0) # 1.08 dB
    c_20 = np.clip(0.12 * (20.0 - 5.0), 0.0, 8.0) # 1.80 dB
    c_30 = np.clip(0.12 * (30.0 - 5.0), 0.0, 8.0) # 3.00 dB
    
    loss_14 = np.where(r_grid > 0.1, c_14 * (1.0 - np.exp(-0.08 * r_grid)), 0.0)
    loss_20 = np.where(r_grid > 0.1, c_20 * (1.0 - np.exp(-0.08 * r_grid)), 0.0)
    loss_30 = np.where(r_grid > 0.1, c_30 * (1.0 - np.exp(-0.08 * r_grid)), 0.0)
    
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(r_grid, loss_14, 'b-', label='Ku-Band (14 GHz)', linewidth=2)
    plt.plot(r_grid, loss_20, 'g--', label='Ka-Band Downlink (20 GHz)', linewidth=2)
    plt.plot(r_grid, loss_30, 'r-.', label='Ka-Band Uplink (30 GHz)', linewidth=2)
    plt.title('Wet Antenna Radome Water Film Loss Saturation Curves')
    plt.xlabel('Rainfall Rate R (mm/h)')
    plt.ylabel('Radome Attenuation L_wet (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "val_wet_antenna_loss.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 6. Multipath Rician Fading Distribution
    # -------------------------------------------------------------
    print("Generating 6. Multipath Rician Plot...")
    cfg = ObservationConfig(scenario="typical", environment="rural", enable_tracking=False, enable_calibration=False,
                            enable_agc=False, enable_multipath=True, enable_wet_antenna=False, enable_scintillation=False)
    obs = ObservationModel(cfg, seed=42)
    n_steps = 10000
    res_obj_5 = make_dummy_result(n_steps, el=5.0)
    res_obj_45 = make_dummy_result(n_steps, el=45.0)
    gs = {"g_rx_dbi": 40.0, "system_temp_k": 290.0}
    
    res_5 = obs.observe(gs, freq_hz=14e9, bandwidth_hz=36e6, polarization="vertical", res=res_obj_5)
    res_45 = obs.observe(gs, freq_hz=14e9, bandwidth_hz=36e6, polarization="vertical", res=res_obj_45)
    
    mp_5 = res_5["_latent_multipath_loss_db"]
    mp_45 = res_45["_latent_multipath_loss_db"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    ax1.hist(mp_5, bins=60, density=True, color='crimson', alpha=0.6, label=r'5° Elevation ($K_{dB} = 1.57$ dB)')
    ax1.set_title('Multipath Loss PDF at Low Elevation (5°)')
    ax1.set_xlabel('Multipath Loss (dB)')
    ax1.set_ylabel('Probability Density')
    ax1.set_xlim(0, 15)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.hist(mp_45, bins=60, density=True, color='navy', alpha=0.6, label=r'45° Elevation ($K_{dB} = 12.73$ dB)')
    ax2.set_title('Multipath Loss PDF at High Elevation (45°)')
    ax2.set_xlabel('Multipath Loss (dB)')
    ax2.set_ylabel('Probability Density')
    ax2.set_xlim(0, 5)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "val_multipath_rician.png"), dpi=300)
    plt.close()
    
    print("All plots generated successfully in:", plots_dir)

if __name__ == "__main__":
    import scipy.signal
    generate_all_plots()
