import numpy as np
from dataclasses import dataclass
from satlinksim.domain.models import StationResult

@dataclass
class ObservationConfig:
    scenario: str = "typical"  # "ideal", "typical", "severe"
    
    # Manual overrides for isolated impairment testing
    enable_scintillation: bool = True
    enable_tracking: bool = True
    enable_calibration: bool = True
    enable_agc: bool = True
    enable_multipath: bool = True
    enable_wet_antenna: bool = True
    
    # Multipath environment: "rural" (low reflections), "marine" (water reflections), "urban" (high reflections)
    environment: str = "rural"

class ObservationModel:
    """
    Observation Model Layer:
    Distorts the true Physical World state into observed telemetry by simulating:
    1. Satellite Hardware (EIRP):
       - Solar angle -> Solar panel temperature (thermal inertia) -> TWTA efficiency -> Transmit EIRP.
    2. Receiver Hardware (Noise Floor):
       - Estimated Noise Floor = True Noise Floor + Measurement Error.
       - True Noise Floor based on True system temperature (with thermal jitter).
    3. Tracking Errors (Elevation-dependent):
       - Auto-correlated pointing jitter where sigma is inversely proportional to sin(elevation).
       - Frequency-dependent pointing loss based on antenna beamwidth.
       - Tracking sensor noise on reported elevation/slant range.
    4. Calibration Errors:
       - Low-frequency AR process (Ornstein-Uhlenbeck) + Random Walk for slow drift.
    5. Atmospheric and Hardware Impairments:
       - Wet antenna attenuation (radome loss activating only if rain > 0.1 mm/h).
       - Polarization mismatch loss.
       - Multipath fading (degrading as elevation decreases, scaling by environment).
       - Receiver AGC response time (lag) and ADC power quantization.
    6. Uncertainty Estimator:
       - Output includes observed SNR, uncertainty estimate, and calibration state.
    """
    def __init__(self, config: ObservationConfig = None, seed=None):
        self.config = config or ObservationConfig()
        self.rng = np.random.default_rng(seed)

    def observe(self, gs: dict, freq_hz: float, bandwidth_hz: float, polarization: str, res: StationResult) -> dict:
        n_steps = len(res.snr_series)
        freq_ghz = freq_hz / 1e9
        
        # Time steps (1 minute resolution)
        dt = 60.0 # seconds
        t = np.arange(n_steps)
        
        # Load parameters based on scenario
        scenario = self.config.scenario.lower()
        
        if scenario == "ideal":
            sat_eirp_bias_std = 0.01
            sat_eirp_drift_std = 0.001
            rx_gain_bias_std = 0.01
            t_sys_bias_std = 0.1
            t_sys_noise_std = 0.1
            sigma_track_nominal = 0.001 # almost perfect tracking
            ranging_noise_std = 0.001
            cal_bias_std = 0.01
            cal_slow_drift_std = 0.001
            cal_aging_std = 0.0001
            adc_lsb = 0.001 # very high resolution
            alpha_agc = 1.0 # instantaneous AGC response
            pol_offset = 0.0
            pol_jitter_std = 0.001
            c_wet_multiplier = 0.0
            k_base_db = 40.0 # extremely high Rician K (no fading)
        elif scenario == "severe":
            sat_eirp_bias_std = 1.0
            sat_eirp_drift_std = 0.3
            rx_gain_bias_std = 0.8
            t_sys_bias_std = 15.0
            t_sys_noise_std = 5.0
            sigma_track_nominal = 0.15 # bad tracking loop
            ranging_noise_std = 0.1 # 100 meters error
            cal_bias_std = 0.5
            cal_slow_drift_std = 0.4
            cal_aging_std = 0.005
            adc_lsb = 0.20 # coarse quantization
            alpha_agc = 0.05 # slow AGC lag
            pol_offset = 5.0 # 5 degrees mismatch
            pol_jitter_std = 0.3
            c_wet_multiplier = 0.24
            
            # Multipath base K-factor based on environment
            env = self.config.environment.lower()
            if env == "urban":
                k_base_db = 4.0
            elif env == "marine":
                k_base_db = 8.0
            else:
                k_base_db = 12.0
        else: # typical
            sat_eirp_bias_std = 0.3
            sat_eirp_drift_std = 0.15
            rx_gain_bias_std = 0.2
            t_sys_bias_std = 5.0
            t_sys_noise_std = 2.0
            sigma_track_nominal = 0.04
            ranging_noise_std = 0.015
            cal_bias_std = 0.20
            cal_slow_drift_std = 0.15
            cal_aging_std = 0.0015
            adc_lsb = 0.05
            alpha_agc = 0.20
            pol_offset = 2.0
            pol_jitter_std = 0.1
            c_wet_multiplier = 0.12
            
            env = self.config.environment.lower()
            if env == "urban":
                k_base_db = 6.0
            elif env == "marine":
                k_base_db = 12.0
            else:
                k_base_db = 18.0

        # Apply global config overrides for isolated impairment experiments
        enable_tracking = self.config.enable_tracking and (scenario != "ideal")
        enable_calibration = self.config.enable_calibration and (scenario != "ideal")
        enable_agc = self.config.enable_agc and (scenario != "ideal")
        enable_multipath = self.config.enable_multipath and (scenario != "ideal")
        enable_wet_antenna = self.config.enable_wet_antenna and (scenario != "ideal")
        enable_scintillation = self.config.enable_scintillation
        
        # 1. Satellite Hardware (EIRP) via Solar Thermal TWTA Chain
        orbit_period_s = 5760.0
        solar_angle_rad = 2.0 * np.pi * (t * dt) / orbit_period_s
        is_illuminated = np.sin(solar_angle_rad) >= -0.1
        
        # If tracking panels is enabled
        solar_tracking_error = self.rng.normal(0.0, 0.05, size=n_steps)
        solar_flux = np.where(is_illuminated, 1361.0 * np.cos(solar_tracking_error), 0.0)
        
        # Solar panel temperature model (RC filter)
        sigma_sb = 5.670374419e-8
        absorptivity = 0.70
        emissivity = 0.85
        t_cosmic = 3.0
        t_steady = ((absorptivity * solar_flux) / (2.0 * emissivity * sigma_sb) + t_cosmic**4)**0.25
        
        t_panel = np.zeros(n_steps)
        t_panel[0] = t_steady[0]
        alpha_thermal = dt / 300.0
        for idx in range(1, n_steps):
            t_panel[idx] = t_panel[idx-1] * (1.0 - alpha_thermal) + t_steady[idx] * alpha_thermal
            
        # TWTA efficiency
        eta_nominal = 0.55
        gamma_temp = 0.0008
        eta_twta = eta_nominal * (1.0 - gamma_temp * (t_panel - 290.0))
        eta_twta = np.clip(eta_twta, 0.10, 0.75)
        
        p_dc_watts = 75.0
        p_rf_watts = p_dc_watts * eta_twta
        p_rf_dbw = 10.0 * np.log10(p_rf_watts)
        
        sat_ant_gain = gs.get("sat_ant_gain_dbi", 17.0)
        
        # EIRP physical errors
        if enable_calibration:
            sat_eirp_rw = np.cumsum(self.rng.normal(0, sat_eirp_drift_std * 0.01, size=n_steps))
            sat_eirp_actual = p_rf_dbw + sat_ant_gain + self.rng.normal(0.0, sat_eirp_bias_std) + sat_eirp_rw
        else:
            sat_eirp_actual = np.full(n_steps, p_rf_dbw + sat_ant_gain)
            
        # 2. Receiver Hardware (Noise Floor)
        t_sys_nominal = gs.get("system_temp_k", 290.0)
        if enable_calibration:
            t_sys_bias = self.rng.normal(0.0, t_sys_bias_std)
            t_sys_noise = self.rng.normal(0.0, t_sys_noise_std, size=n_steps)
            t_sys_actual = np.maximum(t_sys_nominal + t_sys_bias + t_sys_noise, 15.0)
        else:
            t_sys_actual = np.full(n_steps, t_sys_nominal)
            
        k_b = 1.380649e-23
        noise_floor_true = 10.0 * np.log10(k_b * t_sys_actual * bandwidth_hz)
        
        # Estimated Noise Floor
        if enable_calibration:
            ar_noise_meas = 0.95
            noise_meas_err = np.zeros(n_steps)
            noise_meas_err[0] = self.rng.normal(0.0, 0.15)
            for idx in range(1, n_steps):
                noise_meas_err[idx] = ar_noise_meas * noise_meas_err[idx-1] + self.rng.normal(0.0, 0.15 * np.sqrt(1.0 - ar_noise_meas**2))
            noise_floor_estimated = noise_floor_true + noise_meas_err
        else:
            noise_floor_estimated = noise_floor_true
            
        # 3. Tracking Errors (Elevation-dependent)
        true_el_deg = np.array(res.elevation_series)
        true_slant_km = np.array(res.slant_range_series)
        
        if enable_tracking:
            el_clamped = np.maximum(true_el_deg, 3.0)
            sigma_tracking = sigma_track_nominal / np.sin(np.radians(el_clamped))
            
            ar_track = 0.96
            e_az = np.zeros(n_steps)
            e_el = np.zeros(n_steps)
            
            noise_step_az = self.rng.normal(0, 1, size=n_steps)
            noise_step_el = self.rng.normal(0, 1, size=n_steps)
            e_az[0] = sigma_tracking[0] * noise_step_az[0]
            e_el[0] = sigma_tracking[0] * noise_step_el[0]
            for idx in range(1, n_steps):
                e_az[idx] = ar_track * e_az[idx-1] + np.sqrt(1.0 - ar_track**2) * sigma_tracking[idx] * noise_step_az[idx]
                e_el[idx] = ar_track * e_el[idx-1] + np.sqrt(1.0 - ar_track**2) * sigma_tracking[idx] * noise_step_el[idx]
                
            theta_error = np.sqrt( (e_az * np.cos(np.radians(true_el_deg)))**2 + e_el**2 )
            
            ant_diam = gs.get("antenna_diam_m", 1.2)
            theta_3dB = (70.0 * 2.99792e8) / (freq_hz * ant_diam)
            pointing_loss = 12.0 * (theta_error / theta_3dB)**2
            
            obs_el_deg = np.clip(true_el_deg + e_el, 0.0, 90.0)
            ranging_noise = self.rng.normal(0.0, ranging_noise_std, size=n_steps)
            obs_slant_km = np.maximum(true_slant_km + ranging_noise, 0.1)
        else:
            pointing_loss = np.zeros(n_steps)
            theta_error = np.zeros(n_steps)
            obs_el_deg = true_el_deg.copy()
            obs_slant_km = true_slant_km.copy()
            sigma_tracking = np.zeros(n_steps)
            
        # 4. Calibration Drift
        if enable_calibration:
            ar_cal = 0.9995
            cal_slow_drift = np.zeros(n_steps)
            cal_slow_drift[0] = self.rng.normal(0, cal_slow_drift_std)
            for idx in range(1, n_steps):
                cal_slow_drift[idx] = ar_cal * cal_slow_drift[idx-1] + self.rng.normal(0, cal_slow_drift_std * np.sqrt(1.0 - ar_cal**2))
                
            cal_aging = np.cumsum(self.rng.normal(0, cal_aging_std, size=n_steps))
            cal_power_bias = self.rng.normal(0.0, cal_bias_std)
            cal_high_freq_noise = self.rng.normal(0.0, 0.03, size=n_steps)
            total_cal_error = cal_power_bias + cal_slow_drift + cal_aging + cal_high_freq_noise
        else:
            total_cal_error = np.zeros(n_steps)
            
        # 5. Missing Impairments
        # A. Wet Antenna Attenuation (radome loss) - ACTIVATES ONLY if Rain Rate > 0.1 mm/h
        true_rain = np.array(res.rain_series)
        if enable_wet_antenna:
            c_wet = np.clip(c_wet_multiplier * (freq_ghz - 5.0), 0.0, 8.0)
            # Only activate radome loss when raining (rain > 0.1 mm/h)
            wet_antenna_loss = np.where(true_rain > 0.1, c_wet * (1.0 - np.exp(-0.08 * true_rain)), 0.0)
        else:
            wet_antenna_loss = np.zeros(n_steps)
            
        # B. Polarization Mismatch Loss
        if enable_tracking:
            pol_jitter = self.rng.normal(0.0, pol_jitter_std, size=n_steps)
            theta_pol = np.radians(pol_offset + pol_jitter)
            polarization_loss = -20.0 * np.log10(np.cos(theta_pol))
        else:
            polarization_loss = np.zeros(n_steps)
            
        # C. Multipath Fading (Elevation and Environment Dependent Rician model)
        if enable_multipath:
            # K-factor degrades significantly at low elevation, almost none (very high K) at high elevation
            sin_el = np.sin(np.radians(np.maximum(true_el_deg, 3.0)))
            k_fading_db = k_base_db * sin_el
            k_fading = 10.0**(k_fading_db / 10.0)
            
            cn_real = self.rng.normal(0.0, 1.0, size=n_steps)
            cn_imag = self.rng.normal(0.0, 1.0, size=n_steps)
            direct_comp = np.sqrt(k_fading / (k_fading + 1.0))
            diffuse_comp = np.sqrt(1.0 / (2.0 * (k_fading + 1.0))) * (cn_real + 1j * cn_imag)
            fading_magnitude = np.abs(direct_comp + diffuse_comp)
            multipath_loss = -20.0 * np.log10(np.clip(fading_magnitude, 1e-4, 5.0))
        else:
            multipath_loss = np.zeros(n_steps)
            
        # D. Net Physical Received Power at ground station LNA feed
        true_fspl = np.array(res.path_loss) if isinstance(res.path_loss, np.ndarray) else np.full(n_steps, res.path_loss)
        true_gas = np.array(res.gas_loss) if isinstance(res.gas_loss, np.ndarray) else np.full(n_steps, res.gas_loss)
        true_rain_attn = np.array(res.rain_db_series)
        true_scint = np.array(res.scint_series) if enable_scintillation else np.zeros(n_steps)
        
        rx_gain_nominal = gs.get("g_rx_dbi", 40.0)
        rx_gain_actual = rx_gain_nominal + (self.rng.normal(0, rx_gain_bias_std) if enable_calibration else 0.0)
        
        prx_physical = (sat_eirp_actual 
                        - true_fspl 
                        - true_gas 
                        - true_rain_attn 
                        - true_scint 
                        - pointing_loss 
                        - wet_antenna_loss 
                        - polarization_loss 
                        - multipath_loss
                        + rx_gain_actual)
        
        # E. Receiver AGC and ADC Quantization
        if enable_agc:
            prx_agc = np.zeros(n_steps)
            prx_agc[0] = prx_physical[0]
            for idx in range(1, n_steps):
                prx_agc[idx] = prx_agc[idx-1] * (1.0 - alpha_agc) + prx_physical[idx] * alpha_agc
                
            prx_observed = np.round((prx_agc + total_cal_error) / adc_lsb) * adc_lsb
        else:
            prx_observed = prx_physical + total_cal_error
            
        # Net observed SNR = Quantized Power - Estimated Noise Floor
        obs_snr = prx_observed - noise_floor_estimated
        
        # 6. Uncertainty Estimator
        # Thermal Jitter
        sigma_thermal = 10.0**(-np.clip(obs_snr, -5.0, 30.0) / 20.0)
        # Tracking Jitter uncertainty
        el_clamped = np.maximum(true_el_deg, 3.0)
        sigma_track_est = (sigma_track_nominal / np.sin(np.radians(el_clamped))) if enable_tracking else 0.001
        # Calibration state drift uncertainty (simulated days since calibration)
        days_since_cal = int(self.rng.integers(0, 30)) if enable_calibration else 0
        sigma_cal = 0.05 + 0.01 * days_since_cal
        
        obs_snr_uncertainty = np.clip(np.sqrt(sigma_thermal**2 + sigma_track_est**2 + sigma_cal**2), 0.01, 5.0)
        cal_state = 1.0 if days_since_cal < 10 else (0.5 if days_since_cal < 22 else 0.0)
        
        # Assemble Telemetry dictionary
        obs_telemetry = {
            "observed_snr_db": obs_snr,
            "observed_elevation_deg": obs_el_deg,
            "observed_slant_range_km": obs_slant_km,
            "observed_snr_uncertainty_db": obs_snr_uncertainty,
            "calibration_state": np.full(n_steps, cal_state),
            
            # Latent variables (hidden from ML features)
            "_latent_sat_eirp_actual_dbw": sat_eirp_actual,
            "_latent_pointing_loss_db": pointing_loss,
            "_latent_tracking_error_deg": theta_error,
            "_latent_wet_antenna_loss_db": wet_antenna_loss,
            "_latent_multipath_loss_db": multipath_loss,
            "_latent_calibration_error_db": total_cal_error
        }
        
        return obs_telemetry
