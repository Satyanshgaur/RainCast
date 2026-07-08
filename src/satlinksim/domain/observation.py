import numpy as np
from satlinksim.domain.models import StationResult

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
       - Wet antenna attenuation (radome loss scaling with rain rate and frequency).
       - Polarization mismatch loss.
       - Multipath fading (Rician fading scaling with elevation).
       - Receiver AGC response time (lag) and ADC power quantization.
    6. Uncertainty Estimator:
       - Output includes observed SNR, uncertainty estimate, and calibration state.
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def observe(self, gs: dict, freq_hz: float, bandwidth_hz: float, polarization: str, res: StationResult) -> dict:
        n_steps = len(res.snr_series)
        freq_ghz = freq_hz / 1e9
        
        # Time steps (1 minute resolution)
        dt = 60.0 # seconds
        t = np.arange(n_steps)
        
        # 1. Satellite Hardware (EIRP) via Solar Thermal TWTA Chain
        # Orbit period: LEO ~96 mins (5760s), GEO 24h (86400s). We default to 96 mins.
        orbit_period_s = 5760.0
        # Orbit phase / solar angle relative to panels normal
        solar_angle_rad = 2.0 * np.pi * (t * dt) / orbit_period_s
        # projected solar flux (W/m^2), including earth shadow eclipse (when sin < -0.1)
        # assuming sun-tracking panels with a small tracking error
        is_illuminated = np.sin(solar_angle_rad) >= -0.1
        solar_tracking_error = self.rng.normal(0.0, 0.05, size=n_steps)
        solar_flux = np.where(is_illuminated, 1361.0 * np.cos(solar_tracking_error), 0.0)
        
        # Solar panel temperature model (Thermal mass transient RC filter)
        # Steady-state temp based on radiative equilibrium (Stefan-Boltzmann)
        sigma_sb = 5.670374419e-8
        absorptivity = 0.70
        emissivity = 0.85
        t_cosmic = 3.0 # Cosmic microwave background
        # T_steady = ((absorptivity * SolarFlux) / (2 * emissivity * sigma_sb) + T_cosmic^4)^(1/4)
        t_steady = ((absorptivity * solar_flux) / (2.0 * emissivity * sigma_sb) + t_cosmic**4)**0.25
        
        # Thermal lag (tau_thermal ~ 300 seconds)
        t_panel = np.zeros(n_steps)
        t_panel[0] = t_steady[0]
        alpha_thermal = dt / 300.0
        for idx in range(1, n_steps):
            t_panel[idx] = t_panel[idx-1] * (1.0 - alpha_thermal) + t_steady[idx] * alpha_thermal
            
        # TWTA efficiency (decreases as temperature increases)
        eta_nominal = 0.55
        gamma_temp = 0.0008 # efficiency drop per Kelvin above 290K
        eta_twta = eta_nominal * (1.0 - gamma_temp * (t_panel - 290.0))
        eta_twta = np.clip(eta_twta, 0.10, 0.75)
        
        # TWTA Power output (Watts)
        p_dc_watts = 75.0 # Nominal 75 W solar cell DC bus allotment
        p_rf_watts = p_dc_watts * eta_twta
        p_rf_dbw = 10.0 * np.log10(p_rf_watts)
        
        # Actual Transmit EIRP = P_rf_dbw + Antenna Gain + small structural ageing random walk
        sat_ant_gain = gs.get("sat_ant_gain_dbi", 17.0) # standard satellite transmit gain
        sat_eirp_rw = np.cumsum(self.rng.normal(0, 0.001, size=n_steps)) # slow satellite hardware decay
        sat_eirp_actual = p_rf_dbw + sat_ant_gain + sat_eirp_rw
        
        # 2. Receiver Hardware (Noise Floor)
        # System Temperature has a measurement bias and sensor noise
        t_sys_nominal = gs.get("system_temp_k", 290.0)
        t_sys_bias = self.rng.normal(0.0, 4.0)
        t_sys_noise = self.rng.normal(0.0, 1.5, size=n_steps)
        t_sys_actual = np.maximum(t_sys_nominal + t_sys_bias + t_sys_noise, 15.0)
        
        # True Noise Floor
        k_b = 1.380649e-23
        noise_floor_true = 10.0 * np.log10(k_b * t_sys_actual * bandwidth_hz)
        
        # Estimated Noise Floor = True Noise Floor + Measurement Error (AR1 process representing sensor uncertainty)
        ar_noise_meas = 0.95
        noise_meas_err = np.zeros(n_steps)
        noise_meas_err[0] = self.rng.normal(0.0, 0.15)
        for idx in range(1, n_steps):
            noise_meas_err[idx] = ar_noise_meas * noise_meas_err[idx-1] + self.rng.normal(0.0, 0.15 * np.sqrt(1.0 - ar_noise_meas**2))
        noise_floor_estimated = noise_floor_true + noise_meas_err
        
        # 3. Tracking Errors (Elevation-dependent)
        # Tracking precision degrades at low elevation due to refraction jitter and ground clutter
        true_el_deg = np.array(res.elevation_series)
        true_slant_km = np.array(res.slant_range_series)
        
        # Tracking sigma proportional to 1 / sin(elevation)
        el_clamped = np.maximum(true_el_deg, 3.0)
        sigma_tracking = 0.04 / np.sin(np.radians(el_clamped)) # degrees
        
        # AR(1) process for Azimuth and Elevation tracking offsets
        ar_track = 0.96
        e_az = np.zeros(n_steps)
        e_el = np.zeros(n_steps)
        
        # Generate correlated step noise scaling with the elevation-dependent sigma
        noise_step_az = self.rng.normal(0, 1, size=n_steps)
        noise_step_el = self.rng.normal(0, 1, size=n_steps)
        e_az[0] = sigma_tracking[0] * noise_step_az[0]
        e_el[0] = sigma_tracking[0] * noise_step_el[0]
        for idx in range(1, n_steps):
            e_az[idx] = ar_track * e_az[idx-1] + np.sqrt(1.0 - ar_track**2) * sigma_tracking[idx] * noise_step_az[idx]
            e_el[idx] = ar_track * e_el[idx-1] + np.sqrt(1.0 - ar_track**2) * sigma_tracking[idx] * noise_step_el[idx]
            
        # Total pointing error angle theta
        theta_error = np.sqrt( (e_az * np.cos(np.radians(true_el_deg)))**2 + e_el**2 )
        
        # Pointing Loss (dB) using dynamic 3dB beamwidth
        ant_diam = gs.get("antenna_diam_m", 1.2)
        theta_3dB = (70.0 * 2.99792e8) / (freq_hz * ant_diam)
        pointing_loss = 12.0 * (theta_error / theta_3dB)**2
        
        # Reported geometry observations (distorted by tracking resolver errors)
        obs_el_deg = np.clip(true_el_deg + e_el, 0.0, 90.0)
        ranging_noise = self.rng.normal(0.0, 0.015, size=n_steps) # 15 meters ranging error
        obs_slant_km = np.maximum(true_slant_km + ranging_noise, 0.1)
        
        # 4. Calibration Drift (Random Walk + very low-frequency AR process)
        # Slow thermal drift (AR process with 4-hour correlation time)
        ar_cal = 0.9995
        cal_slow_drift = np.zeros(n_steps)
        cal_slow_drift[0] = self.rng.normal(0, 0.15)
        for idx in range(1, n_steps):
            cal_slow_drift[idx] = ar_cal * cal_slow_drift[idx-1] + self.rng.normal(0, 0.15 * np.sqrt(1.0 - ar_cal**2))
            
        # Age-related calibration decay (Random Walk)
        cal_aging = np.cumsum(self.rng.normal(0, 0.0015, size=n_steps))
        
        cal_power_bias = self.rng.normal(0.0, 0.15)
        cal_high_freq_noise = self.rng.normal(0.0, 0.03, size=n_steps)
        total_cal_error = cal_power_bias + cal_slow_drift + cal_aging + cal_high_freq_noise
        
        # 5. Missing Impairments
        # A. Wet Antenna Attenuation (radome loss)
        # Ka-band is significantly more affected than Ku-band
        true_rain = np.array(res.rain_series)
        # Wet antenna loss coefficient scales with frequency
        c_wet = np.clip(0.12 * (freq_ghz - 5.0), 0.1, 4.0)
        wet_antenna_loss = c_wet * (1.0 - np.exp(-0.08 * true_rain))
        
        # B. Polarization Mismatch Loss
        # Polarization alignment offset (typically 1.5 - 3.0 degrees + tiny tracking jitter)
        pol_offset = 2.0 # degrees
        pol_jitter = self.rng.normal(0.0, 0.1, size=n_steps)
        theta_pol = np.radians(pol_offset + pol_jitter)
        polarization_loss = -20.0 * np.log10(np.cos(theta_pol))
        
        # C. Multipath Fading (Elevation-dependent Rician fading model)
        # Fading envelope computed dynamically
        k_fading_db = 6.0 + 0.22 * true_el_deg # higher K at high elevation, low fading
        k_fading = 10.0**(k_fading_db / 10.0)
        
        # Complex channels
        cn_real = self.rng.normal(0.0, 1.0, size=n_steps)
        cn_imag = self.rng.normal(0.0, 1.0, size=n_steps)
        direct_comp = np.sqrt(k_fading / (k_fading + 1.0))
        diffuse_comp = np.sqrt(1.0 / (2.0 * (k_fading + 1.0))) * (cn_real + 1j * cn_imag)
        fading_magnitude = np.abs(direct_comp + diffuse_comp)
        multipath_loss = -20.0 * np.log10(np.clip(fading_magnitude, 1e-4, 5.0))
        
        # D. Net Physical Received Power at ground station LNA feed
        true_fspl = np.array(res.path_loss) if isinstance(res.path_loss, np.ndarray) else np.full(n_steps, res.path_loss)
        true_gas = np.array(res.gas_loss) if isinstance(res.gas_loss, np.ndarray) else np.full(n_steps, res.gas_loss)
        true_rain_attn = np.array(res.rain_db_series)
        true_scint = np.array(res.scint_series)
        
        rx_gain_nominal = gs.get("g_rx_dbi", 40.0)
        rx_gain_actual = rx_gain_nominal + self.rng.normal(0, 0.05) # small receiver construction bias
        
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
        # AGC response lag (first-order low-pass filter on power level)
        prx_agc = np.zeros(n_steps)
        prx_agc[0] = prx_physical[0]
        alpha_agc = 0.20 # AGC response coefficient
        for idx in range(1, n_steps):
            prx_agc[idx] = prx_agc[idx-1] * (1.0 - alpha_agc) + prx_physical[idx] * alpha_agc
            
        # ADC Quantization (receiver powers are digitized, e.g. LSB = 0.05 dB resolution)
        adc_lsb = 0.05
        prx_observed = np.round((prx_agc + total_cal_error) / adc_lsb) * adc_lsb
        
        # Net observed SNR = Quantized Power - Estimated Noise Floor
        obs_snr = prx_observed - noise_floor_estimated
        
        # 6. Uncertainty Estimator
        # Thermal Jitter: higher uncertainty at low SNR
        sigma_thermal = 10.0**(-np.clip(obs_snr, -5.0, 30.0) / 20.0)
        # Tracking Jitter uncertainty: higher at low elevation
        sigma_track_est = 0.12 / np.sin(np.radians(el_clamped))
        # Calibration state drift uncertainty (simulated days since calibration)
        # We model calibration drift uncertainty as slowly growing over time
        days_since_cal = int(self.rng.integers(0, 30)) # static state for the run
        sigma_cal = 0.05 + 0.01 * days_since_cal
        
        obs_snr_uncertainty = np.clip(np.sqrt(sigma_thermal**2 + sigma_track_est**2 + sigma_cal**2), 0.1, 3.0)
        cal_state = 1.0 if days_since_cal < 10 else (0.5 if days_since_cal < 22 else 0.0) # 1 = Clean, 0.5 = Moderate, 0 = Needs Recalibration
        
        # Assemble Telemetry dictionary
        # Excludes all latent variables ( E.g., pointing_loss_db, tracking_error_deg, cal_error_db, true_snr_db )
        # which would allow machine learning models to "cheat"
        obs_telemetry = {
            "observed_snr_db": obs_snr,
            "observed_elevation_deg": obs_el_deg,
            "observed_slant_range_km": obs_slant_km,
            "observed_snr_uncertainty_db": obs_snr_uncertainty,
            "calibration_state": np.full(n_steps, cal_state),
            
            # Latent variables are preserved in a separate dict key, 
            # NOT placed in the top-level so that they are never saved into telemetry columns.
            "_latent_sat_eirp_actual_dbw": sat_eirp_actual,
            "_latent_pointing_loss_db": pointing_loss,
            "_latent_tracking_error_deg": theta_error,
            "_latent_wet_antenna_loss_db": wet_antenna_loss,
            "_latent_multipath_loss_db": multipath_loss,
            "_latent_calibration_error_db": total_cal_error
        }
        
        return obs_telemetry
