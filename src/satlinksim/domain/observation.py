import numpy as np
from satlinksim.domain.models import StationResult
from satlinksim.domain.link.itu_models import itu_rain_coefficients, itu_rain_height, effective_path_length, gaseous_absorption_db
from satlinksim.domain.link.budget import fspl_db, noise_power_dbw

class ObservationModel:
    """
    Observation Model Layer:
    Distorts the true Physical World state into observed telemetry by simulating:
    1. Satellite Hardware imperfections (EIRP bias and thermal drift, carrier frequency offset).
    2. Ground station Receiver Hardware imperfections (LNA gain bias/drift, LNA system temp noise).
    3. Tracking Errors:
       - Auto-correlated pointing error in Azimuth and Elevation.
       - Pointing loss based on frequency-dependent antenna beamwidth.
       - Sensor measurement noise on reported Elevation angle and Slant Range.
    4. Calibration Errors:
       - Power measurement bias, slow daily drift, and high-frequency thermal noise.
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def observe(self, gs: dict, freq_hz: float, bandwidth_hz: float, polarization: str, res: StationResult) -> dict:
        n_steps = len(res.snr_series)
        freq_ghz = freq_hz / 1e9
        
        # 1. Satellite Hardware Imperfections
        # EIRP has a constant bias and slow diurnal thermal drift + small random walk
        sat_eirp_bias = self.rng.normal(0.0, 0.3) # +/- 0.3 dB standard bias
        t = np.arange(n_steps)
        # Slow drift (6-hour period thermal cycles + random walk)
        sat_eirp_drift = 0.15 * np.sin(2 * np.pi * t / (360 * 60)) 
        # Random walk component
        rw_step = self.rng.normal(0, 0.005, size=n_steps)
        sat_eirp_rw = np.cumsum(rw_step)
        total_sat_eirp_error = sat_eirp_bias + sat_eirp_drift + sat_eirp_rw
        
        # Nominal/expected sat EIRP and actual physical sat EIRP
        sat_eirp_nominal = gs.get("eirp_dbw", 50.0)
        sat_eirp_actual = sat_eirp_nominal + total_sat_eirp_error
        
        # 2. Receiver Hardware Imperfections
        # LNA rx gain has bias + 12-hour thermal cycle drift + random walk
        rx_gain_bias = self.rng.normal(0.0, 0.2)
        rx_gain_drift = 0.1 * np.cos(2 * np.pi * t / (720 * 60))
        rx_gain_rw = np.cumsum(self.rng.normal(0, 0.003, size=n_steps))
        total_rx_gain_error = rx_gain_bias + rx_gain_drift + rx_gain_rw
        
        rx_gain_nominal = gs.get("g_rx_dbi", 40.0)
        rx_gain_actual = rx_gain_nominal + total_rx_gain_error
        
        # System Temperature has a measurement bias and sensor noise
        t_sys_nominal = gs.get("system_temp_k", 290.0)
        t_sys_bias = self.rng.normal(0.0, 5.0) # +/- 5 K bias
        t_sys_noise = self.rng.normal(0.0, 2.0, size=n_steps) # 2 K standard noise
        t_sys_actual = np.maximum(t_sys_nominal + t_sys_bias + t_sys_noise, 10.0)
        
        # Noise floor (actual vs nominal expected)
        noise_floor_actual = 10 * np.log10(1.380649e-23 * t_sys_actual * bandwidth_hz)
        noise_floor_nominal = 10 * np.log10(1.380649e-23 * t_sys_nominal * bandwidth_hz)
        
        # 3. Tracking Errors
        # GS Antenna Diameter (defaults to 1.2 meters)
        ant_diam = gs.get("antenna_diam_m", 1.2)
        # Antenna 3dB beamwidth: theta_3dB = 70 * c / (f * D) in degrees
        theta_3dB = (70.0 * 2.99792e8) / (freq_hz * ant_diam)
        
        # Simulate tracking errors in Azimuth (az) and Elevation (el) as AR(1) processes
        # to ensure temporal correlation
        ar_coeff = 0.98
        sigma_track = 0.08 # degrees standard deviation
        
        e_az = np.zeros(n_steps)
        e_el = np.zeros(n_steps)
        
        noise_az = self.rng.normal(0, np.sqrt(1.0 - ar_coeff**2) * sigma_track, size=n_steps)
        noise_el = self.rng.normal(0, np.sqrt(1.0 - ar_coeff**2) * sigma_track, size=n_steps)
        
        e_az[0] = noise_az[0]
        e_el[0] = noise_el[0]
        for idx in range(1, n_steps):
            e_az[idx] = ar_coeff * e_az[idx-1] + noise_az[idx]
            e_el[idx] = ar_coeff * e_el[idx-1] + noise_el[idx]
            
        # Pointing error angle: theta_error = sqrt(e_az^2 * cos^2(el) + e_el^2)
        true_el_deg = np.array(res.elevation_series)
        true_slant_km = np.array(res.slant_range_series)
        
        theta_error = np.sqrt( (e_az * np.cos(np.radians(true_el_deg)))**2 + e_el**2 )
        
        # Pointing Loss in dB
        pointing_loss = 12.0 * (theta_error / theta_3dB)**2
        
        # Measured/Observed geometry reported by tracking system
        # Encoder/sensor error added to reported tracking elevation and range
        obs_el_deg = true_el_deg + e_el
        # Ensure elevation is physically bounded
        obs_el_deg = np.clip(obs_el_deg, 0.0, 90.0)
        
        ranging_sensor_noise = self.rng.normal(0, 0.02, size=n_steps) # 20 meters error
        obs_slant_km = true_slant_km + ranging_sensor_noise
        obs_slant_km = np.maximum(obs_slant_km, 0.1)
        
        # 4. Calibration & Measurement Errors
        # The power meter measuring the received signal has calibration bias + daily thermal cycle drift
        cal_power_bias = self.rng.normal(0.0, 0.2)
        cal_power_drift = 0.08 * np.sin(2 * np.pi * t / (1440 * 60)) # diurnal cycle
        cal_power_noise = self.rng.normal(0.0, 0.05, size=n_steps)
        total_cal_error = cal_power_bias + cal_power_drift + cal_power_noise
        
        # Calculate True received power at antenna feed
        # In res: path_loss is FSPL, gas_loss is GasLoss, rain_db_series is RainAttn, scint_series is Scint
        true_fspl = np.array(res.path_loss) if isinstance(res.path_loss, np.ndarray) else np.full(n_steps, res.path_loss)
        true_gas = np.array(res.gas_loss) if isinstance(res.gas_loss, np.ndarray) else np.full(n_steps, res.gas_loss)
        true_rain_attn = np.array(res.rain_db_series)
        true_scint = np.array(res.scint_series)
        
        # Physical received power at antenna feed (incorporating actual hardware states and pointing loss)
        prx_physical = (sat_eirp_actual 
                        - true_fspl 
                        - true_gas 
                        - true_rain_attn 
                        - true_scint 
                        - pointing_loss 
                        + rx_gain_actual)
        
        # Observed (measured) received power at the receiver LNA output
        prx_observed = prx_physical + total_cal_error
        
        # Observed SNR measured by receiver: SNR_obs = Prx_observed - NoiseFloor_nominal
        obs_snr = prx_observed - noise_floor_nominal
        
        # Compile observed telemetry dictionary
        obs_telemetry = {
            "observed_snr_db": obs_snr,
            "observed_elevation_deg": obs_el_deg,
            "observed_slant_range_km": obs_slant_km,
            "pointing_loss_db": pointing_loss,
            "tracking_error_deg": theta_error,
            "sat_eirp_error_db": total_sat_eirp_error,
            "rx_gain_error_db": total_rx_gain_error,
            "calibration_error_db": total_cal_error
        }
        
        return obs_telemetry
