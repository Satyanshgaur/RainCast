import numpy as np
import time

from satlinksim.config import config
from satlinksim.domain.interfaces import RainModel
from satlinksim.infrastructure.metrics import RAIN_GENERATION_TIME

TAU_COHERENCE_S = config.rain.tau_c

def _simulate_rain_kernel(n_steps, n_stations, rho, mu, sigma, p_onset, p_clear, force_rain, ln_R, raining):
    """
    Pure NumPy implementation of Maseng-Bakken rain process.
    """
    start_time = time.time()
    rates = np.zeros((n_steps, n_stations))
    
    # Work with arrays internally
    curr_ln_R = np.atleast_1d(ln_R).astype(float)
    curr_raining = np.atleast_1d(raining).astype(bool)
    
    for t in range(n_steps):
        if not force_rain:
            onset_roll = np.random.random(n_stations)
            clear_roll = np.random.random(n_stations)
            
            onset_mask = (~curr_raining) & (onset_roll < p_onset)
            curr_raining[onset_mask] = True
            curr_ln_R[onset_mask] = mu[onset_mask]
            
            clear_mask = (curr_raining) & (clear_roll < p_clear)
            curr_raining[clear_mask] = False
        
        noise = np.random.standard_normal(n_stations)
        curr_ln_R[:] = (rho * curr_ln_R
                        + np.sqrt(1 - rho**2) * sigma * noise
                        + (1 - rho) * mu)
        
        current_rates = np.exp(curr_ln_R)
        current_rates[current_rates > 150.0] = 150.0
        rates[t] = np.where(curr_raining, current_rates, 0.0)
                
    RAIN_GENERATION_TIME.observe(time.time() - start_time)
    return rates, curr_ln_R, curr_raining

class CorrelatedRainProcess(RainModel):
    def __init__(self, gs, dt_s, tau_c=TAU_COHERENCE_S,
                 force_rain=False, rain_rate_scale=1.0):
        if isinstance(gs, dict):
            gs = [gs]
        
        self.n_stations = len(gs)
        self.dt_s = dt_s
        self.force_rain = force_rain
        
        self.sigma = np.zeros(self.n_stations)
        self.mu = np.zeros(self.n_stations)
        self.rho = np.zeros(self.n_stations)
        self._p_onset = np.zeros(self.n_stations)
        self._p_clear = np.zeros(self.n_stations)
        
        for i, station in enumerate(gs):
            p = station["itu_rain"]
            R001 = max(p["R001"] * rain_rate_scale, 0.1)
            R01  = max(p["R01"]  * rain_rate_scale, 0.05)
            P_rain = p["P_rain"]
            
            _z001, _z01 = 3.0902, 2.3263
            self.sigma[i] = (np.log(R001) - np.log(R01)) / (_z001 - _z01)
            self.mu[i]    = np.log(R01) - _z01 * self.sigma[i]
            self.rho[i]   = np.exp(-dt_s / tau_c)
            
            mean_rain_dur_s  = tau_c
            mean_clear_dur_s = tau_c * (1 - P_rain) / (P_rain + 1e-9)
            self._p_onset[i] = 1 - np.exp(-dt_s / mean_clear_dur_s)
            self._p_clear[i] = 1 - np.exp(-dt_s / mean_rain_dur_s)
        
        self.ln_R = self.mu.copy()
        self.raining = np.zeros(self.n_stations, dtype=np.bool_)
        if self.force_rain:
            self.raining[:] = True

    def generate_batch(self, n_steps):
        rates, new_ln_R, new_raining = _simulate_rain_kernel(
            n_steps, self.n_stations, self.rho, self.mu, self.sigma, 
            self._p_onset, self._p_clear, self.force_rain,
            self.ln_R, self.raining
        )
        self.ln_R = new_ln_R
        self.raining = new_raining
        return rates

    def step(self):
        res = self.generate_batch(1)
        return res[0, 0]

    @property
    def ln_R_val(self):
        """Helper to get a scalar value for ln_R in single-station mode."""
        if self.n_stations == 1:
            return float(self.ln_R[0])
        return self.ln_R
