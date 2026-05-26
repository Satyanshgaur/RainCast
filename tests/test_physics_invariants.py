import pytest
import math
import random
import statistics
import numpy as np
from satellite_link_sim import (
    fspl_db, 
    noise_power_dbw, 
    rain_attenuation_db, 
    geo_slant_range_km,
    geo_elevation_deg,
    CorrelatedRainProcess,
    itu_rain_coefficients,
    effective_path_length,
    itu_rain_height
)
from ground_stations import GROUND_STATIONS

def test_fspl_monotonic_with_distance():
    """FSPL monotonic with distance: verify that path loss increases as distance increases."""
    freq_hz = 14e9
    distances = [100, 1000, 10000, 40000, 50000]
    losses = [fspl_db(freq_hz, d) for d in distances]
    
    assert all(losses[i] < losses[i+1] for i in range(len(losses)-1))

def test_noise_increases_with_bandwidth():
    """Noise increases with bandwidth: verify that thermal noise power increases with bandwidth."""
    T_sys = 500
    bandwidths = [1e6, 10e6, 36e6, 100e6]
    noise_powers = [noise_power_dbw(T_sys, b) for b in bandwidths]
    
    assert all(noise_powers[i] < noise_powers[i+1] for i in range(len(noise_powers)-1))

def test_rain_attenuation_increases_with_rain_rate():
    """Rain attenuation increases with rain rate: verify that attenuation increases with mm/h."""
    # Setup constants for a typical Ku-band link
    freq_ghz = 14.0
    pol = "vertical"
    itu_k, itu_alpha = itu_rain_coefficients(freq_ghz, pol)
    
    # Using Delhi parameters for effective path
    lat = 28.6
    rain_h = itu_rain_height(lat)
    elevation = 35.0
    alt_km = 0.216
    eff_path = effective_path_length(elevation, rain_h, alt_km, itu_k)
    
    rain_rates = [0.1, 1, 5, 20, 50, 100]
    attenuations = [rain_attenuation_db(r, itu_k, itu_alpha, eff_path) for r in rain_rates]
    
    assert all(attenuations[i] < attenuations[i+1] for i in range(len(attenuations)-1))

def test_low_elevation_increases_slant_path():
    """Low elevation increases slant path: verify slant range is longer at lower elevations."""
    lat = 0.0
    lon = 0.0
    # Satellites at different longitudes to vary elevation
    sat_lons = [0.0, 10.0, 30.0, 60.0, 80.0] 
    
    results = []
    for sat_lon in sat_lons:
        el = geo_elevation_deg(lat, lon, sat_lon)
        slant = geo_slant_range_km(lat, lon, sat_lon)
        results.append((el, slant))
    
    # Sort by elevation descending
    results.sort(key=lambda x: x[0], reverse=True)
    
    # As elevation decreases, slant path should increase
    for i in range(len(results) - 1):
        assert results[i][0] > results[i+1][0] # elevation decreasing
        assert results[i][1] < results[i+1][1] # slant range increasing

def test_ar1_correlation_preserved():
    """AR(1) correlation approximately preserved: verify the empirical autocorrelation of the rain process."""
    random.seed(42)
    dt_s = 60
    tau_c = 300.0
    expected_rho = math.exp(-dt_s / tau_c)
    
    # Use Delhi station
    gs = GROUND_STATIONS[0]
    # force_rain=True to get enough samples for correlation
    proc = CorrelatedRainProcess(gs, dt_s=dt_s, tau_c=tau_c, force_rain=True)
    
    # Generate log-rain rate samples (the AR(1) process is linear in log domain)
    samples = []
    for _ in range(10000):
        # We need the internal ln_R state or we can take log of the output
        rate = proc.step()
        if rate > 0:
            samples.append(math.log(rate))
    
    # Calculate empirical autocorrelation at lag 1
    samples = np.array(samples)
    samples_centered = samples - np.mean(samples)
    autocorr = np.correlate(samples_centered, samples_centered, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    rho_empirical = autocorr[1] / autocorr[0]
    
    print(f"Expected rho: {expected_rho:.4f}, Empirical rho: {rho_empirical:.4f}")
    
    # Allow some tolerance for stochasticity
    assert pytest.approx(rho_empirical, abs=0.05) == expected_rho
