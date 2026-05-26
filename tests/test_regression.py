import pytest
import random
from datetime import datetime, timezone
from satellite_link_sim import simulate_station
from ground_stations import GROUND_STATIONS

def test_deterministic_seed():
    """Regression test: verify that the same seed produces identical SNR series."""
    gs = GROUND_STATIONS[0] # Delhi
    seed = 42
    n_steps = 10
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    
    res1 = simulate_station(gs, n_steps=n_steps, seed=seed, start_time=start_time)
    res2 = simulate_station(gs, n_steps=n_steps, seed=seed, start_time=start_time)
    
    assert res1.snr_series == res2.snr_series
    assert res1.rain_series == res2.rain_series

def test_different_seeds_different_results():
    """Verify that different seeds produce different results."""
    gs = GROUND_STATIONS[0]
    n_steps = 20
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    
    res1 = simulate_station(gs, n_steps=n_steps, seed=42, start_time=start_time)
    res2 = simulate_station(gs, n_steps=n_steps, seed=43, start_time=start_time)
    
    # It's theoretically possible but extremely unlikely they are identical
    assert res1.snr_series != res2.snr_series

def test_force_rain_flag():
    """Regression test: verify force_rain=True actually results in rain at every step."""
    gs = GROUND_STATIONS[0]
    n_steps = 20
    
    res = simulate_station(gs, n_steps=n_steps, force_rain=True)
    
    assert all(r > 0 for r in res.rain_series)
    assert res.rain_fraction == 1.0

def test_snr_summary_stats():
    """Verify that summary statistics are calculated correctly."""
    gs = GROUND_STATIONS[0]
    n_steps = 50
    res = simulate_station(gs, n_steps=n_steps, seed=123)
    
    import statistics
    assert res.snr_mean == statistics.mean(res.snr_series)
    assert res.snr_min == min(res.snr_series)
    assert res.snr_std == statistics.stdev(res.snr_series)
