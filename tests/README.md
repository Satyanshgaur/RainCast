# Satellite Link Simulator Tests

This directory contains the test suite for the Satellite Link Simulator.

## Running Tests

To run the tests, you need `pytest` and `numpy` installed. Run the following command from the project root:

```bash
python3 -m pytest tests/
```

## Test Structure

### `test_physics_invariants.py`
Verifies that the core physics models adhere to fundamental invariants:
- **FSPL monotonic with distance**: Path loss must increase as distance increases.
- **Noise increases with bandwidth**: Thermal noise power must scale with bandwidth.
- **Rain attenuation increases with rain rate**: Specific attenuation must increase with rainfall intensity.
- **Low elevation increases slant path**: Geometrical slant range must increase as the elevation angle decreases.
- **AR(1) correlation**: The Maseng-Bakken rain process must preserve the specified temporal autocorrelation.

### `test_regression.py`
Ensures that the simulation remains stable and deterministic:
- **Deterministic seeds**: Using the same seed and start time must produce identical SNR and rain series.
- **Force rain flag**: Verifies that the `force_rain` parameter correctly overrides the stochastic onset model.
- **Summary statistics**: Verifies the correctness of mean, min, and standard deviation calculations.
