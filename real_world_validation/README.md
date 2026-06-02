# Real World Validation - Rain Models

This folder contains scripts to validate the simulation engine's rain models against real-world datasets and alternative standards.

## NASA GPM vs ITU-R Comparison

The script `validate_rain_nasa_gpm.py` compares the simulator's implementation of:
1. **ITU-R P.837-7**: Climatological rain statistics.
2. **ITU-R P.1853-2**: Time series synthesis (Maseng-Bakken model).

Against **NASA GPM IMERG** derived statistics for Delhi.

## SatNOGS SGP4 Validation

The script `validate_sgp4_satnogs.py` verifies the orbital propagation accuracy by comparing predicted satellite elevations against real-world observations from the **SatNOGS Network**.

### Why SatNOGS?
SatNOGS provides a global network of ground stations that record satellite passes. By comparing the simulator's SGP4 output with verified observation metadata (start/end times, ground station location, and reported peak elevation), we ensure that:
1. The SGP4 kernels are correctly integrated.
2. Coordinate frame transformations (TEME to ECEF) are accurate.
3. Time synchronization (Julian Date handling) is correct.

### Running the Validation
```bash
python3 real_world_validation/validate_sgp4_satnogs.py
```

### Outputs
- `real_world_validation/plots/val_sgp4_satnogs.png`: Comparison of predicted elevation curve vs. reality.

### Why NASA GPM?
Recent research (e.g., studies at Delhi Earth Station) indicates that ITU-R P.837 often underestimates peak rainfall rates in subtropical monsoon regions like Delhi. NASA's Global Precipitation Measurement (GPM) constellation provides more accurate estimates of convective rain intensities ($R_{0.01}$), often showing values 40-100% higher than ITU maps.

### Running the Validation
```bash
python3 real_world_validation/validate_rain_nasa_gpm.py
```

### Outputs
- `rain_comparison_table.csv`: Comparison of $R_1$, $R_{0.1}$, and $R_{0.01}$ quantiles.
- `rain_comparison_graph.png`: 
    - **Top**: CCDF (Exceedance Probability) plot comparing the distributions.
    - **Bottom**: Sample time series showing temporal correlation and peak intensity differences.

### Findings
- The simulator currently captures the log-normal distribution but may show lower tails than theoretical targets due to the finite coherence time and Markov-chain reset behavior.
- NASA GPM reference highlights the need for higher fade margins in tropical regions compared to standard ITU-R maps.
