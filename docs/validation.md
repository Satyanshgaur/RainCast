# Validation Methodology

The simulator includes an automated suite to ensure physical accuracy against ITU standards and analytical references. The following sections detail the validation of core physical models.

## 1. Free-Space Path Loss (ITU-R P.525)
Validated against the standard formula: $L_{fs} = 92.45 + 20\log_{10}(f_{GHz}) + 20\log_{10}(d_{km})$. 
The implementation maintains numerical precision within $10^{-4}$ dB across all operational frequencies (10–30 GHz) and slant ranges (35,000–45,000 km).

## 2. Rain Attenuation (ITU-R P.838 / P.839)
- **Coefficients:** Specific attenuation coefficients ($k, \alpha$) are verified via log-linear interpolation of ITU-R P.838-3 tables.
- **Rain Height:** Latitude-dependent model (P.839-4) tested for climate zone accuracy (e.g., Delhi at 4.58 km vs. Berlin at 3.32 km).

![Rain Attenuation Validation](../val_and_bench/val_rain_attenuation.png)
*Figure 1: Comparison of simulated rain attenuation against ITU-R P.838/P.618 analytical references across various rain rates.*

## 3. Geometry & SGP4
- **Slant Range:** Analytical checks for Zenith ($90^\circ$) and Horizon ($0^\circ$) elevations.
- **SGP4 vs. Analytical:** Cross-validation of SGP4-propagated slant range against GEO analytical models ensures that orbital drift is correctly captured without violating geostationary constraints.

![Geometry Validation](../val_and_bench/val_geometry.png)
*Figure 2: Validation of SGP4-derived elevation and slant range against static geometric benchmarks.*

## 4. Stochastic Rain Process (ITU-R P.1853)
- **Autocorrelation:** Verified decay constant $\rho = e^{-dt/\tau_c}$ matches the 5-minute ($\sim 300\text{s}$) correlation time characteristic of convective rain cells.
- **Stationary Distribution:** Long-run convergence to ITU-R P.837 lognormal mean ensures that simulated link availability matches climatological averages.

![Autocorrelation Validation](../val_and_bench/val_autocorr.png)
*Figure 3: Empirical autocorrelation of the Maseng-Bakken process showing the expected exponential decay over a 3600-step simulation.*
