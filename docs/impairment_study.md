# Isolated Impairment Impact Experiment Study

This document logs the incremental impact of adding physical receiver and link impairments on rainfall narrowcasting.

| Experiment | F1-Score | Regressor RMSE (mm/h) | Regressor MAE (mm/h) | Regressor $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| 1. Only Scintillation | 0.9989 | 1.5534 | 0.5461 | 0.9588 |
| 2. + Tracking | 0.9519 | 4.8974 | 2.5569 | 0.5909 |
| 3. + Calibration | 0.8933 | 4.7679 | 2.7113 | 0.6123 |
| 4. + AGC & ADC | 0.9162 | 5.0085 | 2.7528 | 0.5722 |
| 5. + Multipath | 0.8252 | 5.2038 | 2.9469 | 0.5381 |
| 6. + Wet Antenna | 0.8487 | 5.3262 | 3.0163 | 0.5162 |
