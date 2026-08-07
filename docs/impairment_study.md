# Isolated Impairment Impact Experiment Study

This document logs the impact of physical receiver and link impairments on rainfall narrowcasting, evaluating both incremental addition and individual (solo) effects.

## 1. Incremental Degradation
| Experiment | F1-Score | Regressor RMSE (mm/h) | Regressor MAE (mm/h) | Regressor $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| 1. Only Scintillation | 0.9989 | 1.1321 | 0.4030 | 0.9781 |
| 2. + Tracking | 0.9307 | 3.8518 | 1.8311 | 0.7469 |
| 3. + Calibration | 0.8788 | 3.4515 | 1.8306 | 0.7968 |
| 4. + AGC & ADC | 0.8707 | 4.3229 | 2.2851 | 0.6813 |
| 5. + Multipath | 0.9048 | 4.4266 | 2.3146 | 0.6658 |
| 6. + Wet Antenna | 0.9045 | 4.4238 | 2.3929 | 0.6662 |

## 2. Solo Impairment Effects
| Experiment | F1-Score | Regressor RMSE (mm/h) | Regressor MAE (mm/h) | Regressor $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| 1. Scintillation Solo | 0.9989 | 1.1321 | 0.4030 | 0.9781 |
| 2. Tracking Solo | 0.9436 | 3.7713 | 1.8293 | 0.7574 |
| 3. Calibration Solo | 0.9922 | 1.2643 | 0.5656 | 0.9727 |
| 4. AGC & ADC Solo | 0.9836 | 2.1833 | 1.0047 | 0.9187 |
| 5. Multipath Solo | 0.9754 | 2.5520 | 1.4019 | 0.8889 |
| 6. Wet Antenna Solo | 0.9991 | 1.1099 | 0.3916 | 0.9790 |
