# Isolated Impairment Impact Experiment Study

This document logs the impact of physical receiver and link impairments on rainfall narrowcasting, evaluating both incremental addition and individual (solo) effects.

## 1. Incremental Degradation
| Experiment | F1-Score | Regressor RMSE (mm/h) | Regressor MAE (mm/h) | Regressor $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| 1. Only Scintillation | 0.9989 | 1.5534 | 0.5461 | 0.9588 |
| 2. + Tracking | 0.9519 | 4.8974 | 2.5569 | 0.5909 |
| 3. + Calibration | 0.8933 | 4.7679 | 2.7113 | 0.6123 |
| 4. + AGC & ADC | 0.9162 | 5.0085 | 2.7528 | 0.5722 |
| 5. + Multipath | 0.8252 | 5.2038 | 2.9469 | 0.5381 |
| 6. + Wet Antenna | 0.8487 | 5.3262 | 3.0163 | 0.5162 |

## 2. Solo Impairment Effects
| Experiment | F1-Score | Regressor RMSE (mm/h) | Regressor MAE (mm/h) | Regressor $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| 1. Scintillation Solo | 0.9989 | 1.5534 | 0.5461 | 0.9588 |
| 2. Tracking Solo | 0.9467 | 5.0040 | 2.5901 | 0.5729 |
| 3. Calibration Solo | 0.9814 | 2.1470 | 0.7964 | 0.9214 |
| 4. AGC & ADC Solo | 0.9878 | 2.0245 | 1.0156 | 0.9301 |
| 5. Multipath Solo | 0.9645 | 2.7274 | 1.5284 | 0.8731 |
| 6. Wet Antenna Solo | 0.9991 | 1.9094 | 0.6181 | 0.9378 |
