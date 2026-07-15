# Tracking Noise Sweep Experiment Study

This document logs the impact of increasing nominal antenna tracking loop error standard deviation (σ_tracking) on rainfall narrowcasting classification F1 and regression $R^2$ scores.

| Tracking σ | F1-Score | Regressor $R^2$ Score |
| :--- | :---: | :---: |
| 0.00° | 0.9989 | 0.9588 |
| 0.02° | 0.9564 | 0.9054 |
| 0.05° | 0.9054 | 0.3677 |
| 0.10° | 0.8787 | 0.2823 |
| 0.20° | 0.7393 | 0.2465 |
| 0.50° | 0.6244 | 0.0943 |
