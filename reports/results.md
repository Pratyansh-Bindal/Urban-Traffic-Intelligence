# Evaluated results

Models were selected on validation dates, then scored on final held-out dates. The main conditional estimator and the separate rolling advance-forecast experiments use different weather protocols, described below.

## Main road/date/time estimator

Chronological conditional prediction: recorded weather is supplied as known context on held-out rows. This measures estimation given weather, not the accuracy of an archived weather forecast.

| Target | Selected ML model | Test MAE | Test RMSE | Test R² | Seasonal baseline MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| daily_volume (vehicles/day) | Boosting: full | 8578.4194 | 10650.7009 | 0.3145 | 8546.4342 |
| travel_pace (minutes/km) | Boosting: full | 0.2660 | 0.4135 | 0.8210 | 0.2570 |

| Added context | Matched-model test MAE reduction |
| --- | ---: |
| Weather category | -0.434% |
| Time and corridor features | +56.912% |

No hourly count labels are created or used to train a model. Daily counts are allocated over 24 hours in proportion to predicted travel pace. This is an explicit modelling assumption; congestion pace is not a physical measurement of flow.

Hourly vehicle-count MAE is unavailable. Model selection is among the two ML candidates; the historical baseline is shown even where it performs better.

The sections below are the separate advance-forecast research experiments, which use only earlier observed weather.

## Volume

Selected model: Road / route median

Test period: 2024-03-20 to 2024-08-09. Unit: vehicles.

| Model | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: |
| Last observation | 11674.1106 | 14882.1519 | -0.3385 |
| Road / route median | 8398.1392 | 10573.7350 | 0.3243 |
| Weekday-hour median | 8546.4342 | 10716.9524 | 0.3059 |
| Random Forest | 8590.8157 | 10641.4679 | 0.3156 |
| Boosting: full | 8626.2687 | 10691.9931 | 0.3091 |
| Boosting: no weather | 8631.9894 | 10721.9383 | 0.3053 |
| Boosting: no festivals | 8637.0072 | 10724.5478 | 0.3049 |

Measured 90% band coverage: 92.02%

| Added feature group | MAE reduction on identical test rows |
| --- | ---: |
| Weather history | +0.066% |
| Festival calendar | +0.124% |

Positive reduction means improvement. Negative reduction means the feature group worsened this controlled experiment. This compares the full boosting model with matched ablations; it does not necessarily compare the selected model.

## Journey

Selected model: Random Forest

Test period: 2026-09-01 to 2026-10-31. Unit: minutes.

| Model | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: |
| Last observation | 3.0327 | 4.4086 | 0.8583 |
| Road / route median | 7.8633 | 9.7915 | 0.3012 |
| Weekday-hour median | 2.6986 | 4.3500 | 0.8621 |
| Random Forest | 1.7668 | 2.8464 | 0.9409 |
| Boosting: full | 1.7268 | 2.7141 | 0.9463 |
| Boosting: no weather | 1.7333 | 2.7252 | 0.9459 |
| Boosting: no festivals | 1.7629 | 2.7812 | 0.9436 |

Measured 90% band coverage: 91.34%

| Added feature group | MAE reduction on identical test rows |
| --- | ---: |
| Weather history | +0.376% |
| Festival calendar | +2.048% |

Positive reduction means improvement. Negative reduction means the feature group worsened this controlled experiment. This compares the full boosting model with matched ablations; it does not necessarily compare the selected model.

## Limits

Daily counts have unverified collection provenance. Google Maps journey durations are estimates. Weather covers only about 22% of the supplied route archive. Associations and input contrasts do not establish causation. Synthetic anomaly scores do not measure real incident accuracy.

Exact source revisions, hashes, training configuration and package versions are recorded in artifacts/*/manifest.json.
