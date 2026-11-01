# Verification report

Verified on **1 November 2026 (Asia/Kolkata)** using macOS arm64, Python 3.14.5 and Streamlit 1.66.1. The working source, including the prediction-summary feature and fixes below, passed the checks listed here. These checks establish software behaviour and reproducibility; they do not establish hourly vehicle-count accuracy.

## Completed checks

| Check | Result |
| --- | --- |
| Python syntax | All 27 application and test modules parsed successfully |
| Automated tests, prepared local environment | **45 passed**, no skips |
| Automated tests, newly installed environment after full training | **45 passed**, no skips |
| Dependency consistency | `pip check` passed in both environments; installed pins also satisfy `requirements.txt` |
| Known dependency vulnerabilities | PyPA `pip-audit` checked all **44 pinned packages**; no known vulnerabilities and no skipped packages were reported |
| Source integrity | SHA-256 checks passed for the volume CSV, TML traffic CSV, route CSV and schema |
| Independent source download | Re-downloaded all three TML files at the pinned upstream commit; every recorded checksum matched |
| Data audit | 8,936 usable daily-count rows across 16 roads; 113,782 completed-hour journey rows across 13 routes; no invalid source rows |
| Stored histories | Both saved histories equal the outputs of the raw-data loaders |
| Prediction scenarios | **64 passed** across all 16 roads: recorded dates, upcoming midnight, 23:45 and dates beyond the weather forecast horizon |
| Scenario invariants | 24 unique hourly bins; finite, nonnegative values; hourly totals equal daily volume; selected-hour values match the graph; completed observations only; valid JSON |
| Temporal checks | Separate chronological partitions, current/future observations excluded, expired weather excluded and event announcement cutoffs enforced |
| Model reports | Independently recomputed selected-model test errors and interval coverage; validation scores determine the reported winners |
| Fresh full training | Reproduced all three branches' reported metrics and model choices; generated `reports/results.md` matched byte for byte |
| Prediction summary | Weather, calendar, time period and corridor context checked; missing weather and insufficient anomaly history disclosed; historical daily counts remain distinguished from hourly estimates |
| Saved results | Summary and full payload survive SQLite round trips and JSON export; older records still open without a stored summary |
| Visitor privacy | Separate sessions cannot list, export or open another session's saved results in visitor mode; SQL filtering occurs before the result limit |
| Git hygiene | Diff whitespace check passed; runtime models, raw CSVs and prediction databases remain excluded from tracked files; basic credential-pattern scan found no matches |

## Fresh-install and retraining procedure

A temporary source copy was created without an existing environment, trained models or prediction database. All packages were independently installed from `requirements-lock.txt`. The documented fetch command copied the supplied count CSV and downloaded the immutable TML snapshot.

After the independent download checks passed, the original source manifest's acquisition cutoff, **2026-10-31T13:10:16.745475+00:00**, was retained for the strict retraining comparison. This ensures both runs use the same set of completed hours rather than allowing a later download time to include an additional hour. Full training ran without `--quick`, followed by all 45 tests.

The verified main-estimator artifacts were installed locally with a separate manifest recording package versions. The user's original source histories and prediction database were preserved.

## Live-service and browser checks

| Request | Result |
| --- | --- |
| Weather for 2026-11-01 | Available forecast, 24 hourly rows |
| Weather for 2026-10-02 | Available historical reanalysis, 24 hourly rows |
| Weather for 2026-11-17 | Correctly returned the outside-horizon fallback without a weather request |
| Temporary dashboard health endpoint | `ok` |
| Ballari Road, 2026-11-01 21:30 IST | Forecast rendered with weather, Rajyotsava, summary and graphs; approximately 1,264 vehicles for the selected hour |
| Immediately following search: Sarjapur Road, 2026-11-02 18:30 IST | Updated road, date and time correctly; forecast rain and summary rendered; approximately 2,553 vehicles for the selected hour |
| JSON download | Actual browser download parsed successfully and exactly matched the saved result payload |
| Reopen saved result | Original weather, summary, graph and requested time were restored |
| Separate visitor tab | Displayed an empty saved history while the first session retained its two requests |
| Browser console | No error-level messages recorded during the checked workflow |

Live forecast values above describe the verification run, not measured traffic or accuracy results. Browser tests used the temporary project's database, rather than the user's database.

The subsequent weather-display update was also checked with all **45 tests passing** and a separate temporary browser preview. The result now has a prominent weather-condition card, available temperature/humidity/precipitation values and a visible explanation of weather inputs. The live preview showed **Rainy** for the selected hour and **Cloudy / overcast** as the day's most common category, with their separate roles labelled. Missing weather is shown as **Unavailable**. This display update did not require retraining.

## Fixes made during verification

1. **Consecutive searches:** request widgets now use stable keys and a fixed initial default, so the previous result cannot replace a new road/date/time submission.
2. **Custom-event availability:** the forecast issue time is passed through feature generation, so an event announced later cannot enter an earlier prediction or its explanation. Tests include the UTC-to-IST announcement boundary.
3. **Streamlit requirement:** raised the minimum to the tested 1.66.1, matching the layout APIs used by the application.
4. **Sharing preparation:** `TRAFFIC_PUBLIC_MODE=1` restricts saved-result access to the visitor's current session and hides Research explorer. The normal local mode retains the owner's full history.
5. **Route-label fallback:** removed a pandas dtype warning when route metadata is absent.
6. **Estimator provenance:** full training now saves a standalone estimator manifest with the package versions, source metadata and events checksum.

The two reproduced functional failures were observed failing before the fixes and passing afterwards. Regression tests remain in the repository.

## Measured results and remaining limits

| Target | Validation-selected model | Recomputed held-out MAE |
| --- | --- | ---: |
| Advance daily volume | Road / route median | 8,398.1392 vehicles/day |
| Advance journey duration | Random Forest | 1.7668 minutes |
| Conditional daily volume | Boosting: full | 8,578.4194 vehicles/day |
| Conditional travel pace | Boosting: full | 0.2660 minutes/km |

- Hourly vehicle counts remain an estimated allocation of daily volume using travel patterns; independent hourly counts are still needed to measure their accuracy.
- The main estimator still selects between the two ML candidates. Its reported seasonal baselines have slightly lower test MAE; making baselines eligible would require validation-based model selection, rather than switching based on the final test set.
- Weather and festival connections are working; their presence does not establish a causal effect or a consistent improvement in accuracy. Recorded-weather conditional evaluation is distinct from an archived weather-forecast backtest.
- Anomaly tests verify the detector logic and reference handling. They do not establish real incident-detection accuracy without labelled incidents.
- The laptop-to-internet sharing launcher and public URL remain pending. The temporary browser server was loopback-only; visitor mode was verified locally.
- These runs verify this Mac and Python version. Cross-platform installation and GitHub Actions are not covered by this report.
- The repeated **Moderate** traffic-condition label is under investigation. This update does not change the existing travel-pace classification thresholds.

Temporary environments, downloaded test copies, browser-test prediction records and the JSON test download were removed after verification. The preview server was stopped. Useful source changes, regression tests, local model artifacts and this report were retained. This report records the verification run; publication history is available in Git.
