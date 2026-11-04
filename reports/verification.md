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

## Traffic classification update

New predictions now use three explicit **vehicle-volume** bands, replacing the previous travel-pace label. For each road, the one-third and two-thirds quantiles of earlier daily counts are divided by 24. The lower cutoff is rounded down and the upper cutoff up. The displayed whole-vehicle hourly estimate is **Low** below the lower cutoff, **High** above the upper cutoff and **Moderate** between them, including both boundaries. At least eight earlier valid daily records are required. The cutoffs are estimated hourly references derived from daily counts, not measured hourly thresholds or road capacities.

- **57 automated tests passed**, including exact boundaries, count rounding, constant histories, road-specific thresholds, insufficient data, invalid estimates, future-observation exclusion, saved-result compatibility and UI rendering.
- **64 complete requests across all 16 roads passed** at 00:00, 08:00, 12:00 and 18:00 on 2 November 2026 using unavailable-weather fallback: 8 Low, 23 Moderate and 33 High. All labels matched the displayed count and stored cutoffs, and each road's cutoffs stayed consistent across the selected day.
- For **100 Feet Road**, cutoffs were 1,058 and 1,554 vehicles/hour: 905 at 00:00 was Low, 1,403 at 12:00 Moderate and 1,702 at 18:00 High.
- The live browser showed **Sarjapur Road at 18:30** as **High**: approximately 2,553 vehicles/hour exceeded its upper cutoff of 1,878; its lower cutoff was 1,355. Both rules were visible below the metrics, both horizontal guides rendered on the graph and the summary included the volume level.
- Each new saved result includes its cutoffs, reference sample size, comparison value and boundary rule. Older saved results keep their original label and visibly identify the earlier travel-time classification.

The trained predictions and anomaly Z-scores are unchanged; this is a transparent classification policy applied to the vehicle estimate. Full model retraining was not required. The earlier label was not hardcoded: a separate pre-change 1,152-hour probe produced 749 Moderate labels, reflecting its broad travel-pace band and the forecasts falling within it. That probe and the new 64-request check use different scenario sets and are not an accuracy comparison.

## Public-link launcher · 4 November 2026

The laptop-sharing feature is implemented. Run `python -m traffic_intelligence share` with the project environment, or double-click `Share Traffic.command` on this Mac. The launcher owns a separate loopback Streamlit server and a Cloudflare Quick Tunnel, enables visitor mode, prints a verified HTTPS URL and exposes it in the sidebar. Ctrl+C, terminal termination and startup failures clean up the child processes. The normal owner dashboard remains available through the original local command.

- **74 automated tests passed**, with no skips. The new cases cover download integrity and reuse, corrupt-cache repair, safe archive handling, URL validation, the real Streamlit CLI's host/origin lists, visitor mode, startup failures, shutdown, browser-opening failure, occupied ports and DNS-publication ordering. Existing model, temporal-integrity and dashboard checks still pass.
- All **29 Python files** parsed successfully; the Mac shortcut passed `zsh -n` and retains executable permissions. `git diff --check` passed.
- The official macOS ARM64 cloudflared 2026.10.7 archive matched its GitHub release-asset digest (`3586ccf397e98e5f437e60e7c307ca6ac463e110580503af8bb8e6f9ebf831bd`). Only the executable and its integrity checksum are retained in ignored `.tools/`; the archive is not retained.
- The app was opened through its actual public HTTPS hostname, including an interactive WebSocket session. A **100 Feet Road, 6 November 2026, 18:30 IST** request returned approximately **1,740 vehicles/hour**, **High**, cloudy/overcast weather, festival context, the summary and graphs. This is a test prediction, not a measured count. The browser recorded no error-level console messages.
- A separate visitor session saw an empty saved history while the first visitor retained its request. Research explorer was absent. These checks used a temporary source copy and database while reading the existing trained artifacts, leaving personal prediction records untouched.
- Public HTTPS prediction checks passed again after restarting with a new hostname and automatically reusing the installed executable. The sidebar displayed the complete copyable URL with wrapped lines.
- Live shutdown checks confirmed that the local listener stopped and the old public health endpoint returned an unavailable response. A second launcher targeting an active shared-app port failed before creating another tunnel.

The restart checks exposed two startup issues: a recently closed TCP connection could make the port check report a false conflict, and a premature lookup could leave macOS with a stale negative DNS result for a new hostname. The port check now allows TCP address reuse on POSIX while rejecting active listeners; on Windows it requests exclusive address use. Startup waits for a positive public DNS record through Cloudflare's HTTPS DNS API before using the normal OS resolver. DNS settings and certificate verification remain unchanged. Failures report the last health-check error and stop sharing without claiming a usable link.

Verification covers this Mac's ARM64 runtime. Other listed download platforms have not been tested end to end. Quick Tunnels have changing URLs and no uptime guarantee; laptop availability, network restrictions and available CPU/memory still limit sharing. A fixed hostname or service while the laptop is off requires a hosted deployment. Test tunnels were stopped and temporary test files removed after verification.

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
- Laptop sharing is verified above. It uses a temporary Quick Tunnel; a stable public deployment and availability while the laptop is off remain future work.
- These runs verify this Mac and Python version. Cross-platform installation and GitHub Actions are not covered by this report.

Temporary environments, downloaded test copies, browser-test prediction records and the JSON test download were removed after verification. The preview server was stopped. Useful source changes, regression tests, local model artifacts and this report were retained. This report records the verification run; publication history is available in Git.
