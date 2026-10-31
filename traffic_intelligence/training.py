from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

from .anomalies import apply_detectors, fit_detectors, rolling_reference, synthetic_evaluation
from .data import audit_all, load_dataset
from .features import assert_no_leakage, make_features
from .models import build_model, chronological_split, conformal_radius, regression_metrics
from .settings import ARTIFACTS, DATASETS, EVENTS, RAW, ROOT, SEED


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False))


def historical_effects(training):
    data = training.copy()
    medians = data.groupby(['entity', 'weekday', 'hour']).y.transform('median')
    data['relative_to_context'] = data.y / medians
    results = {}
    for column, name in [('weather', 'weather'), ('is_festival', 'festival')]:
        summary = data.groupby(column, dropna=False).agg(rows=('y', 'size'), dates=('timestamp', lambda x: x.dt.normalize().nunique()),
                                                        median_relative=('relative_to_context', 'median'), median_value=('y', 'median')).reset_index()
        results[name] = summary.to_dict('records')
    results['note'] = 'Training-period descriptive association after normalising by road/route, weekday and hour medians. Confounding remains; this is not a causal weather or festival effect.'
    return results


def train_dataset(dataset: str, quick=False):
    folder = ARTIFACTS / dataset
    folder.mkdir(parents=True, exist_ok=True)
    print(f'[{dataset}] Loading and auditing data', flush=True)
    history, audit = load_dataset(dataset)
    print(f'[{dataset}] Building historical features and rolling anomaly references ({len(history):,} periods)', flush=True)
    features = make_features(history, dataset)
    assert_no_leakage(features)
    features = rolling_reference(features, dataset)
    features, splits = chronological_split(features)
    train = features.loc[features.partition.eq('train')].copy()
    selection = features.loc[features.partition.eq('selection')].copy()
    calibration = features.loc[features.partition.eq('calibration')].copy()
    test = features.loc[features.partition.eq('test')].copy()
    specs = [('Last observation', True, True), ('Road / route median', True, True), ('Weekday-hour median', True, True),
             ('Random Forest', True, True), ('Boosting: full', True, True),
             ('Boosting: no weather', False, True), ('Boosting: no festivals', True, False)]
    models, metrics, predictions = {}, [], {}
    for name, weather, festivals in specs:
        print(f'[{dataset}] Training {name}', flush=True)
        started = time.perf_counter()
        model = build_model(name, weather=weather, festivals=festivals, quick=quick).fit(train)
        elapsed = time.perf_counter() - started
        models[name] = model
        validation_pred = model.predict(selection)
        test_pred = model.predict(test)
        predictions[name] = test_pred
        record = {'model': name, 'training_seconds': elapsed,
                  'selection': regression_metrics(selection.y, validation_pred), 'test': regression_metrics(test.y, test_pred)}
        metrics.append(record)
    # Model choice uses the first half of validation only. Test scores do not choose the winner.
    winner = min(metrics, key=lambda item: item['selection']['mae'])['model']
    best = models[winner]
    cal_predictions = best.predict(calibration)
    cal_errors = np.abs(calibration.y.to_numpy(float) - cal_predictions)
    global_radius = conformal_radius(cal_errors)
    radii = {}
    for entity, group in calibration.assign(error=cal_errors).groupby('entity'):
        radii[str(entity)] = conformal_radius(group.error) if len(group) >= 30 else global_radius
    prediction_frame = test[['timestamp', 'entity', 'label', 'y', 'weather', 'is_festival', 'holiday_name',
                             'reference_median', 'reference_scale', 'mad_score']].copy()
    prediction_frame['prediction'] = predictions[winner]
    prediction_frame['radius'] = prediction_frame.entity.map(radii).fillna(global_radius)
    prediction_frame['lower'] = np.maximum(0, prediction_frame.prediction - prediction_frame.radius)
    prediction_frame['upper'] = prediction_frame.prediction + prediction_frame.radius
    prediction_frame['residual'] = prediction_frame.y - prediction_frame.prediction
    coverage = float(prediction_frame.y.between(prediction_frame.lower, prediction_frame.upper).mean())
    detectors = fit_detectors(train, calibration, cal_predictions)
    for name, flags in apply_detectors(test, predictions[winner], detectors).items():
        prediction_frame['alert_' + name.lower().replace(' ', '_')] = flags
    synthetic = synthetic_evaluation(test, predictions[winner], detectors)
    effects = historical_effects(train)
    full = next(item for item in metrics if item['model'] == 'Boosting: full')
    ablations = []
    for name, label in [('Boosting: no weather', 'Weather history'), ('Boosting: no festivals', 'Festival calendar')]:
        omitted = next(item for item in metrics if item['model'] == name)
        ablations.append({'feature_group': label, 'without_mae': omitted['test']['mae'], 'with_mae': full['test']['mae'],
                          'mae_improvement_pct': 100 * (omitted['test']['mae'] - full['test']['mae']) / omitted['test']['mae']})
    print(f'[{dataset}] Explaining the full model on validation data', flush=True)
    sample = selection.sample(min(1000, len(selection)), random_state=SEED)
    # Importance is evaluated on selection data, so it cannot guide edits based on the final test set.
    importance = permutation_importance(models['Boosting: full'].estimator, sample[models['Boosting: full'].columns],
                                       sample.y, scoring='neg_mean_absolute_error', n_repeats=2, random_state=SEED, n_jobs=1)
    ranked = sorted([{'feature': c, 'mae_increase': float(v)} for c,v in zip(models['Boosting: full'].columns, importance.importances_mean)],
                    key=lambda r: r['mae_increase'], reverse=True)
    test_baseline = next(item for item in metrics if item['model'] == 'Weekday-hour median')['test']['mae']
    chosen_test = next(item for item in metrics if item['model'] == winner)['test']
    report = {'dataset': dataset, 'unit': DATASETS[dataset]['unit'], 'winner': winner, 'splits': splits,
              'models': metrics, 'ablations': ablations, 'test_interval_coverage': coverage, 'nominal_interval_coverage': .90,
              'mae_improvement_over_seasonal_baseline_pct': 100 * (test_baseline - chosen_test['mae']) / test_baseline,
              'evaluation': 'Rolling one-period-ahead evaluation with observed history updated after each completed period; model weights stay frozen.',
              'weather_protocol': 'Only prior completed-period weather is used. Future realised weather is never supplied to historical forecasts.',
              'interval_note': 'Split-conformal-style absolute-error bands from separate calibration dates. Temporal dependence and drift mean nominal coverage is not guaranteed; measured test coverage is reported.',
              'festival_note': 'Calendar features are learned from data. No positive uplift is hardcoded. Manual uplift is a separate scenario.',
              'seed': SEED, 'quick_mode': quick}
    source_manifest = json.loads((RAW / 'sources.json').read_text())
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'dataset': dataset,
                'versions': {p: importlib.metadata.version(p) for p in ['pandas', 'numpy', 'scikit-learn', 'holidays', 'streamlit']},
                'sources': source_manifest,
                'custom_events_sha256': hashlib.sha256(EVENTS.read_bytes()).hexdigest() if EVENTS.exists() else None}
    bundle = {'dataset': dataset, 'models': models, 'winner': winner, 'radius_by_entity': radii, 'global_radius': global_radius,
              'detectors': detectors, 'report': report, 'manifest': manifest,
              'entities': history[['entity', 'label']].drop_duplicates().to_dict('records')}
    history.to_parquet(folder / 'history.parquet', index=False)
    prediction_frame.to_parquet(folder / 'predictions.parquet', index=False)
    prediction_frame.to_csv(folder / 'predictions.csv', index=False)
    joblib.dump(bundle, folder / 'model.joblib', compress=3)
    for filename, payload in [('report.json', report), ('audit.json', audit), ('manifest.json', manifest),
                               ('feature_importance.json', ranked), ('effects.json', effects), ('synthetic_anomalies.json', synthetic)]:
        save_json(folder / filename, payload)
    print(f'[{dataset}] Selected {winner}; test MAE {chosen_test["mae"]:.3f} {DATASETS[dataset]["unit"]}', flush=True)
    return {'winner': winner, 'test': chosen_test, 'interval_coverage': coverage, 'periods': len(history)}


def train_all(dataset='all', quick=False):
    audit_all()
    names = ['volume', 'journey'] if dataset == 'all' else [] if dataset == 'estimator' else [dataset]
    results = {}
    with threadpool_limits(limits=2):
        for name in names:
            results[name] = train_dataset(name, quick=quick)
        if dataset in ('all', 'estimator'):
            from .estimation_training import train_estimator
            results['estimator'] = train_estimator(quick=quick)
    write_result_summary()
    return results


def write_result_summary():
    """Keep the human-readable results in sync after every completed training run."""
    output = ROOT / 'reports'
    output.mkdir(exist_ok=True)
    lines = ['# Evaluated results', '',
             'Models were selected on validation dates, then scored on final held-out dates. The main conditional estimator and the separate rolling advance-forecast experiments use different weather protocols, described below.', '']
    estimator_path = ARTIFACTS / 'estimator' / 'report.json'
    if estimator_path.exists():
        estimator = json.loads(estimator_path.read_text())
        lines += ['## Main road/date/time estimator', '', estimator['evaluation_protocol'], '',
                  '| Target | Selected ML model | Test MAE | Test RMSE | Test R² | Seasonal baseline MAE |',
                  '| --- | --- | ---: | ---: | ---: | ---: |']
        for target, result in estimator['models'].items():
            metric = result['selected_test']
            lines.append(f'| {target} ({result["unit"]}) | {result["model"]} | {metric["mae"]:.4f} | {metric["rmse"]:.4f} | {metric["r2"]:.4f} | {result["baseline_test"]["mae"]:.4f} |')
        lines += ['', '| Added context | Matched-model test MAE reduction |', '| --- | ---: |']
        for target, label in [('daily_volume', 'Weather category'), ('travel_pace', 'Time and corridor features')]:
            lines.append(f'| {label} | {estimator["models"][target]["ablation"]["mae_improvement_pct"]:+.3f}% |')
        lines += ['', estimator['hourly_volume_protocol'], '',
                  'Hourly vehicle-count MAE is unavailable. Model selection is among the two ML candidates; the historical baseline is shown even where it performs better.', '',
                  'The sections below are the separate advance-forecast research experiments, which use only earlier observed weather.', '']
    for name in DATASETS:
        path = ARTIFACTS / name / 'report.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        dates = report['splits']['test']
        lines += [f'## {name.title()}', '', f'Selected model: {report["winner"]}', '',
                  f'Test period: {dates["first_date"]} to {dates["last_date"]}. Unit: {report["unit"]}.', '',
                  '| Model | Test MAE | Test RMSE | Test R² |', '| --- | ---: | ---: | ---: |']
        for model in report['models']:
            m = model['test']
            lines.append(f'| {model["model"]} | {m["mae"]:.4f} | {m["rmse"]:.4f} | {m["r2"]:.4f} |')
        lines += ['', f'Measured 90% band coverage: {report["test_interval_coverage"]:.2%}', '',
                  '| Added feature group | MAE reduction on identical test rows |', '| --- | ---: |']
        for effect in report['ablations']:
            lines.append(f'| {effect["feature_group"]} | {effect["mae_improvement_pct"]:+.3f}% |')
        lines += ['', 'Positive reduction means improvement. Negative reduction means the feature group worsened this controlled experiment. This compares the full boosting model with matched ablations; it does not necessarily compare the selected model.', '']
    lines += ['## Limits', '',
              'Daily counts have unverified collection provenance. Google Maps journey durations are estimates. Weather covers only about 22% of the supplied route archive. Associations and input contrasts do not establish causation. Synthetic anomaly scores do not measure real incident accuracy.', '',
              'Exact source revisions, hashes, training configuration and package versions are recorded in artifacts/*/manifest.json.', '']
    (output / 'results.md').write_text('\n'.join(lines))
