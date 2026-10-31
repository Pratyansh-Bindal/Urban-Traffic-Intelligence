"""Post-observation detectors with causal rolling references and synthetic stress tests."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

from .settings import SEED


def rolling_reference(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = frame.copy()
    out['weekend_group'] = (out.timestamp.dt.day_of_week >= 5).astype(int)
    out['hour_group'] = out.timestamp.dt.hour if dataset == 'journey' else 0
    group_columns = ['entity', 'hour_group', 'weekend_group'] if dataset == 'journey' else ['entity']
    parts = []
    for _, group in out.groupby(group_columns, sort=False):
        g = group.sort_values('timestamp').copy()
        values = pd.Series(g.y.to_numpy(dtype=float), index=pd.DatetimeIndex(g.timestamp))
        window = values.rolling('28D', closed='left', min_periods=6 if dataset == 'journey' else 8)
        g['reference_median'] = window.median().to_numpy()
        g['reference_mad'] = window.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True).to_numpy()
        g['reference_iqr'] = (window.quantile(.75) - window.quantile(.25)).to_numpy()
        g['reference_count'] = values.rolling('28D', closed='left').count().to_numpy()
        scale = np.where(g.reference_mad > 1e-9, g.reference_mad * 1.4826,
                         np.where(g.reference_iqr > 1e-9, g.reference_iqr / 1.349,
                                  np.maximum(g.reference_median.abs() * .01, 1e-6)))
        g['reference_scale'] = scale
        g['mad_score'] = (g.y - g.reference_median) / g.reference_scale
        g['mad_alert'] = g.mad_score.abs().gt(3) & g.reference_median.notna()
        parts.append(g)
    return pd.concat(parts).sort_values(['timestamp', 'entity']).reset_index(drop=True)


def detector_matrix(frame, observed=None):
    y = frame.y.to_numpy(float) if observed is None else np.asarray(observed, float)
    med = frame.reference_median.to_numpy(float)
    scale = frame.reference_scale.to_numpy(float)
    z = np.divide(y - med, scale, out=np.zeros(len(y)), where=np.isfinite(scale) & (scale > 0))
    relative = np.divide(y - med, med, out=np.zeros(len(y)), where=np.isfinite(med) & (med > 0))
    matrix = np.column_stack([z, relative, frame.hour_sin, frame.hour_cos, frame.is_weekend])
    return np.nan_to_num(matrix, nan=0.0, posinf=1000.0, neginf=-1000.0)


def fit_detectors(training, calibration, calibration_predictions):
    eligible = training.reference_median.notna()
    if eligible.sum() < 30:
        raise ValueError('Insufficient history to fit the anomaly reference model.')
    forest = IsolationForest(n_estimators=100, contamination='auto', random_state=SEED, n_jobs=2)
    forest.fit(detector_matrix(training.loc[eligible]))
    cal_valid = calibration.reference_median.notna()
    scores = -forest.score_samples(detector_matrix(calibration.loc[cal_valid]))
    threshold = float(np.quantile(scores, .98))
    residuals = np.abs(calibration.y.to_numpy(float) - calibration_predictions)
    residual_threshold = float(np.quantile(residuals, .98))
    return {'isolation_forest': forest, 'isolation_threshold': threshold, 'residual_threshold': residual_threshold}


def apply_detectors(frame, predictions, detectors, observed=None):
    y = frame.y.to_numpy(float) if observed is None else np.asarray(observed, float)
    scores = (y - frame.reference_median.to_numpy(float)) / frame.reference_scale.to_numpy(float)
    valid = frame.reference_median.notna().to_numpy()
    return {
        'MAD': (np.abs(scores) > 3) & valid,
        'Isolation Forest': (-detectors['isolation_forest'].score_samples(detector_matrix(frame, y)) > detectors['isolation_threshold']) & valid,
        'Forecast residual': np.abs(y - np.asarray(predictions)) > detectors['residual_threshold'],
    }


def synthetic_evaluation(test, predictions, detectors):
    eligible = np.flatnonzero(test.reference_median.notna().to_numpy())
    rng = np.random.default_rng(SEED)
    number = max(1, int(.05 * len(eligible)))
    chosen = rng.choice(eligible, size=min(2 * number, len(eligible)), replace=False)
    observed = test.y.to_numpy(float).copy()
    labels = np.zeros(len(test), dtype=bool)
    labels[chosen] = True
    halfway = len(chosen) // 2
    observed[chosen[:halfway]] *= 1.5
    observed[chosen[halfway:]] *= .6
    alerts = apply_detectors(test, predictions, detectors, observed=observed)
    result = []
    for name, flags in alerts.items():
        result.append({'method': name, 'precision_on_injections': float(precision_score(labels, flags, zero_division=0)),
                       'recall_on_injections': float(recall_score(labels, flags, zero_division=0)),
                       'f1_on_injections': float(f1_score(labels, flags, zero_division=0)),
                       'unchanged_reference_alert_rate': float(flags[~labels].mean()),
                       'injected_rows': int(labels.sum()), 'reference_rows': int((~labels).sum())})
    return {'experiment': 'Isolated +50% spikes and -40% drops in held-out observations; original references and forecast features are held fixed.',
            'limitation': 'Unchanged reference rows can contain natural anomalies. These scores are synthetic stress tests, not real incident accuracy.',
            'results': result}
