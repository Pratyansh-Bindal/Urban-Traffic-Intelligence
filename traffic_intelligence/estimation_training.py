"""Train the product's conditional volume and travel-pattern models on real labels."""
from datetime import datetime, timezone
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .calendar import FESTIVAL_FEATURES
from .context import CONDITION_FEATURES, TIME_FEATURES, condition_frame
from .data import load_journey, load_volume
from .features import BASIC_FEATURES, HISTORY_FEATURES
from .models import build_model, chronological_split, conformal_radius, regression_metrics
from .settings import ARTIFACTS, EVENTS, RAW, SEED


def train_estimator(quick=False):
    folder = ARTIFACTS / 'estimator'
    folder.mkdir(parents=True, exist_ok=True)
    volume, _ = load_volume()
    journey, _ = load_journey()
    daily = condition_frame(volume, use_history=True)
    pace = journey.copy()
    pace['y'] = pace.y / pace.distance
    pace = condition_frame(pace)
    daily_columns = BASIC_FEATURES + HISTORY_FEATURES + FESTIVAL_FEATURES + TIME_FEATURES + ['weather_context']
    pace_columns = BASIC_FEATURES + FESTIVAL_FEATURES + TIME_FEATURES + CONDITION_FEATURES
    fitted, reports = {}, {}
    with threadpool_limits(limits=2):
        for target, frame, columns, unit in [('daily_volume', daily, daily_columns, 'vehicles/day'),
                                              ('travel_pace', pace, pace_columns, 'minutes/km')]:
            frame, splits = chronological_split(frame)
            partitions = {name: frame.loc[frame.partition.eq(name)].copy()
                          for name in ('train', 'selection', 'calibration', 'test')}
            models, scores = {}, []
            for name in ('Random Forest', 'Boosting: full'):
                print(f'[estimator] Training {target}: {name}', flush=True)
                model = build_model(name, columns=columns, quick=quick).fit(partitions['train'])
                models[name] = model
                scores.append({'model': name,
                               'selection': regression_metrics(partitions['selection'].y, model.predict(partitions['selection'])),
                               'test': regression_metrics(partitions['test'].y, model.predict(partitions['test']))})
            winner = min(scores, key=lambda item: item['selection']['mae'])['model']
            baseline = build_model('Weekday-hour median', columns=['entity', 'weekday', 'hour']).fit(partitions['train'])
            baseline_scores = regression_metrics(partitions['test'].y, baseline.predict(partitions['test']))
            model = models[winner]
            calibration_errors = abs(partitions['calibration'].y.to_numpy() - model.predict(partitions['calibration']))
            radius = conformal_radius(calibration_errors)
            # Matched ablation: same model family and chronological rows.
            removed = ['weather_context'] if target == 'daily_volume' else ['hour', 'hour_sin', 'hour_cos', *TIME_FEATURES]
            ablated_columns = [c for c in columns if c not in removed]
            ablated = build_model(winner, columns=ablated_columns, quick=quick).fit(partitions['train'])
            ablated_score = regression_metrics(partitions['test'].y, ablated.predict(partitions['test']))
            selected_score = next(s for s in scores if s['model'] == winner)['test']
            reports[target] = {'model': winner, 'unit': unit, 'splits': splits, 'models': scores,
                               'baseline_test': baseline_scores, 'selected_test': selected_score,
                               'calibration_radius': radius,
                               'ablation': {'omitted': removed, 'without_test': ablated_score,
                                            'with_test': selected_score,
                                            'mae_improvement_pct': 100 * (ablated_score['mae'] - selected_score['mae']) / ablated_score['mae']}}
            fitted[target] = model
            fitted[target + '_radius'] = radius
            print(f'[estimator] {target}: {winner}, test MAE {selected_score["mae"]:.3f} {unit}', flush=True)
    created = datetime.now(timezone.utc).isoformat()
    priors = {int(month): group.weather.value_counts(normalize=True).to_dict()
              for month, group in volume.groupby(volume.timestamp.dt.month)}
    report = {'created_at_utc': created, 'seed': SEED, 'models': reports,
              'evaluation_protocol': 'Chronological conditional prediction: recorded weather is supplied as known context on held-out rows. This measures estimation given weather, not the accuracy of an archived weather forecast.',
              'hourly_volume_protocol': 'No hourly count labels are created or used to train a model. Daily counts are allocated over 24 hours in proportion to predicted travel pace. This is an explicit modelling assumption; congestion pace is not a physical measurement of flow.',
              'hourly_volume_accuracy': None,
              'time_features': TIME_FEATURES,
              'airport_note': 'Airport-corridor identity is a feature. No always-busy rule or fixed lunch/morning multiplier is imposed.',
              'quick_mode': quick}
    manifest = {'created_at_utc': created, 'sources': json.loads((RAW / 'sources.json').read_text()),
                'custom_events_sha256': hashlib.sha256(EVENTS.read_bytes()).hexdigest() if EVENTS.exists() else None}
    payload = {'models': fitted, 'report': report, 'manifest': manifest, 'weather_priors': priors}
    temporary = folder / 'model.joblib.part'
    joblib.dump(payload, temporary, compress=3)
    temporary.replace(folder / 'model.joblib')
    (folder / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
