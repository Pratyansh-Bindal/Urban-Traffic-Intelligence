import hashlib

import numpy as np
import pandas as pd

from .features import make_features
from .settings import DATASETS, EVENTS


def predict_one(bundle, history, entity, timestamp, *, weather=None, uplift_pct=0.0):
    """One completed-period forecast, plus explicitly conditional model responses."""
    dataset = bundle['dataset']
    period = pd.Timedelta(DATASETS[dataset]['frequency'])
    target = pd.Timestamp(timestamp)
    if target.tzinfo is not None:
        target = target.tz_convert('Asia/Kolkata').tz_localize(None)
    if target != target.floor(DATASETS[dataset]['frequency']):
        raise ValueError('Select midnight for a daily forecast or an exact hour for a journey forecast.')
    if entity not in history.entity.unique():
        raise ValueError('This road or route is not represented in the trained dataset.')
    if target.normalize() <= pd.Timestamp(bundle['report']['splits']['train']['last_date']):
        raise ValueError('Choose a date after the training period for a forecast or historical replay.')
    if not 0 <= uplift_pct <= 200:
        raise ValueError('The additional event uplift must be between 0 and 200 percent.')
    prior = history.loc[history.entity.eq(entity) & (history.timestamp + period <= target)].copy()
    if prior.empty:
        raise ValueError('No completed observations are available before this forecast origin.')
    prior = prior.sort_values('timestamp')
    available_at = prior.timestamp.max() + period
    age_hours = float((target - available_at).total_seconds() / 3600)
    query = prior.iloc[[-1]].copy()
    query['timestamp'] = target
    query['y'] = np.nan
    query['weather'] = 'Unknown'
    for c in ('temperature', 'humidity', 'feels_like', 'aqi', 'speed', 'distance'):
        if c in query:
            query[c] = np.nan
    engineered = make_features(pd.concat([prior, query], ignore_index=True), dataset)
    row = engineered.loc[engineered.timestamp.eq(target)].copy()
    primary = float(bundle['models'][bundle['winner']].predict(row)[0])
    full_model = bundle['models']['Boosting: full']
    context_prediction = float(full_model.predict(row)[0])
    scenario = row.copy()
    weather_assumed = weather is not None
    if weather:
        category = weather.get('condition', 'Unknown')
        if dataset == 'journey' and category in ('Clear', 'Overcast'):
            category = 'Dry'
        scenario['weather_last'] = category
        scenario['weather_age_hours'] = 0.0
        scenario['weather_available'] = int(category != 'Unknown')
        for name in ('temperature', 'humidity', 'feels_like', 'aqi'):
            if name in weather and weather[name] is not None:
                scenario[name + '_last'] = float(weather[name])
    scenario_model_value = float(full_model.predict(scenario)[0])
    no_festival = scenario.copy()
    for c in ('is_holiday', 'is_festival', 'festival_window', 'is_custom_event'):
        no_festival[c] = 0
    no_festival['days_to_festival'] = 60
    no_festival['days_since_festival'] = 60
    no_festival['festival_name'] = 'None'
    no_festival_value = float(full_model.predict(no_festival)[0])
    radius = bundle['radius_by_entity'].get(entity, bundle['global_radius'])
    events_hash = hashlib.sha256(EVENTS.read_bytes()).hexdigest() if EVENTS.exists() else None
    warnings = []
    if age_hours > 0:
        warnings.append(f'Traffic history is {age_hours:g} hours older than the forecast origin. This is an unvalidated long-range scenario; recent inputs are missing.')
    if events_hash != bundle['manifest']['custom_events_sha256']:
        warnings.append('Custom calendar events changed after training. Retrain before interpreting their learned effects.')
    if dataset == 'volume':
        warnings.append('Daily counts have unverified measurement provenance. Latest underlying observations are from August 2024.')
    return {'timestamp': str(target), 'entity': entity, 'model': bundle['winner'], 'prediction': primary,
            'lower': float(max(0, primary - radius)) if age_hours == 0 else None,
            'upper': float(primary + radius) if age_hours == 0 else None,
            'latest_history_available_at': str(available_at), 'history_age_hours': age_hours,
            'weather_at_origin': str(row.weather_last.iloc[0]), 'weather_assumed': weather_assumed,
            'holiday_name': str(row.holiday_name.iloc[0]), 'is_festival': bool(row.is_festival.iloc[0]),
            'festival_window': bool(row.festival_window.iloc[0]),
            'full_model_context_prediction': context_prediction,
            'weather_model_response': scenario_model_value - context_prediction,
            'calendar_model_response': scenario_model_value - no_festival_value,
            'scenario_model_prediction': scenario_model_value,
            'manual_uplift_pct': float(uplift_pct), 'manual_uplift_amount': scenario_model_value * uplift_pct / 100,
            'scenario_prediction': scenario_model_value * (1 + uplift_pct / 100),
            'warnings': warnings,
            'scenario_note': 'Feature contrasts describe the full boosting model, which may differ from the validation-selected forecast model. Manual uplift is a user assumption, not a learned effect.'}
