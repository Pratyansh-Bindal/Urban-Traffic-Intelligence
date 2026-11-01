"""Road + local date/time -> recorded daily lookup or a stored, labelled estimate."""
import hashlib

import numpy as np
import pandas as pd

from .context import add_context, condition_frame, road_context
from .explanations import prediction_summary
from .settings import EVENTS, TIMEZONE


def local_time(value):
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError('Choose a valid date and time.')
    return result.tz_convert(TIMEZONE).tz_localize(None) if result.tzinfo is not None else result


def _weather_at(frame, timestamp):
    if frame is None or frame.empty:
        return {'condition': 'Unknown', 'temperature': None, 'humidity': None, 'precipitation': None}
    selected = frame.loc[frame.timestamp.eq(timestamp.floor('h'))]
    if selected.empty:
        return {'condition': 'Unknown', 'temperature': None, 'humidity': None, 'precipitation': None}
    row = selected.iloc[0]
    result = {'condition': str(row.get('condition', 'Unknown'))}
    for name in ('temperature', 'humidity', 'precipitation'):
        value = row.get(name)
        result[name] = float(value) if value is not None and pd.notna(value) else None
    return result


def _daily_prediction(bundle, row):
    model = bundle['models']['daily_volume']
    if row.weather_context.iloc[0] != 'Unknown':
        return float(model.predict(row)[0])
    # Marginalise unavailable weather rather than inventing a future weather report.
    priors = bundle['weather_priors'][int(row.month.iloc[0])]
    variants = pd.concat([row.assign(weather_context=name) for name in priors], ignore_index=True)
    return float(np.dot(model.predict(variants), list(priors.values())))


def _profile(bundle, journey, road, day, weather_frame, now, *, neutral=False):
    context = road_context(road)
    routes = journey[['entity', 'label']].drop_duplicates()
    if context['reference_route'] is not None:
        routes = routes.loc[routes.label.eq(context['reference_route'])]
    if routes.empty:
        raise ValueError('No travel-time reference is available for this road.')
    hours = pd.DataFrame({'timestamp': pd.date_range(day, periods=24, freq='h')})
    queries = hours.merge(routes, how='cross')
    weather = [_weather_at(None if neutral else weather_frame, t) for t in queries.timestamp]
    # TML records precipitation as Dry/Rain, while the provider distinguishes
    # Clear/Overcast. Keep the model's training vocabulary at inference time.
    queries['weather'] = ['Dry' if w['condition'] in ('Clear', 'Overcast') else w['condition'] for w in weather]
    queries['temperature'] = [w['temperature'] for w in weather]
    queries['humidity'] = [w['humidity'] for w in weather]
    queries = condition_frame(queries, as_of=now)
    queries['pace'] = np.maximum(.001, bundle['models']['travel_pace'].predict(queries))
    profile = queries.groupby('timestamp').pace.median().reindex(hours.timestamp).reset_index()
    profile['source'] = 'ML travel-pattern estimate'
    # For a retrospective query prefer the available recorded route estimates.
    # Future profiles never read a future source observation.
    if not neutral and day <= now.normalize():
        observed = journey.loc[journey.entity.isin(routes.entity) & journey.timestamp.dt.normalize().eq(day)
                               & (journey.timestamp + pd.Timedelta('1h') <= now)].copy()
        if not observed.empty:
            observed['pace'] = observed.y / observed.distance
            values = observed.groupby('timestamp').pace.median()
            matched = profile.timestamp.map(values)
            profile.loc[matched.notna(), 'pace'] = matched.dropna().to_numpy()
            profile.loc[matched.notna(), 'source'] = 'Recorded reference-route travel estimate'
    profile['share'] = profile.pace / profile.pace.sum()
    return profile, routes


def robust_reference(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 8:
        return None
    median = float(np.median(values))
    mad = float(np.median(abs(values - median)))
    iqr = float(np.quantile(values, .75) - np.quantile(values, .25))
    scale = mad * 1.4826 if mad > 1e-9 else iqr / 1.349 if iqr > 1e-9 else max(abs(median) * .01, 1e-6)
    return {'median': median, 'scale': scale, 'count': len(values)}


def classify_traffic_volume(hourly_estimate, daily_counts):
    """Three volume bands from earlier daily-count terciles / 24, not road capacity."""
    if not np.isfinite(hourly_estimate) or hourly_estimate < 0:
        raise ValueError('Traffic volume must be a finite, nonnegative estimate.')
    counts = np.asarray(daily_counts, dtype=float)
    counts = counts[np.isfinite(counts) & (counts > 0)]
    value = int(round(float(hourly_estimate)))
    result = {'level': 'Unavailable', 'value': value, 'unit': 'estimated vehicles/hour',
              'lower_threshold': None, 'upper_threshold': None, 'reference_count': len(counts),
              'method': 'Earlier daily-count terciles divided by 24; estimated hourly reference.',
              'boundary_rule': 'Below lower: Low; above upper: High; both boundaries included in Moderate.',
              'minimum_reference_days': 8}
    if len(counts) < result['minimum_reference_days']:
        return result
    lower, upper = np.quantile(counts, [1 / 3, 2 / 3]) / 24
    # Round cutoffs outward and compare the displayed whole-vehicle estimate,
    # so a rounded value on a visible boundary cannot get a conflicting label.
    lower, upper = int(np.floor(lower)), int(np.ceil(upper))
    level = 'Low' if value < lower else 'High' if value > upper else 'Moderate'
    result.update(level=level, lower_threshold=lower, upper_threshold=upper)
    return result


def estimate_traffic(bundle, volume, journey, road, timestamp, *, weather_frame=None, weather_metadata=None, now=None):
    target = local_time(timestamp)
    current = local_time(now if now is not None else pd.Timestamp.now(tz=TIMEZONE))
    if road not in volume.entity.unique():
        raise ValueError('Select a road represented in the vehicle-volume dataset.')
    day = target.normalize()
    hour = target.floor('h')
    future = target >= current
    road_data = volume.loc[volume.entity.eq(road)].sort_values('timestamp')
    # Completed records only. Forecast records in SQLite are never loaded here.
    prior = road_data.loc[road_data.timestamp + pd.Timedelta('1D') <= min(day, current)].copy()
    actual = road_data.loc[road_data.timestamp.eq(day) & (road_data.timestamp + pd.Timedelta('1D') <= current)]
    observed_daily = float(actual.y.iloc[0]) if not actual.empty else None
    weather = _weather_at(weather_frame, target)
    if weather_frame is not None and not weather_frame.empty:
        categories = weather_frame.loc[weather_frame.timestamp.dt.normalize().eq(day), 'condition']
        categories = categories.loc[categories.ne('Unknown')]
        daily_weather = str(categories.mode().iloc[0]) if not categories.empty else 'Unknown'
    else:
        daily_weather = 'Unknown'
    if observed_daily is not None:
        daily_weather = str(actual.weather.iloc[0])
    query = road_data.iloc[[-1]].copy()
    query['timestamp'], query['y'], query['weather'] = day, np.nan, daily_weather
    for name in ('speed', 'temperature', 'humidity'):
        if name in query:
            query[name] = np.nan
    features = condition_frame(pd.concat([prior, query], ignore_index=True), use_history=True, as_of=current)
    row = features.loc[features.timestamp.eq(day)].copy()
    daily_estimate = observed_daily if observed_daily is not None else _daily_prediction(bundle, row)
    profile, routes = _profile(bundle, journey, road, day, weather_frame, current)
    neutral, _ = _profile(bundle, journey, road, day, None, current, neutral=True)
    profile['vehicles'] = daily_estimate * profile.share
    selected = profile.loc[profile.timestamp.eq(hour)].iloc[0]
    hourly_estimate = float(selected.vehicles)
    neutral_share = float(neutral.loc[neutral.timestamp.eq(hour), 'share'].iloc[0])
    warnings = []
    if observed_daily is None and not prior.empty:
        age_days = max(0, (day - (prior.timestamp.max() + pd.Timedelta('1D'))).days)
        if age_days > 7:
            warnings.append(f'The latest vehicle-count input for this road is {prior.timestamp.max():%d %b %Y}. This estimate extrapolates beyond that archive.')
    else:
        age_days = 0
    metadata = weather_metadata or {'status': 'unavailable', 'kind': 'not_supplied', 'provider': 'Open-Meteo'}
    if weather['condition'] == 'Unknown':
        warnings.append(metadata.get('message', 'Weather could not be obtained. The daily model averages historical seasonal weather categories; the travel model uses its missing-weather context.'))
    if observed_daily is None and day <= pd.Timestamp(bundle['report']['models']['daily_volume']['splits']['train']['last_date']):
        warnings.append('This is a retrospective estimate made with a later-trained model, rather than a historical forecasting backtest.')
    if observed_daily is not None and not (journey.timestamp < day).any():
        warnings.append('The daily count is recorded; its hourly breakdown is a retrospective estimate informed by later travel-pattern data.')
    events_hash = hashlib.sha256(EVENTS.read_bytes()).hexdigest() if EVENTS.exists() else None
    if events_hash != bundle['manifest']['custom_events_sha256']:
        warnings.append('Custom events changed after training. Retrain to include the updated calendar consistently.')
    # Use only earlier real observations for anomaly references, with a stale-history fallback.
    reference_values = prior.loc[prior.timestamp >= day - pd.Timedelta('28D'), 'y']
    if len(reference_values) < 8 and not prior.empty:
        reference_values = prior.loc[prior.timestamp >= prior.timestamp.max() - pd.Timedelta('90D'), 'y']
    classification = classify_traffic_volume(hourly_estimate, reference_values)
    reference = robust_reference(reference_values)
    volume_z = None
    normal_hourly = None
    observed_daily_z = None
    if reference is not None:
        normal_hourly = reference['median'] * neutral_share
        volume_z = (hourly_estimate - normal_hourly) / (reference['scale'] * neutral_share)
        if observed_daily is not None:
            observed_daily_z = (observed_daily - reference['median']) / reference['scale']
    route_prior = journey.loc[journey.entity.isin(routes.entity)
                              & (journey.timestamp + pd.Timedelta('1h') <= min(hour, current))].copy()
    route_prior['pace'] = route_prior.y / route_prior.distance
    if not route_prior.empty:
        recent = route_prior.loc[route_prior.timestamp >= route_prior.timestamp.max() - pd.Timedelta('90D')]
    else:
        recent = route_prior
    matched = recent.loc[recent.timestamp.dt.hour.eq(hour.hour)
                         & (recent.timestamp.dt.day_of_week.ge(5).eq(hour.day_of_week >= 5))]
    pace_reference = robust_reference(matched.pace)
    pace_z = (float(selected.pace) - pace_reference['median']) / pace_reference['scale'] if pace_reference else None
    scores = [float(z) for z in (volume_z, pace_z) if z is not None and np.isfinite(z)]
    largest = max(scores, key=abs) if scores else None
    flagged = largest is not None and abs(largest) > 3
    state = ('Predicted unusual traffic' if future else 'Estimated unusual traffic') if flagged else ('Within usual range' if scores else 'Insufficient earlier reference')
    no_weather_row = row.assign(weather_context='Unknown')
    model_context_value = _daily_prediction(bundle, row)
    weather_response = model_context_value - _daily_prediction(bundle, no_weather_row)
    no_calendar = row.copy()
    for name in ('is_holiday', 'is_festival', 'festival_window', 'is_custom_event'):
        no_calendar[name] = 0
    no_calendar['festival_name'] = 'None'
    no_calendar['days_to_festival'] = no_calendar['days_since_festival'] = 60
    calendar_response = model_context_value - _daily_prediction(bundle, no_calendar)
    context = road_context(road)
    selected_context = add_context(pd.DataFrame({'timestamp': [target], 'entity': [road], 'label': [road]}), as_of=current).iloc[0]
    normal_shares = neutral.set_index('timestamp').share
    profile_rows = [{'timestamp': str(r.timestamp), 'estimated_vehicles': float(r.vehicles),
                     'usual_vehicles': float(reference['median'] * normal_shares.loc[r.timestamp]) if reference else None,
                     'share': float(r.share), 'travel_pace': float(r.pace), 'travel_source': r.source}
                    for r in profile.itertuples()]
    daily_quality = bundle['report']['models']['daily_volume']
    pace_quality = bundle['report']['models']['travel_pace']
    result = {
        'road': road, 'requested_at': str(target), 'period_start': str(hour),
        'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'input_cutoff_local': str(current),
        'period_end': str(hour + pd.Timedelta('1h')), 'timezone': TIMEZONE,
        'kind': 'Forecast estimate' if future else 'Historical lookup + hourly estimate' if observed_daily is not None else 'Historical estimate',
        'hourly_estimate': hourly_estimate, 'hourly_unit': 'vehicles in the selected hour',
        'daily_estimate': float(daily_estimate), 'observed_daily': observed_daily,
        'normal_hourly': float(normal_hourly) if normal_hourly is not None else None,
        'traffic_level': classification['level'], 'traffic_classification': classification,
        'weather': weather, 'daily_weather_context': daily_weather,
        'weather_metadata': metadata,
        'calendar': {'holiday_name': str(row.holiday_name.iloc[0]), 'is_holiday': bool(row.is_holiday.iloc[0]),
                     'is_festival': bool(row.is_festival.iloc[0]), 'festival_window': bool(row.festival_window.iloc[0]),
                     'is_custom_event': bool(row.is_custom_event.iloc[0])},
        'day_part': str(selected_context.day_part), 'corridor': context,
        'reference_routes': routes.to_dict('records'),
        'reference_travel_pace': float(selected.pace), 'reference_travel_source': str(selected.source),
        'anomaly': {'status': state, 'flagged': bool(flagged), 'threshold': 3.0,
                    'score': largest, 'volume_z': float(volume_z) if volume_z is not None else None,
                    'travel_pace_z': float(pace_z) if pace_z is not None else None,
                    'observed_daily_z': float(observed_daily_z) if observed_daily_z is not None else None,
                    'direction': 'Higher than usual' if largest is not None and largest > 0 else 'Lower than usual' if largest is not None and largest < 0 else 'No deviation score',
                    'volume_reference_count': reference['count'] if reference else 0,
                    'travel_reference_count': pace_reference['count'] if pace_reference else 0,
                    'reference_age_days': age_days,
                    'note': 'Robust Z-score flags deviations from earlier records. The hourly volume score uses an estimated allocation; an alert is not a confirmed incident.'},
        'model': {'daily_volume': daily_quality['model'], 'travel_pace': pace_quality['model'],
                  'version': bundle['manifest']['created_at_utc'],
                  'daily_test_mae': daily_quality['selected_test']['mae'],
                  'travel_pace_test_mae': pace_quality['selected_test']['mae'],
                  'hourly_count_test_mae': None,
                  'evaluation_protocol': bundle['report']['evaluation_protocol']},
        'model_responses': {'weather_daily_vehicles': float(weather_response), 'calendar_daily_vehicles': float(calendar_response)},
        'latest_volume_input': str(prior.timestamp.max()) if not prior.empty else None,
        'latest_travel_input': str(route_prior.timestamp.max()) if not route_prior.empty else None,
        'hourly_profile': profile_rows,
        'history': [{'date': str(r.timestamp), 'vehicles': float(r.y)} for r in prior.tail(40).itertuples()],
        'allocation_note': bundle['report']['hourly_volume_protocol'], 'warnings': warnings,
    }
    result['summary'] = prediction_summary(result)
    return result
