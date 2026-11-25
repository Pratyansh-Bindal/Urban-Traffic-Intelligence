"""Road + local date/time -> recorded daily lookup or a stored, labelled estimate."""
import hashlib

import numpy as np
import pandas as pd

from .context import add_context, condition_frame, road_context
from .explanations import prediction_summary
from .settings import EVENTS, TIMEZONE


ANNUAL_VOLUME_GROWTH = 0.10
PRE_FESTIVAL_VOLUME_CHANGE = 0.05
FESTIVAL_DAY_VOLUME_CHANGE = -0.05
RAINY_HOUR_VOLUME_CHANGE = 0.05


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
    code = row.get('weather_code')
    if pd.notna(code):
        result['weather_code'] = int(code)
    for name in ('temperature', 'humidity', 'precipitation'):
        value = row.get(name)
        result[name] = float(value) if value is not None and pd.notna(value) else None
    return result


def _daily_weather(frame, day):
    if frame is None or frame.empty:
        return 'Unknown'
    categories = frame.loc[frame.timestamp.dt.normalize().eq(day), 'condition']
    categories = categories.loc[categories.ne('Unknown')]
    return str(categories.mode().iloc[0]) if not categories.empty else 'Unknown'


def _without_festival_signal(frame):
    """Use a comparable baseline before applying the explicit festival assumption."""
    return frame.assign(
        is_holiday=frame.is_holiday.where(frame.is_festival.eq(0), 0),
        is_festival=0,
        festival_window=frame.is_custom_event,
        festival_name='None',
        days_to_festival=60,
        days_since_festival=60,
    )


def _daily_prediction(bundle, row):
    model = bundle['models']['daily_volume']
    if row.weather_context.iloc[0] != 'Unknown':
        return float(model.predict(row)[0])
    # Marginalise unavailable weather rather than inventing a future weather report.
    priors = bundle['weather_priors'][int(row.month.iloc[0])]
    variants = pd.concat([row.assign(weather_context=name) for name in priors], ignore_index=True)
    return float(np.dot(model.predict(variants), list(priors.values())))


def _growth_adjustment(day, archive_end, base_daily_estimate):
    """Compound the assumed growth rate after the available count archive ends."""
    if pd.isna(archive_end) or day <= archive_end:
        return None
    years = day.year - archive_end.year
    anniversary = archive_end + pd.DateOffset(years=years)
    if anniversary > day:
        years -= 1
        anniversary = archive_end + pd.DateOffset(years=years)
    next_anniversary = archive_end + pd.DateOffset(years=years + 1)
    elapsed_years = years + (day - anniversary) / (next_anniversary - anniversary)
    return {'annual_rate': ANNUAL_VOLUME_GROWTH, 'archive_end': str(archive_end.date()),
            'elapsed_years': float(elapsed_years),
            'multiplier': float((1 + ANNUAL_VOLUME_GROWTH) ** elapsed_years),
            'base_daily_estimate': base_daily_estimate}


def _profile(bundle, journey, road, day, weather_frame, now, *, neutral=False, volume_baseline=False):
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
    if volume_baseline:
        queries = _without_festival_signal(queries)
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


def _apply_volume_assumptions(bundle, journey, road, day, hour, weather_frame, current, row, profile):
    """Build volume from a festival-neutral, dry-for-rain baseline."""
    days_to_festival = int(row.days_to_festival.iloc[0])
    if row.is_festival.iloc[0]:
        phase, festival_rate = 'festival_day', FESTIVAL_DAY_VOLUME_CHANGE
    elif 1 <= days_to_festival <= 3:
        phase, festival_rate = 'before_festival', PRE_FESTIVAL_VOLUME_CHANGE
    else:
        phase, festival_rate = 'regular', 0.0

    rain_hours = pd.Series(False, index=profile.index)
    if weather_frame is not None and not weather_frame.empty:
        rainy_times = weather_frame.loc[weather_frame.condition.eq('Rain'), 'timestamp']
        rain_hours = profile.timestamp.isin(rainy_times)
    baseline_weather_frame = weather_frame
    if rain_hours.any():
        baseline_weather_frame = weather_frame.assign(condition=weather_frame.condition.replace('Rain', 'Clear'))
    baseline_weather = _daily_weather(baseline_weather_frame, day)
    baseline_row = _without_festival_signal(row).assign(weather_context=baseline_weather)
    baseline_total = _daily_prediction(bundle, baseline_row)
    baseline_profile, _ = _profile(bundle, journey, road, day, baseline_weather_frame, current, volume_baseline=True)
    rainy_share = float(baseline_profile.loc[rain_hours, 'share'].sum())
    daily_rain_rate = RAINY_HOUR_VOLUME_CHANGE * rainy_share
    rain_multipliers = 1 + rain_hours.astype(float) * RAINY_HOUR_VOLUME_CHANGE
    profile['share'] = baseline_profile.share * rain_multipliers / (1 + daily_rain_rate)
    condition_multipliers = (1 + festival_rate) * rain_multipliers
    adjusted_total = baseline_total * (1 + festival_rate) * (1 + daily_rain_rate)
    assumptions = {
        'festival_phase': phase, 'festival_rate': festival_rate, 'days_to_festival': days_to_festival,
        'rainy_hours': [int(timestamp.hour) for timestamp in profile.loc[rain_hours, 'timestamp']],
        'rainy_hour_rate': RAINY_HOUR_VOLUME_CHANGE, 'daily_rain_rate': daily_rain_rate,
        'selected_hour_rainy': bool(rain_hours.loc[profile.timestamp.eq(hour)].iloc[0]),
        'baseline_weather': baseline_weather, 'baseline_daily_estimate': baseline_total,
    }
    return adjusted_total, condition_multipliers, assumptions


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
    daily_weather = _daily_weather(weather_frame, day)
    if observed_daily is not None:
        daily_weather = str(actual.weather.iloc[0])
    query = road_data.iloc[[-1]].copy()
    query['timestamp'], query['y'], query['weather'] = day, np.nan, daily_weather
    for name in ('speed', 'temperature', 'humidity'):
        if name in query:
            query[name] = np.nan
    features = condition_frame(pd.concat([prior, query], ignore_index=True), use_history=True, as_of=current)
    row = features.loc[features.timestamp.eq(day)].copy()
    model_context_value = _daily_prediction(bundle, row)
    profile, routes = _profile(bundle, journey, road, day, weather_frame, current)
    growth = assumptions = None
    condition_multipliers = pd.Series(1.0, index=profile.index)
    daily_estimate = observed_daily
    if observed_daily is None:
        daily_estimate, condition_multipliers, assumptions = _apply_volume_assumptions(
            bundle, journey, road, day, hour, weather_frame, current, row, profile
        )
        completed = volume.timestamp + pd.Timedelta('1D') <= current
        archive_end = volume.loc[completed, 'timestamp'].max()
        growth = _growth_adjustment(day, archive_end, daily_estimate)
    growth_factor = growth['multiplier'] if growth else 1.0
    daily_estimate *= growth_factor
    neutral, _ = _profile(bundle, journey, road, day, None, current,
                          neutral=True, volume_baseline=assumptions is not None)
    reference_shares = neutral.share * condition_multipliers * growth_factor
    profile['vehicles'] = daily_estimate * profile.share
    selected = profile.loc[profile.timestamp.eq(hour)].iloc[0]
    hourly_estimate = float(selected.vehicles)
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
        # Compare against the same assumed conditions so they cannot trigger an anomaly alone.
        normal_hourly = reference['median'] * reference_shares.loc[selected.name]
        hourly_scale = reference['scale'] * reference_shares.loc[selected.name]
        volume_z = (hourly_estimate - normal_hourly) / hourly_scale
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
    weather_response = model_context_value - _daily_prediction(bundle, no_weather_row)
    no_calendar = row.copy()
    for name in ('is_holiday', 'is_festival', 'festival_window', 'is_custom_event'):
        no_calendar[name] = 0
    no_calendar['festival_name'] = 'None'
    no_calendar['days_to_festival'] = no_calendar['days_since_festival'] = 60
    calendar_response = model_context_value - _daily_prediction(bundle, no_calendar)
    context = road_context(road)
    selected_context = add_context(pd.DataFrame({'timestamp': [target], 'entity': [road], 'label': [road]}), as_of=current).iloc[0]
    profile_rows = [{'timestamp': str(r.timestamp), 'estimated_vehicles': float(r.vehicles),
                     'usual_vehicles': float(reference['median'] * reference_shares.loc[r.Index]) if reference else None,
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
        'growth_adjustment': growth, 'volume_assumptions': assumptions,
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
        'allocation_note': (
            'Hourly volume follows the baseline travel pattern with explicit festival and rain adjustments.'
            if assumptions else bundle['report']['hourly_volume_protocol']
        ),
        'warnings': warnings,
    }
    result['summary'] = prediction_summary(result)
    return result
