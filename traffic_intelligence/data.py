"""Load immutable inputs, keep targets separate, and report rather than erase spikes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .settings import ARTIFACTS, RAW


def _require(frame: pd.DataFrame, columns: list[str], source: str):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{source}: missing columns {missing}')


def _summary(frame: pd.DataFrame) -> dict:
    counts = frame.groupby('entity').size()
    return {'usable_rows': len(frame), 'entities': len(counts),
            'first_period': str(frame.timestamp.min()), 'last_period': str(frame.timestamp.max()),
            'rows_per_entity': {str(k): int(v) for k, v in counts.items()},
            'weather_observed_fraction': float(frame.weather.ne('Unknown').mean())}


def load_volume(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    path = path or RAW / 'bangalore_volume.csv'
    if not path.exists():
        raise FileNotFoundError('Daily CSV missing. Run the fetch command with --volume PATH first.')
    raw = pd.read_csv(path)
    _require(raw, ['Date', 'Road/Intersection Name', 'Area Name', 'Traffic Volume', 'Weather Conditions'], 'Volume CSV')
    frame = pd.DataFrame({
        'timestamp': pd.to_datetime(raw['Date'], errors='coerce'),
        'entity': raw['Road/Intersection Name'].str.strip(),
        'label': raw['Road/Intersection Name'].str.strip(),
        'area': raw['Area Name'].str.strip(),
        'y': pd.to_numeric(raw['Traffic Volume'], errors='coerce'),
        'weather': raw['Weather Conditions'].fillna('Unknown').str.strip(),
        'speed': pd.to_numeric(raw.get('Average Speed'), errors='coerce'),
    })
    valid = frame.timestamp.notna() & frame.entity.notna() & frame.y.gt(0)
    frame = frame.loc[valid].copy()
    if frame.duplicated(['entity', 'timestamp']).any():
        raise ValueError('Volume CSV contains duplicate road-date keys; resolve their meaning before training.')
    frame = frame.sort_values(['entity', 'timestamp']).reset_index(drop=True)
    span = int((frame.timestamp.max() - frame.timestamp.min()).days) + 1
    audit = _summary(frame)
    audit.update({'raw_rows': len(raw), 'raw_columns': len(raw.columns), 'excluded_invalid_rows': int((~valid).sum()),
                  'missing_raw_cells': int(raw.isna().sum().sum()), 'exact_duplicates': int(raw.duplicated().sum()),
                  'calendar_days': span, 'distinct_dates': int(frame.timestamp.nunique()),
                  'expected_road_dates': int(span * frame.entity.nunique()),
                  'coverage_fraction': len(frame) / (span * frame.entity.nunique()),
                  'target': 'Daily vehicle volume', 'provenance': 'Unverified collection origin; derived and capped fields exist.',
                  'excluded_predictors': ['Environmental Impact', 'same-period speed', 'same-period congestion',
                                          'same-period capacity utilization', 'same-period incidents', 'same-period weather'],
                  'frequency': 'Daily observations with gaps; no time-of-day data.'})
    if 'Environmental Impact' in raw:
        err = pd.to_numeric(raw['Environmental Impact'], errors='coerce') - (50 + .002 * pd.to_numeric(raw['Traffic Volume'], errors='coerce'))
        audit['environmental_formula'] = {'formula': '50 + 0.002 * Traffic Volume', 'max_absolute_error': float(err.abs().max()),
                                          'matching_rows': int(err.abs().le(1e-10).sum())}
    audit['capped_fraction'] = {c: float(pd.to_numeric(raw[c], errors='coerce').eq(limit).mean()) for c, limit in
                                [('Road Capacity Utilization', 100), ('Travel Time Index', 1.5), ('Congestion Level', 100)] if c in raw}
    return frame, audit


def rain_category(flag: str, has_weather: bool) -> str:
    text = str(flag).lower().strip()
    if not has_weather:
        return 'Unknown'
    if any(word in text for word in ('rain', 'shower', 'thunder', 'drizzle')) and 'no precipitation' not in text:
        return 'Rain'
    if 'fog' in text:
        return 'Fog'
    if 'wind' in text:
        return 'Windy'
    # Blank rsi_flag means no precipitation ONLY where accompanying weather exists.
    return 'Dry'


def load_journey(path: Path | None = None, as_of=None) -> tuple[pd.DataFrame, dict]:
    is_default = path is None
    path = path or RAW / 'csv-traffic-bangalore.csv'
    if not path.exists():
        raise FileNotFoundError('Traffic Monitor Lizard snapshot missing. Run python -m traffic_intelligence fetch.')
    raw = pd.read_csv(path, keep_default_na=True)
    _require(raw, ['date', 'time', 'route_code', 'duration', 'distance'], 'Traffic Monitor Lizard')
    timestamp = pd.to_datetime(raw.date.astype(str) + ' ' + raw.time.astype(str), errors='coerce', format='mixed')
    canonical = pd.DataFrame({'observed_at': timestamp, 'entity': raw.route_code.str.strip(),
                              'y': pd.to_numeric(raw.duration, errors='coerce'),
                              'distance': pd.to_numeric(raw.distance, errors='coerce')})
    for source, target in [('temp', 'temperature'), ('realfeel', 'feels_like'), ('humidity', 'humidity'), ('aqi', 'aqi')]:
        canonical[target] = pd.to_numeric(raw[source], errors='coerce') if source in raw else np.nan
    # Weather errors become missing weather, never missing traffic.
    canonical.loc[~canonical.temperature.between(-10, 60), 'temperature'] = np.nan
    canonical.loc[~canonical.feels_like.between(-20, 70), 'feels_like'] = np.nan
    canonical.loc[~canonical.humidity.between(0, 100), 'humidity'] = np.nan
    canonical.loc[~canonical.aqi.between(0, 1000), 'aqi'] = np.nan
    flags = raw['rsi_flag'].astype('string').fillna('') if 'rsi_flag' in raw else pd.Series('', index=raw.index)
    has_weather = canonical[['temperature', 'humidity']].notna().any(axis=1)
    canonical['weather'] = [rain_category(f, bool(h)) for f, h in zip(flags, has_weather)]
    valid = canonical.observed_at.notna() & canonical.entity.notna() & canonical.y.gt(0) & canonical.distance.gt(0)
    canonical = canonical.loc[valid].copy()
    duplicates = int(canonical.duplicated(['entity', 'observed_at']).sum())
    canonical = canonical.drop_duplicates(['entity', 'observed_at'], keep='last')
    canonical['timestamp'] = canonical.observed_at.dt.floor('h')
    # The hour is available only when it has finished. No partial latest hour is used.
    now_local = pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None)
    if as_of is not None:
        now_local = pd.Timestamp(as_of)
        if now_local.tzinfo is not None:
            now_local = now_local.tz_convert('Asia/Kolkata').tz_localize(None)
    elif is_default and (RAW / 'sources.json').exists():
        downloaded = json.loads((RAW / 'sources.json').read_text()).get('journey', {}).get('downloaded_at_utc')
        if downloaded:
            now_local = min(now_local, pd.Timestamp(downloaded).tz_convert('Asia/Kolkata').tz_localize(None))
    canonical = canonical.loc[canonical.timestamp + pd.Timedelta('1h') <= now_local]
    groups = canonical.groupby(['entity', 'timestamp'], sort=True)
    frame = groups.agg(y=('y', 'median'), distance=('distance', 'median'), temperature=('temperature', 'median'),
                       feels_like=('feels_like', 'median'), humidity=('humidity', 'median'), aqi=('aqi', 'median'),
                       weather=('weather', lambda s: s.mode().iloc[0]), source_readings=('y', 'size'),
                       latest_reading=('observed_at', 'max')).reset_index()
    routes_path = RAW / 'csv-routes-bangalore.csv'
    routes = pd.read_csv(routes_path) if routes_path.exists() else pd.DataFrame(columns=['route_code', 'label_short'])
    mapping = routes.drop_duplicates('route_code').set_index('route_code')['label_short'].to_dict()
    frame['label'] = frame.entity.map(lambda entity: mapping.get(entity, entity))
    frame['area'] = 'Bengaluru monitored route'
    frame = frame.sort_values(['entity', 'timestamp']).reset_index(drop=True)
    audit = _summary(frame)
    audit.update({'raw_rows': len(raw), 'raw_columns': len(raw.columns), 'excluded_invalid_rows': int((~valid).sum()),
                  'duplicate_route_timestamps_removed': duplicates, 'hourly_aggregation': 'Median of readings in [hour, hour+1h); available at hour+1h.',
                  'target': 'Google Maps estimated route duration, minutes; not measured travel times or vehicle counts.',
                  'weather_missing_fraction': {c: float(frame[c].isna().mean()) for c in ['temperature', 'humidity', 'aqi']},
                  'raw_timestamp_range': [str(timestamp.min()), str(timestamp.max())],
                  'snapshot_as_of_local': str(now_local),
                  'distance_range_km': {str(k): {'min': float(g.distance.min()), 'max': float(g.distance.max())} for k,g in frame.groupby('entity')},
                  'excluded_predictors': ['same-hour duration-derived speed', 'same-hour weather', 'same-hour distance', 'upstream anomaly labels'],
                  'provenance': 'Traffic Monitor Lizard / thecont1; Google Maps estimates; CC BY 4.0.'})
    return frame, audit


def load_dataset(name: str):
    if name not in ('volume', 'journey'):
        raise ValueError(f'Unknown dataset: {name}')
    return load_volume() if name == 'volume' else load_journey()


def audit_all() -> dict:
    a, aa = load_volume()
    b, ba = load_journey()
    overlap = sorted(set(a.timestamp.dt.date) & set(b.timestamp.dt.date))
    fusion = {'overlapping_dates': len(overlap), 'eligible_for_join': False,
              'reason': 'No overlapping observation dates.' if not overlap else 'Route geometry and measurement compatibility have not been verified.',
              'policy': 'Separate forecast targets; never infer vehicle counts from journey duration.'}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result = {'volume': aa, 'journey': ba, 'fusion': fusion}
    (ARTIFACTS / 'audit.json').write_text(json.dumps(result, indent=2))
    return result
