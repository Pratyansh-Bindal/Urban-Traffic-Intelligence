"""All outcome and weather features come from finished periods strictly before the target."""
import numpy as np
import pandas as pd

from .calendar import FESTIVAL_FEATURES, add_calendar_features
from .settings import DATASETS

HISTORY_FEATURES = ['lag_last', 'lag_exact_1', 'lag_exact_2', 'lag_exact_3', 'history_age_hours',
                    'rolling_mean_7d', 'rolling_median_28d', 'rolling_std_28d', 'history_count_28d']
BASIC_FEATURES = ['entity', 'weekday', 'is_weekend', 'month', 'hour', 'hour_sin', 'hour_cos',
                  'weekday_sin', 'weekday_cos', 'year_sin', 'year_cos']
WEATHER_FEATURES = ['weather_last', 'temperature_last', 'humidity_last', 'feels_like_last', 'aqi_last',
                    'weather_age_hours', 'weather_available']


def make_features(observations: pd.DataFrame, dataset: str) -> pd.DataFrame:
    frame = observations.sort_values(['entity', 'timestamp']).copy()
    if frame.duplicated(['entity', 'timestamp']).any():
        raise ValueError('Feature generation requires unique entity-period keys.')
    period = pd.Timedelta(DATASETS[dataset]['frequency'])
    lags = [pd.Timedelta(days=n) for n in (1, 7, 14)] if dataset == 'volume' else [pd.Timedelta(hours=n) for n in (1, 24, 168)]
    parts = []
    for entity, group in frame.groupby('entity', sort=False):
        g = group.sort_values('timestamp').copy()
        series = pd.Series(g.y.to_numpy(dtype=float), index=pd.DatetimeIndex(g.timestamp))
        g['lag_last'] = g.y.shift(1)
        for i, lag in enumerate(lags, 1):
            g[f'lag_exact_{i}'] = series.reindex(pd.DatetimeIndex(g.timestamp - lag)).to_numpy()
        prior_end = g.timestamp.shift(1) + period
        g['history_age_hours'] = (g.timestamp - prior_end).dt.total_seconds() / 3600
        g['last_history_period'] = g.timestamp.shift(1)
        g['last_history_available_at'] = prior_end
        g['rolling_mean_7d'] = series.rolling('7D', closed='left', min_periods=2).mean().to_numpy()
        rolling = series.rolling('28D', closed='left', min_periods=3)
        g['rolling_median_28d'] = rolling.median().to_numpy()
        g['rolling_std_28d'] = rolling.std().to_numpy()
        g['history_count_28d'] = series.rolling('28D', closed='left').count().to_numpy()
        max_weather_age = 48 if dataset == 'volume' else 3
        weather_present = g.weather.ne('Unknown') & g.weather.notna()
        weather_known_at = g.timestamp.where(weather_present).ffill().shift(1) + period
        weather_age = (g.timestamp - weather_known_at).dt.total_seconds() / 3600
        fresh = weather_age.between(0, max_weather_age)
        known_weather = g.weather.where(weather_present).ffill().shift(1)
        g['weather_last'] = known_weather.where(fresh, 'Unknown').fillna('Unknown')
        g['weather_age_hours'] = weather_age.where(fresh)
        g['weather_available'] = fresh.astype(int)
        for c in ['temperature', 'humidity', 'feels_like', 'aqi']:
            if c in g:
                known_at = g.timestamp.where(g[c].notna()).ffill().shift(1) + period
                age = (g.timestamp - known_at).dt.total_seconds() / 3600
                g[c + '_last'] = g[c].ffill().shift(1).where(age.between(0, max_weather_age))
            else:
                g[c + '_last'] = np.nan
        # A current source measurement can never enter features for its own period.
        parts.append(g)
    result = pd.concat(parts).sort_values(['timestamp', 'entity']).reset_index(drop=True)
    return add_calendar_features(result)


def feature_columns(weather: bool = True, festivals: bool = True) -> list[str]:
    return BASIC_FEATURES + HISTORY_FEATURES + (WEATHER_FEATURES if weather else []) + (FESTIVAL_FEATURES if festivals else [])


def assert_no_leakage(frame: pd.DataFrame):
    available = frame.last_history_available_at
    if (available.notna() & available.gt(frame.timestamp)).any():
        raise AssertionError('A feature references an unfinished observation period.')
