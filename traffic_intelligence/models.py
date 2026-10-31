from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import feature_columns
from .settings import SEED


class HistoricalBaseline:
    def __init__(self, kind='seasonal'):
        self.kind = kind

    def fit(self, frame, y):
        data = frame.copy()
        data['target'] = np.asarray(y)
        self.global_median = float(np.median(y))
        self.entity_medians = data.groupby('entity').target.median().to_dict()
        self.seasonal_medians = data.groupby(['entity', 'weekday', 'hour']).target.median().to_dict()
        return self

    def predict(self, frame):
        fallback = frame.entity.map(self.entity_medians).fillna(self.global_median).to_numpy(dtype=float)
        if self.kind == 'naive':
            return np.where(frame.lag_last.notna(), frame.lag_last, fallback)
        if self.kind == 'entity':
            return fallback
        return np.array([self.seasonal_medians.get((e, int(d), int(h)), default) for e,d,h,default in
                         zip(frame.entity, frame.weekday, frame.hour, fallback)])


@dataclass
class ForecastModel:
    name: str
    columns: list[str]
    estimator: object

    def fit(self, frame):
        self.estimator.fit(frame[self.columns], frame.y.to_numpy(dtype=float))
        return self

    def predict(self, frame):
        return np.maximum(0.0, np.asarray(self.estimator.predict(frame[self.columns]), dtype=float))


def build_model(name: str, *, weather=True, festivals=True, quick=False, columns=None) -> ForecastModel:
    columns = columns if columns is not None else feature_columns(weather=weather, festivals=festivals)
    if name in ('Last observation', 'Road / route median', 'Weekday-hour median'):
        kind = {'Last observation': 'naive', 'Road / route median': 'entity', 'Weekday-hour median': 'seasonal'}[name]
        return ForecastModel(name, columns, HistoricalBaseline(kind))
    categorical = [c for c in columns if c in ('entity', 'weather_last', 'festival_name',
                                               'weather_context', 'day_part', 'corridor_type')]
    numeric = [c for c in columns if c not in categorical]
    preprocess = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True), numeric),
        ('category', Pipeline([('fill', SimpleImputer(strategy='constant', fill_value='Unknown')),
                               ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical),
    ], sparse_threshold=0)
    if name == 'Random Forest':
        estimator = RandomForestRegressor(n_estimators=40 if quick else 120, max_depth=14, min_samples_leaf=10,
                                           n_jobs=2, random_state=SEED)
    else:
        estimator = HistGradientBoostingRegressor(max_iter=50 if quick else 150, max_leaf_nodes=15,
                                                   learning_rate=.07, min_samples_leaf=20, l2_regularization=2,
                                                   early_stopping=False, random_state=SEED)
    return ForecastModel(name, columns, Pipeline([('preprocess', preprocess), ('model', estimator)]))


def regression_metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return {'mae': float(mean_absolute_error(actual, predicted)),
            'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
            'r2': float(r2_score(actual, predicted)), 'n': len(actual)}


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dates = np.sort(frame.timestamp.dt.normalize().unique())
    if len(dates) < 40:
        raise ValueError('At least 40 observed dates are required for independent chronological partitions.')
    a, b, c = int(.70 * len(dates)), int(.775 * len(dates)), int(.85 * len(dates))
    cuts = [('train', dates[:a]), ('selection', dates[a:b]), ('calibration', dates[b:c]), ('test', dates[c:])]
    out = frame.copy()
    summaries = {}
    for name, selected_dates in cuts:
        mask = out.timestamp.dt.normalize().isin(selected_dates)
        out.loc[mask, 'partition'] = name
        summaries[name] = {'first_date': str(pd.Timestamp(selected_dates[0]).date()),
                            'last_date': str(pd.Timestamp(selected_dates[-1]).date()),
                            'dates': len(selected_dates), 'rows': int(mask.sum())}
    return out, summaries


def conformal_radius(errors, coverage=.90) -> float:
    values = np.sort(np.asarray(errors, float))
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError('Calibration needs finite held-out errors.')
    rank = min(len(values), int(np.ceil((len(values) + 1) * coverage)))
    return float(values[rank - 1])
