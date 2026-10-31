import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from traffic_intelligence.features import assert_no_leakage, feature_columns, make_features
from traffic_intelligence.models import chronological_split


def test_future_outcomes_and_weather_cannot_change_earlier_features(daily_history):
    before = make_features(daily_history, 'volume')
    changed = daily_history.copy()
    future = changed.timestamp >= '2023-03-01'
    changed.loc[future,'y'] = 9_999_999
    changed.loc[future,'weather'] = 'Windy'
    after = make_features(changed, 'volume')
    earlier = before.timestamp <= '2023-03-01'
    assert_frame_equal(before.loc[earlier,feature_columns()].reset_index(drop=True),
                       after.loc[earlier,feature_columns()].reset_index(drop=True))


def test_same_period_measurements_never_predict_themselves(daily_history):
    before = make_features(daily_history, 'volume')
    changed = daily_history.copy()
    changed.loc[(changed.entity=='A') & (changed.timestamp=='2023-02-20'),'y'] = 10_000_000
    changed.loc[(changed.entity=='A') & (changed.timestamp=='2023-02-20'),'weather'] = 'Fog'
    after = make_features(changed, 'volume')
    row = before.entity.eq('A') & before.timestamp.eq('2023-02-20')
    assert_frame_equal(before.loc[row,feature_columns()], after.loc[row,feature_columns()])


def test_exact_lags_preserve_gaps_and_entity_boundaries(daily_history):
    keep = ~((daily_history.entity=='A') & (daily_history.timestamp=='2023-02-19'))
    engineered = make_features(daily_history.loc[keep], 'volume')
    row = engineered.loc[engineered.entity.eq('A') & engineered.timestamp.eq('2023-02-20')].iloc[0]
    assert np.isnan(row.lag_exact_1)
    expected = daily_history.loc[daily_history.entity.eq('A') & daily_history.timestamp.eq('2023-02-18'),'y'].iloc[0]
    assert row.lag_last == expected
    assert row.history_age_hours == 24
    first_b = engineered.loc[engineered.entity.eq('B')].iloc[0]
    assert pd.isna(first_b.lag_last)
    assert_no_leakage(engineered)


def test_weather_expires_across_long_gaps():
    data = pd.DataFrame({'timestamp':pd.to_datetime(['2026-01-01 08:00','2026-01-03 09:00']),
                         'entity':['A','A'],'label':['A','A'],'y':[20,30],'weather':['Rain','Unknown'],
                         'temperature':[25,np.nan],'humidity':[90,np.nan]})
    last = make_features(data,'journey').iloc[-1]
    assert last.weather_last == 'Unknown'
    assert last.weather_available == 0
    assert pd.isna(last.temperature_last)
    assert pd.isna(last.humidity_last)


def test_partitions_keep_dates_together_and_are_strictly_ordered(daily_history):
    features = make_features(daily_history, 'volume')
    data, summary = chronological_split(features)
    assert data.groupby(data.timestamp.dt.date).partition.nunique().eq(1).all()
    order = ['train','selection','calibration','test']
    for left,right in zip(order,order[1:]):
        assert data.loc[data.partition==left,'timestamp'].max() < data.loc[data.partition==right,'timestamp'].min()
    assert sum(x['rows'] for x in summary.values()) == len(data)


def test_target_derived_columns_are_not_predictors():
    prohibited = {'y','Traffic Volume','Environmental Impact','duration','distance','speed','weather','reference_median'}
    assert prohibited.isdisjoint(feature_columns())
