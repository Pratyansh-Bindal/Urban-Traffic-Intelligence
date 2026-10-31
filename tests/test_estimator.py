import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from traffic_intelligence.calendar import FESTIVAL_FEATURES
from traffic_intelligence.context import CONDITION_FEATURES, TIME_FEATURES, add_context, condition_frame
from traffic_intelligence.estimator import estimate_traffic
from traffic_intelligence.features import BASIC_FEATURES, HISTORY_FEATURES
from traffic_intelligence.models import build_model
from traffic_intelligence.settings import EVENTS
from traffic_intelligence.storage import compare_with_observations, get_estimate, list_estimates, save_estimate
import traffic_intelligence.weather as weather_module


@pytest.fixture(scope='module')
def estimator_data():
    days = pd.date_range('2023-01-01', periods=100, freq='D')
    volume_parts, journey_parts = [], []
    for road, route, label, offset in [('Ballari Road', 'AIRPORT', 'Airport Expy', 50),
                                        ('Hosur Road', 'HOSUR', 'Hosur Road', 0)]:
        volume_parts.append(pd.DataFrame({'timestamp': days, 'entity': road, 'label': road, 'area': 'Test',
                                          'y': 1000 + offset + np.arange(100) * 2,
                                          'weather': np.where(np.arange(100) % 3 == 0, 'Rain', 'Clear')}))
        times = pd.date_range('2023-01-01', periods=24 * 100, freq='h')
        pace = 1.0 + .8 * times.hour.to_series(index=range(len(times))).between(12, 14).to_numpy() + .3 * (times.day_of_week < 5)
        journey_parts.append(pd.DataFrame({'timestamp': times, 'entity': route, 'label': label,
                                           'y': pace * 20, 'distance': 20., 'temperature': 25., 'humidity': 75.,
                                           'weather': 'Dry'}))
    volume = pd.concat(volume_parts, ignore_index=True)
    journey = pd.concat(journey_parts, ignore_index=True)
    daily = condition_frame(volume, use_history=True)
    pace_data = journey.assign(y=journey.y / journey.distance)
    pace_data = condition_frame(pace_data)
    daily_columns = BASIC_FEATURES + HISTORY_FEATURES + FESTIVAL_FEATURES + TIME_FEATURES + ['weather_context']
    pace_columns = BASIC_FEATURES + FESTIVAL_FEATURES + TIME_FEATURES + CONDITION_FEATURES
    models = {'daily_volume': build_model('Boosting: full', columns=daily_columns, quick=True).fit(daily.loc[daily.timestamp < '2023-03-01']),
              'travel_pace': build_model('Boosting: full', columns=pace_columns, quick=True).fit(pace_data.loc[pace_data.timestamp < '2023-03-01'])}
    quality = {'model': 'Boosting: full', 'selected_test': {'mae': 10.}, 'splits': {'train': {'last_date': '2023-02-28'}}}
    bundle = {'models': models, 'weather_priors': {month: {'Rain': .25, 'Clear': .75} for month in range(1, 13)},
              'report': {'models': {'daily_volume': quality, 'travel_pace': quality},
                         'evaluation_protocol': 'Conditional estimation.', 'hourly_volume_protocol': 'Explicit pace allocation assumption.'},
              'manifest': {'created_at_utc': '2023-03-01T00:00:00+00:00',
                           'custom_events_sha256': hashlib.sha256(EVENTS.read_bytes()).hexdigest()}}
    return bundle, volume, journey


def test_common_time_features_do_not_force_traffic_levels():
    frame = pd.DataFrame({'timestamp': pd.to_datetime(['2026-10-31 03:00', '2026-10-31 08:00', '2026-10-31 13:00', '2026-10-31 18:00']),
                          'entity': ['Ballari Road', 'Hosur Road', 'OTHER', 'OTHER'],
                          'label': ['Ballari Road', 'Hosur Road', 'Old Airport Road', 'Airport Expy']})
    output = add_context(frame)
    assert output.is_overnight.tolist() == [1, 0, 0, 0]
    assert output.is_morning_peak.tolist() == [0, 1, 0, 0]
    assert output.is_lunch_period.tolist() == [0, 0, 1, 0]
    assert output.is_evening_peak.tolist() == [0, 0, 0, 1]
    assert output.is_airport_corridor.tolist() == [1, 0, 0, 1]
    assert 'prediction' not in output


def test_past_lookup_retains_recorded_daily_count_and_labels_hourly_estimate(estimator_data):
    bundle, volume, journey = estimator_data
    result = estimate_traffic(bundle, volume, journey, 'Ballari Road', '2023-04-05 12:30', now='2023-04-11')
    actual = volume.loc[volume.entity.eq('Ballari Road') & volume.timestamp.eq('2023-04-05'), 'y'].iloc[0]
    assert result['observed_daily'] == actual == result['daily_estimate']
    assert result['kind'] == 'Historical lookup + hourly estimate'
    assert result['period_start'] == '2023-04-05 12:00:00'
    assert sum(x['estimated_vehicles'] for x in result['hourly_profile']) == pytest.approx(actual)
    assert result['model']['hourly_count_test_mae'] is None
    assert result['reference_travel_source'] == 'Recorded reference-route travel estimate'
    assert result['anomaly']['observed_daily_z'] is not None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('target', ['2023-04-10 12:30', '2023-04-12 12:30'])
def test_upcoming_estimate_excludes_unfinished_and_future_source_observations(estimator_data, target):
    bundle, volume, journey = estimator_data
    before = estimate_traffic(bundle, volume, journey, 'Ballari Road', target, now='2023-04-10 10:30')
    changed_volume = volume.copy()
    changed_volume.loc[changed_volume.timestamp >= '2023-04-10', 'y'] = 99_999_999
    extra = changed_volume.iloc[[0]].assign(timestamp=pd.Timestamp('2023-04-12'), y=99_999_999)
    changed_volume = pd.concat([changed_volume, extra], ignore_index=True)
    changed_journey = journey.copy()
    changed_journey.loc[changed_journey.timestamp + pd.Timedelta('1h') > pd.Timestamp('2023-04-10 10:30'), 'y'] = 99_999_999
    after = estimate_traffic(bundle, changed_volume, changed_journey, 'Ballari Road', target, now='2023-04-10 10:30')
    assert after['daily_estimate'] == before['daily_estimate']
    assert after['hourly_estimate'] == before['hourly_estimate']
    assert after['observed_daily'] is None
    assert after['anomaly'] == before['anomaly']
    assert after['kind'] == 'Forecast estimate'


def test_missing_historical_record_is_estimated_without_becoming_an_observation(estimator_data):
    bundle, volume, journey = estimator_data
    volume = volume.loc[~(volume.entity.eq('Hosur Road') & volume.timestamp.eq('2023-04-05'))]
    result = estimate_traffic(bundle, volume, journey, 'Hosur Road', '2023-04-05 08:00', now='2023-04-11')
    assert result['kind'] == 'Historical estimate'
    assert result['observed_daily'] is None
    assert result['daily_estimate'] > 0


def test_provider_clear_and_overcast_use_the_travel_models_dry_category(estimator_data):
    bundle, volume, journey = estimator_data
    frame = pd.DataFrame({'timestamp': pd.date_range('2023-04-12', periods=24, freq='h'),
                           'condition': ['Clear'] * 24, 'temperature': [25.] * 24, 'humidity': [75.] * 24})
    clear = estimate_traffic(bundle, volume, journey, 'Ballari Road', '2023-04-12 12:00',
                             weather_frame=frame, now='2023-04-10')
    overcast = estimate_traffic(bundle, volume, journey, 'Ballari Road', '2023-04-12 12:00',
                                weather_frame=frame.assign(condition='Overcast'), now='2023-04-10')
    assert [r['travel_pace'] for r in clear['hourly_profile']] == [r['travel_pace'] for r in overcast['hourly_profile']]
    assert clear['weather']['condition'] == 'Clear'
    assert overcast['weather']['condition'] == 'Overcast'


def test_prediction_storage_roundtrip_and_reconciliation_do_not_score_lookups(tmp_path, estimator_data):
    bundle, volume, journey = estimator_data
    result = estimate_traffic(bundle, volume, journey, 'Ballari Road', '2023-04-05 12:30', now='2023-04-11')
    database = tmp_path / 'predictions.sqlite3'
    record_id = save_estimate(result, database)
    assert get_estimate(record_id, database) == result
    assert len(list_estimates(database)) == 1
    compared = compare_with_observations(list_estimates(database), volume)
    assert compared.daily_absolute_error.isna().all()
    assert compared.evaluation_status.eq('Historical estimate / lookup').all()
    with pytest.raises(ValueError, match='not found'):
        get_estimate('missing', database)


def test_daily_forecast_error_waits_for_a_completed_real_observation():
    records = pd.DataFrame({'id': ['a'], 'created_at_utc': ['2026-10-30T00:00:00+00:00'],
                             'road': ['Ballari Road'], 'target_local': ['2026-11-01 12:00:00'],
                             'daily_estimate': [120.], 'observed_daily': [None]})
    measured = pd.DataFrame({'entity': ['Ballari Road'], 'timestamp': pd.to_datetime(['2026-11-01']), 'y': [100.]})
    waiting = compare_with_observations(records, measured, as_of='2026-11-01 13:00')
    assert waiting.daily_absolute_error.isna().all()
    assert waiting.evaluation_status.iloc[0] == 'Awaiting measured daily count'
    scored = compare_with_observations(records, measured, as_of='2026-11-02 00:00')
    assert scored.daily_absolute_error.iloc[0] == 20.
    assert scored.evaluation_status.iloc[0] == 'Measured daily count available'


def test_weather_outside_horizon_makes_no_network_request(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Do not request unavailable weather.')
    monkeypatch.setattr(weather_module, 'get_bytes', forbidden)
    frame, metadata = weather_module.fetch_weather_for_date('2026-12-01', now='2026-10-31')
    assert frame.empty
    assert metadata['kind'] == 'outside_forecast_horizon'


def test_historical_weather_is_labelled_reanalysis(monkeypatch):
    payload = {'hourly': {'time': ['2024-01-01T12:00'], 'temperature_2m': [25.], 'relative_humidity_2m': [80.],
                          'apparent_temperature': [27.], 'precipitation': [1.], 'weather_code': [63], 'wind_speed_10m': [10.]}}
    urls = []
    def response(url, **kwargs):
        urls.append(url)
        return json.dumps(payload).encode()
    monkeypatch.setattr(weather_module, 'get_bytes', response)
    frame, metadata = weather_module.fetch_weather_for_date('2024-01-01', now='2026-10-31')
    assert urls[0].startswith('https://archive-api.open-meteo.com/v1/archive?')
    assert metadata['kind'] == 'historical_reanalysis'
    assert 'historical_reanalysis' in frame.source.iloc[0]
    assert frame.condition.iloc[0] == 'Rain'
