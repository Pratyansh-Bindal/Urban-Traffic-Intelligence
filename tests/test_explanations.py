import pytest

from traffic_intelligence.explanations import prediction_summary


@pytest.fixture
def result():
    return {
        'period_start': '2026-11-01 12:00:00', 'period_end': '2026-11-01 13:00:00',
        'day_part': 'Lunch', 'hourly_estimate': 1303., 'observed_daily': None,
        'weather': {'condition': 'Rain', 'temperature': 25.},
        'weather_metadata': {'kind': 'forecast'},
        'calendar': {'holiday_name': 'Kannada Rajyotsava', 'festival_window': True, 'is_custom_event': False},
        'corridor': {'is_airport_corridor': True, 'reference_route': 'Airport Expy'},
        'anomaly': {'score': 1., 'flagged': False, 'volume_z': 1., 'travel_pace_z': .5},
    }


def test_summary_covers_available_context_without_inventing_a_traffic_increase(result):
    text = prediction_summary(result)
    assert '1,303 vehicles' in text and 'Sunday lunch' in text
    assert '12:00–13:00 IST' in text
    assert 'forecast rain (25 °C)' in text and 'Kannada Rajyotsava' in text
    assert 'airport-corridor reference' in text
    assert 'No unusual deviation is flagged' in text
    assert 'unusually high' not in text
    assert len(text.split()) <= 80


def test_regular_day_does_not_invent_festivals_or_an_airport_link(result):
    result['calendar'] = {'holiday_name': '', 'festival_window': False, 'is_custom_event': False}
    result['corridor'] = {'is_airport_corridor': False, 'reference_route': None}
    text = prediction_summary(result)
    assert 'regular calendar day' in text and 'citywide travel patterns' in text
    assert 'festival' not in text and 'airport' not in text


def test_unavailable_weather_and_reference_are_disclosed(result):
    result['weather'] = {'condition': 'Unknown', 'temperature': None}
    result['weather_metadata'] = {'kind': 'outside_forecast_horizon'}
    result['anomaly'] = {'score': None, 'flagged': False}
    text = prediction_summary(result)
    assert 'weather unavailable' in text and 'seasonal' in text
    assert 'forecast rain' not in text
    assert 'insufficient for an anomaly check' in text
    assert 'No unusual deviation' not in text


def test_historical_count_is_not_described_as_a_modelled_daily_forecast(result):
    result['observed_daily'] = 30000.
    result['weather_metadata'] = {'kind': 'historical_reanalysis'}
    text = prediction_summary(result)
    assert 'daily count is recorded' in text and 'vehicles are estimated' in text
    assert 'historical rain' in text and 'forecast' not in text


@pytest.mark.parametrize('volume_z,pace_z,subject,direction', [
    (4., -1., 'Estimated volume', 'high'),
    (-4., 1., 'Estimated volume', 'low'),
    (1., 4., 'Travel delay', 'high'),
    (1., -4., 'Travel delay', 'low'),
])
def test_flag_explains_the_strongest_anomaly_without_claiming_an_incident(result, volume_z, pace_z, subject, direction):
    result['anomaly'] = {'score': max((volume_z, pace_z), key=abs), 'flagged': True,
                         'volume_z': volume_z, 'travel_pace_z': pace_z}
    text = prediction_summary(result)
    assert f'{subject} is unusually {direction}' in text
    assert 'incident' not in text and 'caused' not in text
