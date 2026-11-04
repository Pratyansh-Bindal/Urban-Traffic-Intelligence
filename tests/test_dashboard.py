from datetime import date, time
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import traffic_intelligence.storage as storage
import traffic_intelligence.weather as weather

ROOT = Path(__file__).resolve().parent.parent
PREPARED = all((ROOT / 'artifacts' / dataset / 'model.joblib').exists() for dataset in ['volume', 'journey', 'estimator'])
pytestmark = pytest.mark.skipif(not PREPARED, reason='Train local artifacts to run dashboard integration checks.')


@pytest.fixture(autouse=True)
def local_mode(monkeypatch):
    monkeypatch.delenv('TRAFFIC_PUBLIC_MODE', raising=False)
    monkeypatch.delenv('TRAFFIC_PUBLIC_URL', raising=False)


def element(elements, label):
    return next(item for item in elements if item.label == label)


def test_road_time_request_automatically_fetches_context_and_saves_once(tmp_path, monkeypatch):
    st.cache_data.clear()
    database = tmp_path / 'predictions.sqlite3'
    monkeypatch.setattr(storage, 'PREDICTIONS_DB', database)
    requested = []
    def provider(selected_date, **kwargs):
        requested.append(selected_date)
        frame = pd.DataFrame({'timestamp': pd.date_range(selected_date, periods=24, freq='h'),
                               'condition': ['Rain'] * 24, 'temperature': [25.] * 24,
                               'humidity': [90.] * 24, 'precipitation': [2.] * 24})
        return frame, {'provider': 'Open-Meteo', 'kind': 'forecast', 'status': 'available'}
    monkeypatch.setattr(weather, 'fetch_weather_for_date', provider)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
    assert not app.exception
    assert [item.label for item in app.selectbox] == ['Road name']
    assert not app.slider and not app.checkbox
    element(app.selectbox, 'Road name').set_value('Ballari Road')
    element(app.date_input, 'Date').set_value(date(2026, 11, 1))
    element(app.time_input, 'Time · IST').set_value(time(12, 30))
    element(app.button, 'Estimate traffic').click().run()
    assert not app.exception, [error.message for error in app.exception]
    assert requested == ['2026-11-01']
    assert app.session_state['traffic_result']['calendar']['is_festival']
    assert 'Rajyotsava' in app.session_state['traffic_result']['calendar']['holiday_name']
    assert app.session_state['traffic_result']['weather']['condition'] == 'Rain'
    assert element(app.metric, 'Weather condition').value == 'Rainy'
    assert any('Weather used for daily volume:' in item.value and 'Rainy' in item.value for item in app.markdown)
    assert any('most common for the day' in item.value for item in app.markdown)
    assert any('dry/rainy conditions, temperature and humidity' in item.value for item in app.caption)
    assert any('Precipitation: 2.0 mm' in item.value for item in app.caption)
    summary = app.session_state['traffic_result']['summary']
    assert 'forecast rain' in summary and 'Rajyotsava' in summary
    assert any(item.value == summary for item in app.markdown)
    assert any(item.label == 'Estimated vehicles in selected hour' for item in app.metric)
    classification = app.session_state['traffic_result']['traffic_classification']
    assert element(app.metric, 'Traffic condition').value == classification['level']
    assert any('**Low:** below' in item.value and '**High:** above' in item.value for item in app.markdown)
    assert any(f'{classification["reference_count"]} earlier daily counts' in item.value for item in app.caption)
    assert len(storage.list_estimates(database)) == 1
    app.run()
    assert len(storage.list_estimates(database)) == 1
    element(app.radio, 'Workspace').set_value('Saved estimates').run()
    assert not app.exception
    assert any(item.label == 'Saved estimates' and item.value == '1' for item in app.metric)
    record_id = app.session_state['traffic_record_id']
    saved = storage.get_estimate(record_id, database)
    assert saved['summary'] == summary
    # Records saved before summaries existed must still open with an explanation.
    legacy = {key: value for key, value in saved.items() if key not in ('summary', 'traffic_classification')}
    monkeypatch.setattr(storage, 'get_estimate', lambda value: legacy)
    element(app.button, 'Open result').click().run()
    assert not app.exception
    assert 'summary' not in app.session_state['opened_result'][1]
    assert any('forecast rain' in item.value and 'Rajyotsava' in item.value for item in app.markdown)
    assert any('earlier travel-time label' in item.value for item in app.caption)


def test_weather_failure_still_returns_a_labelled_estimate(tmp_path, monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(storage, 'PREDICTIONS_DB', tmp_path / 'predictions.sqlite3')
    def disconnected(*args, **kwargs):
        raise OSError('offline')
    monkeypatch.setattr(weather, 'fetch_weather_for_date', disconnected)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
    element(app.date_input, 'Date').set_value(date(2026, 11, 2))
    element(app.button, 'Estimate traffic').click().run()
    assert not app.exception, [error.message for error in app.exception]
    assert any('Weather unavailable' in item.value for item in app.markdown)
    assert app.session_state['traffic_result']['weather_metadata']['status'] == 'unavailable'
    assert element(app.metric, 'Weather condition').value == 'Unavailable'
    assert any('Weather used for daily volume:' in item.value and 'Seasonal weather patterns' in item.value for item in app.markdown)
    assert 'weather unavailable' in app.session_state['traffic_result']['summary']
    assert app.session_state['traffic_record_id']


def test_consecutive_searches_use_the_new_road_date_and_time(tmp_path, monkeypatch):
    st.cache_data.clear()
    database = tmp_path / 'predictions.sqlite3'
    monkeypatch.setattr(storage, 'PREDICTIONS_DB', database)
    monkeypatch.setattr(weather, 'fetch_weather_for_date', lambda *args, **kwargs:
                        (pd.DataFrame(), {'status': 'unavailable', 'kind': 'connection_unavailable'}))
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
    for road, day, hour in [('Ballari Road', date(2026, 11, 1), time(12, 30)),
                            ('Sarjapur Road', date(2026, 11, 2), time(18, 30))]:
        element(app.selectbox, 'Road name').set_value(road)
        element(app.date_input, 'Date').set_value(day)
        element(app.time_input, 'Time · IST').set_value(hour)
        element(app.button, 'Estimate traffic').click().run()
        assert not app.exception
        result = app.session_state['traffic_result']
        assert result['road'] == road
        assert result['requested_at'] == f'{day} {hour}'
        assert any(item.value == result['summary'] for item in app.markdown)
    records = storage.list_estimates(database)
    assert len(records) == 2 and records.road.nunique() == 2


def test_public_mode_keeps_history_private_between_browser_sessions(tmp_path, monkeypatch):
    st.cache_data.clear()
    monkeypatch.setenv('TRAFFIC_PUBLIC_MODE', '1')
    public_url = 'https://traffic-demo.trycloudflare.com'
    monkeypatch.setenv('TRAFFIC_PUBLIC_URL', public_url)
    monkeypatch.setattr(storage, 'PREDICTIONS_DB', tmp_path / 'predictions.sqlite3')
    monkeypatch.setattr(weather, 'fetch_weather_for_date', lambda *args, **kwargs:
                        (pd.DataFrame(), {'status': 'unavailable', 'kind': 'connection_unavailable'}))
    first = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
    assert any(item.value == public_url for item in first.code)
    assert 'Research explorer' not in element(first.radio, 'Workspace').options
    element(first.button, 'Estimate traffic').click().run()
    assert not first.exception
    first_id = first.session_state['traffic_record_id']
    element(first.radio, 'Workspace').set_value('Saved estimates').run()
    assert any(item.label == 'Saved estimates' and item.value == '1' for item in first.metric)
    element(first.button, 'Open result').click().run()
    assert not first.exception
    assert first.session_state['opened_result'][0] == first_id

    second = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
    element(second.radio, 'Workspace').set_value('Saved estimates').run()
    assert not second.exception
    assert any('Your first traffic estimate' in item.value for item in second.info)
    assert not second.dataframe and not second.selectbox
    assert len(storage.list_estimates()) == 1


def test_research_branches_retain_all_analysis_views():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
    element(app.radio, 'Workspace').set_value('Research explorer').run()
    for branch in ['Journey time', 'Daily volume']:
        element(app.radio, 'Analysis branch').set_value(branch).run()
        for view in ['Main estimator evaluation', 'Overview', 'Festival calendar', 'Weather & research', 'Anomaly explorer', 'Model evaluation', 'Data audit']:
            element(app.selectbox, 'Research view').set_value(view).run()
            assert not app.exception, [error.message for error in app.exception]
