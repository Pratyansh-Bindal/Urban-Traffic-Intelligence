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
    assert any(item.label == 'Estimated vehicles in selected hour' for item in app.metric)
    assert len(storage.list_estimates(database)) == 1
    app.run()
    assert len(storage.list_estimates(database)) == 1
    element(app.radio, 'Workspace').set_value('Saved estimates').run()
    assert not app.exception
    assert any(item.label == 'Saved estimates' and item.value == '1' for item in app.metric)


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
    assert app.session_state['traffic_record_id']


def test_research_branches_retain_all_analysis_views():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
    element(app.radio, 'Workspace').set_value('Research explorer').run()
    for branch in ['Journey time', 'Daily volume']:
        element(app.radio, 'Analysis branch').set_value(branch).run()
        for view in ['Main estimator evaluation', 'Overview', 'Festival calendar', 'Weather & research', 'Anomaly explorer', 'Model evaluation', 'Data audit']:
            element(app.selectbox, 'Research view').set_value(view).run()
            assert not app.exception, [error.message for error in app.exception]
