"""Automatic public weather context, with forecast/retrospective provenance."""
from datetime import datetime, timezone
import json
from urllib.parse import urlencode

import pandas as pd

from .download import get_bytes


def condition_from_wmo(code, wind_speed=0):
    if pd.isna(code):
        return 'Unknown'
    code = int(code)
    if 51 <= code <= 99:
        return 'Rain'
    if code in (45, 48):
        return 'Fog'
    if wind_speed is not None and wind_speed >= 40:
        return 'Windy'
    return 'Clear' if code == 0 else 'Overcast'


def parse_outlook(payload):
    hourly = payload.get('hourly', {})
    if not hourly.get('time'):
        raise ValueError('The weather provider returned no hourly forecast.')
    frame = pd.DataFrame(hourly).rename(columns={'time': 'timestamp', 'temperature_2m': 'temperature',
                                                'relative_humidity_2m': 'humidity', 'apparent_temperature': 'feels_like'})
    frame['timestamp'] = pd.to_datetime(frame.timestamp)
    frame['condition'] = [condition_from_wmo(c, w) for c,w in zip(frame.weather_code, frame.wind_speed_10m)]
    frame['source'] = 'Open-Meteo model forecast for Bengaluru city centre; not a route station observation'
    return frame


def fetch_outlook():
    parameters = {'latitude': 12.9716, 'longitude': 77.5946, 'timezone': 'Asia/Kolkata', 'forecast_days': 7,
                  'hourly': 'temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m'}
    url = 'https://api.open-meteo.com/v1/forecast?' + urlencode(parameters)
    payload = json.loads(get_bytes(url, timeout=20))
    frame = parse_outlook(payload)
    return frame, {'provider': 'Open-Meteo', 'url': url, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                   'location': 'Bengaluru city centre', 'timezone': 'Asia/Kolkata'}


def fetch_weather_for_date(selected_date, *, now=None):
    target = pd.Timestamp(selected_date).normalize()
    today = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='Asia/Kolkata')
    if today.tzinfo is not None:
        today = today.tz_convert('Asia/Kolkata').tz_localize(None)
    days = (target - today.normalize()).days
    metadata = {'provider': 'Open-Meteo', 'location': 'Bengaluru city centre', 'timezone': 'Asia/Kolkata',
                'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}
    if days > 15:
        return pd.DataFrame(), {**metadata, 'status': 'unavailable', 'kind': 'outside_forecast_horizon',
                                'message': 'Weather is outside the provider forecast horizon. Calendar and learned seasonal patterns are still used.'}
    parameters = {'latitude': 12.9716, 'longitude': 77.5946, 'timezone': 'Asia/Kolkata',
                  'hourly': 'temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m'}
    if days >= -5:
        endpoint = 'https://api.open-meteo.com/v1/forecast'
        parameters.update(forecast_days=16 if days >= 0 else 1, past_days=max(0, -days))
        kind = 'forecast' if days >= 0 else 'recent_model_history'
    else:
        endpoint = 'https://archive-api.open-meteo.com/v1/archive'
        parameters.update(start_date=target.strftime('%Y-%m-%d'), end_date=target.strftime('%Y-%m-%d'))
        kind = 'historical_reanalysis'
    url = endpoint + '?' + urlencode(parameters)
    frame = parse_outlook(json.loads(get_bytes(url, timeout=20)))
    frame = frame.loc[frame.timestamp.dt.normalize().eq(target)].copy()
    if frame.empty:
        raise ValueError('The provider returned no weather for the selected date.')
    frame['source'] = f'Open-Meteo {kind}; Bengaluru city-centre model grid'
    return frame, {**metadata, 'url': url, 'status': 'available', 'kind': kind,
                   'message': 'City-centre weather context; local road conditions can differ.'}
