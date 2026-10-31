from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import traffic_intelligence.calendar as calendar_module
from traffic_intelligence.calendar import add_calendar_features, calendar_ics, holiday_table
from traffic_intelligence.data import load_journey, load_volume, rain_category
from traffic_intelligence.weather import condition_from_wmo, parse_outlook


def test_partial_hour_is_not_available_and_hourly_readings_are_aggregated(tmp_path):
    file = tmp_path/'traffic.csv'
    pd.DataFrame({'date':['2026-01-01']*3,'time':['14:15','14:45','15:15'],'route_code':['A']*3,
                  'duration':[20,40,99],'distance':[10]*3,'temp':[25]*3,'humidity':[80]*3,
                  'rsi_flag':['']*3}).to_csv(file,index=False)
    result,audit = load_journey(file,as_of='2026-01-01 15:30')
    assert len(result)==1
    assert result.iloc[0].timestamp==pd.Timestamp('2026-01-01 14:00')
    assert result.iloc[0].y==30
    assert result.iloc[0].weather=='Dry'
    assert audit['raw_rows']==3


def test_empty_precipitation_is_not_dry_when_weather_is_missing():
    assert rain_category('',False)=='Unknown'
    assert rain_category('',True)=='Dry'
    assert rain_category('Light Rain',True)=='Rain'


def test_duplicate_road_date_requires_resolution(tmp_path):
    file=tmp_path/'volume.csv'
    pd.DataFrame({'Date':['2024-01-01']*2,'Road/Intersection Name':['A']*2,'Area Name':['Area']*2,
                  'Traffic Volume':[10,20],'Weather Conditions':['Clear']*2,'Average Speed':[30]*2}).to_csv(file,index=False)
    with pytest.raises(ValueError,match='duplicate road-date'):
        load_volume(file)


def test_calendar_distinguishes_public_holiday_and_festival():
    table=holiday_table(2024,2024).set_index('date')
    assert table.loc['2024-01-26','is_holiday']==1
    assert table.loc['2024-01-26','is_festival']==0
    assert table.loc['2024-10-31','is_festival']==1
    assert table.loc['2024-10-30','festival_window']==1
    assert table.loc['2024-10-31','days_to_festival']==0


def test_custom_event_not_used_before_announcement(tmp_path,monkeypatch):
    file=tmp_path/'events.csv'
    file.write_text('date,name,entity,source,known_from\n2024-06-15,Local event,A,https://example.org/event,2024-06-14\n')
    monkeypatch.setattr(calendar_module,'EVENTS',file)
    frame=pd.DataFrame({'timestamp':pd.to_datetime(['2024-06-15','2024-06-15']),'entity':['A','B']})
    output=add_calendar_features(frame)
    assert output.is_custom_event.tolist()==[1,0]
    file.write_text('date,name,entity,source,known_from\n2024-06-15,Local event,A,https://example.org/event,2024-06-16\n')
    with pytest.raises(ValueError,match='after its date'):
        calendar_module.custom_events()


def test_calendar_export_uses_exclusive_next_day_end():
    ics=calendar_ics(2024)
    assert 'BEGIN:VCALENDAR\r\nVERSION:2.0' in ics
    assert 'DTSTART;VALUE=DATE:20240126\r\nDTEND;VALUE=DATE:20240127' in ics
    assert ics.count('BEGIN:VEVENT')==ics.count('END:VEVENT')


def test_state_observance_remains_visible_when_it_falls_on_sunday():
    table = holiday_table(2026, 2026).set_index('date')
    assert table.loc['2026-11-01', 'is_observance'] == 1
    assert table.loc['2026-11-01', 'is_festival'] == 1
    assert 'Rajyotsava' in table.loc['2026-11-01', 'holiday_name']
    assert table.loc['2026-10-31', 'festival_window'] == 1
    assert 'Kannada Rajyotsava' in calendar_ics(2026)
    assert table.loc['2026-03-26', 'is_festival'] == 1


def test_forecast_weather_codes_and_local_timestamps():
    assert condition_from_wmo(63)=='Rain'
    assert condition_from_wmo(45)=='Fog'
    assert condition_from_wmo(0)=='Clear'
    payload={'hourly':{'time':['2026-10-31T15:00'],'temperature_2m':[26.1],'relative_humidity_2m':[77],
                       'apparent_temperature':[28.0],'precipitation':[2.5],'weather_code':[63],'wind_speed_10m':[10]}}
    frame=parse_outlook(payload)
    assert frame.iloc[0].timestamp==pd.Timestamp('2026-10-31 15:00')
    assert frame.iloc[0].condition=='Rain'
    with pytest.raises(ValueError,match='no hourly'):
        parse_outlook({})
