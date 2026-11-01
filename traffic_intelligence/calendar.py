"""Known calendar context, festival windows, editable events and ICS export."""
from functools import lru_cache
import re

import holidays
import numpy as np
import pandas as pd

from .settings import EVENTS, TIMEZONE

FESTIVAL_PATTERN = re.compile(r'ugadi|yugadi|deepavali|diwali|dussehra|dasara|vijaya|ganesh|vinayaka|sankranti|pongal|holi|christmas|eid|id-ul|id-e|bakrid|maha.*navami|rama? navami|krishna|janmashtami|good friday|muharram|maha.*shivaratri|rajyotsava', re.I)
STATE_OBSERVANCE_SOURCE = 'https://dtek.karnataka.gov.in/storage/pdf-files/C-20%20syllabus/Balake%20Kannada-2.pdf'
FESTIVAL_FEATURES = ['is_holiday', 'is_festival', 'festival_window', 'days_to_festival', 'days_since_festival',
                     'festival_name', 'is_custom_event']


@lru_cache(maxsize=32)
def holiday_table(first_year: int, last_year: int) -> pd.DataFrame:
    dates = pd.date_range(f'{first_year}-01-01', f'{last_year}-12-31')
    calendar = holidays.country_holidays('IN', subdiv='KA', years=range(first_year - 1, last_year + 2), language='en_US')
    public_dates = set(calendar)
    # State celebrations matter to traffic even when absent from a holiday library
    # or when the government holiday list omits dates already falling on Sunday.
    # Source: Karnataka DTE curriculum, annual Kannada Rajyotsava on 1 November.
    observance_dates = set()
    for year in range(first_year - 1, last_year + 2):
        observance = pd.Timestamp(year=year, month=11, day=1).date()
        calendar[observance] = 'Kannada Rajyotsava (state observance)'
        observance_dates.add(observance)
    names = [calendar.get(d.date(), '') for d in dates]
    result = pd.DataFrame({'date': dates, 'holiday_name': names})
    result['is_holiday'] = [int(d.date() in public_dates) for d in dates]
    result['is_observance'] = [int(d.date() in observance_dates) for d in dates]
    result['is_festival'] = result.holiday_name.str.contains(FESTIVAL_PATTERN).astype(int)
    festival_dates = pd.DatetimeIndex([pd.Timestamp(d) for d, n in calendar.items() if FESTIVAL_PATTERN.search(n)]).sort_values()
    day_values = dates.to_numpy(dtype='datetime64[D]').astype(int)
    fest_values = festival_dates.to_numpy(dtype='datetime64[D]').astype(int)
    positions = np.searchsorted(fest_values, day_values)
    before = np.clip(positions - 1, 0, len(fest_values) - 1)
    after = np.clip(positions, 0, len(fest_values) - 1)
    result['days_to_festival'] = np.clip(fest_values[after] - day_values, 0, 60)
    # On a festival itself both distances are zero.
    result['days_since_festival'] = np.where(np.isin(day_values, fest_values), 0, np.clip(day_values - fest_values[before], 0, 60))
    result['festival_window'] = ((result.days_to_festival <= 3) | (result.days_since_festival <= 3)).astype(int)
    result['festival_name'] = result.holiday_name.where(result.is_festival.eq(1), 'None')
    result['source'] = f'python-holidays {holidays.__version__}; India / Karnataka; Karnataka DTE annual state observance'
    return result


def custom_events() -> pd.DataFrame:
    if not EVENTS.exists():
        return pd.DataFrame(columns=['date', 'name', 'entity', 'source', 'known_from'])
    events = pd.read_csv(EVENTS).fillna('')
    if events.empty:
        return events
    required = {'date', 'name', 'entity', 'source', 'known_from'}
    if not required.issubset(events):
        raise ValueError('Events need date,name,entity,source,known_from. Use * for all routes.')
    events['date'] = pd.to_datetime(events.date, errors='raise').dt.normalize()
    events['known_from'] = pd.to_datetime(events.known_from, errors='raise').dt.normalize()
    if events.name.eq('').any() or events.source.eq('').any():
        raise ValueError('Every custom event needs a name and source.')
    if (events.known_from > events.date).any():
        raise ValueError('An event known only after its date cannot be a forecast feature.')
    return events


def add_calendar_features(frame: pd.DataFrame, *, as_of=None) -> pd.DataFrame:
    out = frame.copy()
    timestamp = pd.to_datetime(out.timestamp)
    date = timestamp.dt.normalize()
    table = holiday_table(int(timestamp.dt.year.min()), int(timestamp.dt.year.max())).set_index('date')
    for col in ['holiday_name', *FESTIVAL_FEATURES[:-1]]:
        out[col] = date.map(table[col]).to_numpy()
    out['is_custom_event'] = 0
    cutoff = pd.Timestamp(as_of) if as_of is not None else None
    if cutoff is not None and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(TIMEZONE).tz_localize(None)
    events = custom_events()
    for event in events.itertuples(index=False):
        if cutoff is not None and event.known_from > cutoff:
            continue
        mask = date.eq(event.date) & timestamp.ge(event.known_from)
        if event.entity not in ('', '*'):
            mask &= out.entity.eq(event.entity)
        out.loc[mask, 'is_custom_event'] = 1
        out.loc[mask, 'festival_window'] = 1
    out['weekday'] = timestamp.dt.day_of_week
    out['is_weekend'] = (out.weekday >= 5).astype(int)
    out['month'] = timestamp.dt.month
    out['hour'] = timestamp.dt.hour
    for name, values, period in [('hour', out.hour, 24), ('weekday', out.weekday, 7), ('year', timestamp.dt.day_of_year, 365.25)]:
        out[f'{name}_sin'] = np.sin(2 * np.pi * values / period)
        out[f'{name}_cos'] = np.cos(2 * np.pi * values / period)
    return out


def calendar_ics(year: int) -> str:
    table = holiday_table(year, year)
    lines = ['BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//Urban Traffic Intelligence//Karnataka calendar//EN', 'CALSCALE:GREGORIAN']
    for row in table.loc[table.is_holiday.eq(1) | table.is_observance.eq(1)].itertuples():
        start = row.date.strftime('%Y%m%d')
        end = (row.date + pd.Timedelta(days=1)).strftime('%Y%m%d')
        name = row.holiday_name.replace('\\', '\\\\').replace(';', '\\;').replace(',', '\\,').replace('\n', '\\n')
        lines += ['BEGIN:VEVENT', f'UID:{start}-ka@urban-traffic-intelligence.local', 'DTSTAMP:20261031T000000Z',
                  f'DTSTART;VALUE=DATE:{start}', f'DTEND;VALUE=DATE:{end}', f'SUMMARY:{name}',
                  'DESCRIPTION:Karnataka holiday calendar. Local observances and estimated lunar dates should be confirmed.', 'END:VEVENT']
    return '\r\n'.join(lines + ['END:VCALENDAR', ''])
