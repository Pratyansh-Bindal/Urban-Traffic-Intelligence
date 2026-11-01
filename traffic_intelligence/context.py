"""Road metadata and interpretable time features, without fixed traffic multipliers."""
import pandas as pd

from .calendar import add_calendar_features


# References describe a corridor, not a sensor measuring this road's vehicle flow.
ROAD_CONTEXT = {
    'Ballari Road': ('airport_connector', 'Airport Expy', 'Airport-corridor proxy'),
    'Hebbal Flyover': ('airport_connector', 'Airport Expy', 'Airport-corridor proxy'),
    'Hosur Road': ('employment_corridor', 'Hosur Road', 'Same named corridor'),
    'Sarjapur Road': ('employment_corridor', 'Sarjapur Road', 'Same named corridor'),
    'Silk Board Junction': ('employment_corridor', 'Hosur Road', 'Nearby corridor proxy'),
    'Sony World Junction': ('commercial', 'Sarjapur Road', 'Nearby corridor proxy'),
    'Marathahalli Bridge': ('employment_corridor', 'East Outer Ring', 'Nearby corridor proxy'),
    'ITPL Main Road': ('employment_corridor', None, 'Citywide temporal reference'),
    '100 Feet Road': ('commercial', None, 'Citywide temporal reference'),
    'CMH Road': ('commercial', None, 'Citywide temporal reference'),
    'Anil Kumble Circle': ('city_centre', None, 'Citywide temporal reference'),
    'Trinity Circle': ('city_centre', None, 'Citywide temporal reference'),
    'Jayanagar 4th Block': ('mixed_residential', None, 'Citywide temporal reference'),
    'South End Circle': ('mixed_residential', None, 'Citywide temporal reference'),
    'Tumkur Road': ('arterial', None, 'Citywide temporal reference'),
    'Yeshwanthpur Circle': ('arterial', None, 'Citywide temporal reference'),
}

TIME_FEATURES = ['day_part', 'is_overnight', 'is_morning_peak', 'is_lunch_period',
                 'is_evening_peak', 'is_airport_corridor', 'corridor_type']
CONDITION_FEATURES = ['weather_context', 'temperature_context', 'humidity_context']


def road_context(road):
    kind, route, relation = ROAD_CONTEXT.get(road, ('other', None, 'Citywide temporal reference'))
    return {'corridor_type': kind, 'is_airport_corridor': int(kind == 'airport_connector'),
            'reference_route': route, 'reference_relation': relation}


def add_context(frame, *, as_of=None):
    out = add_calendar_features(frame, as_of=as_of)
    hour = out.timestamp.dt.hour
    out['day_part'] = pd.cut(hour, bins=[-1, 5, 9, 11, 14, 16, 19, 23],
                             labels=['Overnight', 'Morning commute', 'Late morning', 'Lunch',
                                     'Afternoon', 'Evening commute', 'Late evening']).astype(str)
    out['is_overnight'] = hour.lt(6).astype(int)
    out['is_morning_peak'] = hour.between(6, 9).astype(int)
    out['is_lunch_period'] = hour.between(12, 14).astype(int)
    out['is_evening_peak'] = hour.between(17, 19).astype(int)
    labels = out['label'] if 'label' in out else out.entity
    out['is_airport_corridor'] = [int(label == 'Airport Expy' or road_context(entity)['is_airport_corridor'])
                                  for entity, label in zip(out.entity, labels)]
    out['corridor_type'] = [road_context(entity)['corridor_type'] if entity in ROAD_CONTEXT else
                             ('airport_connector' if label == 'Airport Expy' else 'monitored_route')
                             for entity, label in zip(out.entity, labels)]
    return out


def condition_frame(frame, *, use_history=False, as_of=None):
    """Realised weather is training context, never claimed to be an archived forecast."""
    from .features import make_features
    out = make_features(frame, 'volume', as_of=as_of) if use_history else frame.copy()
    out = add_context(out, as_of=as_of)
    out['weather_context'] = out.weather.fillna('Unknown')
    for column in ('temperature', 'humidity'):
        out[column + '_context'] = out[column] if column in out else float('nan')
    return out
