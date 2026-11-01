"""Short descriptions of the available prediction context, without causal claims."""
from datetime import datetime


def prediction_summary(result):
    """Explain an estimate using its stored inputs and anomaly reference only."""
    start = datetime.fromisoformat(result['period_start'])
    end = datetime.fromisoformat(result['period_end'])
    period = f'{start:%A} {result["day_part"].lower()} ({start:%H:%M}–{end:%H:%M} IST)'
    count = f'{result["hourly_estimate"]:,.0f}'
    if result['observed_daily'] is not None:
        headline = f'The daily count is recorded; about {count} vehicles are estimated for {period}.'
    else:
        headline = f'About {count} vehicles are estimated for {period}.'

    weather = result['weather']
    if weather['condition'] == 'Unknown':
        weather_text = 'seasonal weather patterns (weather unavailable)'
    else:
        kind = result['weather_metadata'].get('kind')
        prefix = 'forecast ' if kind == 'forecast' else 'historical ' if kind in ('historical_reanalysis', 'recent_model_history') else ''
        condition = weather['condition'].lower()
        weather_text = prefix + (condition + ' weather' if condition in ('clear', 'overcast', 'windy') else condition)
        if weather.get('temperature') is not None:
            weather_text += f' ({weather["temperature"]:.0f} °C)'

    calendar = result['calendar']
    calendar_text = calendar.get('holiday_name') or ('a nearby festival date' if calendar.get('festival_window') else 'a regular calendar day')
    if calendar.get('is_custom_event'):
        calendar_text = (calendar['holiday_name'] + ' and a local event') if calendar.get('holiday_name') else 'a local calendar event'

    corridor = result['corridor']
    if corridor.get('is_airport_corridor'):
        patterns = 'airport-corridor reference patterns'
    elif corridor.get('reference_relation') == 'Nearby corridor proxy':
        patterns = 'nearby-corridor reference patterns'
    elif corridor.get('reference_route'):
        patterns = 'corridor travel patterns'
    else:
        patterns = 'citywide travel patterns'
    context = f'Historical traffic and {patterns} provide context, alongside {weather_text} and {calendar_text}.'

    anomaly = result['anomaly']
    if anomaly.get('score') is None:
        check = 'Earlier data is insufficient for an anomaly check.'
    elif anomaly['flagged']:
        volume_z = anomaly.get('volume_z')
        pace_z = anomaly.get('travel_pace_z')
        use_pace = pace_z is not None and (volume_z is None or abs(pace_z) > abs(volume_z))
        subject = 'Travel delay' if use_pace else 'Estimated volume'
        score = pace_z if use_pace else volume_z
        check = f'{subject} is unusually {"high" if score > 0 else "low"} compared with earlier records.'
    else:
        check = 'No unusual deviation is flagged.'
    return ' '.join([headline, context, check])
