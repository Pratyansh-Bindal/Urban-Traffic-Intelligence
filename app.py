"""Run locally with: python -m streamlit run app.py"""
from datetime import datetime
from html import escape
import json
import sqlite3

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from threadpoolctl import threadpool_limits

from traffic_intelligence.context import road_context
from traffic_intelligence.estimator import estimate_traffic
from traffic_intelligence.settings import ARTIFACTS, TIMEZONE
from traffic_intelligence.storage import compare_with_observations, get_estimate, list_estimates, save_estimate
from traffic_intelligence.weather import fetch_weather_for_date

st.set_page_config(page_title='Urban Traffic Intelligence', page_icon='↗', layout='wide', initial_sidebar_state='auto')
st.markdown('''<style>
.stApp {font-family:system-ui,sans-serif;}
.block-container {padding-top:3rem;max-width:1400px;padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.035em;}
[data-testid="stSidebar"] {border-right:1px solid #DFE7EC;background:#FFF;}
[data-testid="stMetric"] {background:#FFF;border:1px solid #E2E9ED;padding:1.1rem;border-radius:14px;}
[data-testid="stMetricLabel"] {color:#627785;font-size:.85rem;}
[data-testid="stMetricValue"] {font-weight:700;font-size:1.7rem;}
[data-testid="stMetricLabel"] p,[data-testid="stMetricValue"] > div {white-space:normal!important;overflow:visible!important;overflow-wrap:break-word;}
.eyebrow {font-size:.74rem;letter-spacing:.14em;color:#0D8075;font-weight:700;margin-bottom:.6rem;}
.hero {background:#102F3C;color:#FFF;border-radius:18px;padding:1.5rem 1.8rem;margin:1rem 0 1.5rem;}
.hero h2 {color:#FFF;margin:0;font-size:1.65rem;}
.hero p {color:#BCD2DA;margin:.5rem 0 0;}
.note {color:#647C8B;font-size:.9rem;line-height:1.6;}
@media (max-width:1100px) {
 [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {flex-wrap:wrap;}
 [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {min-width:165px;flex:1 1 40%;}
}
</style>''', unsafe_allow_html=True)

TEAL, AMBER, MUTED = '#0D8075', '#DB9B46', '#B7CACF'


@st.cache_resource(show_spinner=False)
def load_product(version):
    return joblib.load(ARTIFACTS / 'estimator' / 'model.joblib')


@st.cache_data(show_spinner=False)
def load_histories(volume_version, journey_version):
    return (pd.read_parquet(ARTIFACTS / 'volume' / 'history.parquet'),
            pd.read_parquet(ARTIFACTS / 'journey' / 'history.parquet'))


@st.cache_data(ttl=3600, show_spinner=False)
def automatic_weather(selected_date, today):
    return fetch_weather_for_date(selected_date, now=today)


def chart(figure, height=330):
    figure.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font=dict(family='system-ui, sans-serif', color='#365260'),
                         height=height, margin=dict(l=10, r=10, t=25, b=10),
                         legend=dict(orientation='h', y=1.15, x=0), hovermode='x unified')
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor='#E4EBEF', zeroline=False)
    st.plotly_chart(figure, width='stretch', config={'displayModeBar': False})


def show_result(result, record_id=None):
    target = pd.Timestamp(result['requested_at'])
    start, end = pd.Timestamp(result['period_start']), pd.Timestamp(result['period_end'])
    st.markdown(f'<div class="hero"><h2>{escape(result["road"])}</h2><p>{target:%A, %d %B %Y · %H:%M} IST · {escape(result["kind"])}</p></div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric('Estimated vehicles in selected hour', f'{result["hourly_estimate"]:,.0f}',
             help=f'Hour: {start:%H:%M}–{end:%H:%M}. An hourly estimate allocated from daily volume using learned travel patterns.')
    b.metric('Recorded daily vehicles' if result['observed_daily'] is not None else 'Estimated daily vehicles', f'{result["daily_estimate"]:,.0f}')
    c.metric('Traffic condition', result['traffic_level'], help='Based on the reference corridor’s travel pace relative to its historical distribution.')
    st.caption(f'Hourly estimate covers {start:%d %b, %H:%M}–{end:%H:%M} IST. Vehicle count means vehicles passing during this hour.')
    weather, calendar = result['weather'], result['calendar']
    left, right = st.columns(2)
    with left:
        st.markdown('**Weather · automatic**')
        if weather['condition'] == 'Unknown':
            st.write('Weather unavailable · seasonal context used')
        else:
            temperature = f' · {weather["temperature"]:.1f} °C' if weather['temperature'] is not None else ''
            st.write(weather['condition'] + temperature)
        st.caption(result['weather_metadata'].get('kind', 'unavailable').replace('_', ' ').capitalize() + ' · Bengaluru city centre')
    with right:
        st.markdown('**Calendar · automatic**')
        st.write(calendar['holiday_name'] or ('Near a festival' if calendar['festival_window'] else 'Regular day'))
        st.caption('Karnataka holiday calendar · ' + result['day_part'] + (' · Airport corridor' if result['corridor']['is_airport_corridor'] else ''))
    anomaly = result['anomaly']
    score = f' · robust Z-score {anomaly["score"]:+.2f}' if anomaly['score'] is not None else ''
    message = f'{anomaly["status"]}{score}'
    if anomaly['flagged']:
        st.warning(message + ' · ' + anomaly['direction'])
    else:
        st.info(message)
    if anomaly['observed_daily_z'] is not None:
        observed_alert = abs(anomaly['observed_daily_z']) > anomaly['threshold']
        st.caption(f'Recorded daily count: {"unusual" if observed_alert else "within usual range"} against earlier daily records (Z = {anomaly["observed_daily_z"]:+.2f}).')
    st.subheader('Traffic through the selected day')
    profile = pd.DataFrame(result['hourly_profile'])
    profile['timestamp'] = pd.to_datetime(profile.timestamp)
    profile['hour'] = profile.timestamp.dt.hour
    highlight = '#C65E4B' if anomaly['flagged'] else AMBER
    colors = [highlight if h == start.hour else TEAL for h in profile.hour]
    figure = go.Figure(go.Bar(x=profile.timestamp, y=profile.estimated_vehicles, name='Estimated vehicles', marker_color=colors,
                             customdata=profile[['travel_pace']], hovertemplate='%{x|%H:%M}<br>%{y:,.0f} estimated vehicles<br>Reference pace: %{customdata[0]:.2f} min/km<extra></extra>'))
    if 'usual_vehicles' in profile and profile.usual_vehicles.notna().any():
        figure.add_trace(go.Scatter(x=profile.timestamp, y=profile.usual_vehicles, name='Historical reference',
                                    mode='lines', line=dict(color='#476C9B', width=2, dash='dot')))
    figure.update_xaxes(dtick=7200000, tickformat='%H:%M', title='Hour beginning · IST')
    figure.update_yaxes(title='Estimated vehicles per hour')
    chart(figure)
    st.caption('The highlighted bar is the selected hour. The 24 hourly estimates sum to the daily volume above. The dotted line shows the historical reference when enough earlier records exist.')
    st.subheader('Patterns learned from travel data')
    period_means = [profile.loc[profile.hour.between(lo, hi), 'estimated_vehicles'].mean()
                    for lo, hi in [(0, 5), (6, 9), (12, 14), (17, 19)]]
    columns = st.columns(4)
    for column, name, value in zip(columns, ['Overnight · 00–06', 'Morning · 06–10', 'Lunch · 12–15', 'Evening · 17–20'], period_means):
        column.metric(name, f'{value:,.0f}', help='Average estimated vehicles per hour in this time window.')
    route_names = ', '.join(item['label'] for item in result['reference_routes'])
    reference_name = result['corridor']['reference_route'] or 'Citywide route pattern'
    st.caption(f'Travel reference: {reference_name} · {result["corridor"]["reference_relation"]}. Morning, lunch and evening effects are learned rather than assigned fixed increases.')
    with st.expander('Historical traffic and anomaly details'):
        historical = pd.DataFrame(result['history'])
        if not historical.empty:
            historical['date'] = pd.to_datetime(historical.date)
            figure = go.Figure(go.Scatter(x=historical.date, y=historical.vehicles, mode='lines+markers',
                                         name='Recorded daily volume', line=dict(color=TEAL), marker=dict(size=4)))
            if result['observed_daily'] is not None:
                figure.add_trace(go.Scatter(x=[target.normalize()], y=[result['observed_daily']], name='Selected recorded day',
                                            mode='markers', marker=dict(color=AMBER, size=10)))
            figure.update_yaxes(title='Recorded vehicles per day')
            chart(figure, 280)
        st.write(f'**Volume Z-score:** {anomaly["volume_z"]:+.2f}' if anomaly['volume_z'] is not None else 'No earlier volume reference is available.')
        if anomaly['travel_pace_z'] is not None:
            st.write(f'**Travel-pace Z-score:** {anomaly["travel_pace_z"]:+.2f}')
        st.caption(anomaly['note'])
    with st.expander('How the estimate was made'):
        st.write('A trained vehicle-volume model uses the road, weekday, season, past counts, weather category and holiday/festival features. A second trained model estimates travel pace using hour, morning/lunch/evening periods, route, airport-corridor identity, weather and calendar context.')
        st.write('Hourly volume is an approximation: the daily volume is allocated across the day in proportion to the predicted travel pace. Travel delay and vehicle flow can differ, so this allocation remains a modelling assumption.')
        st.write(f'**Models:** {result["model"]["daily_volume"]} for daily volume; {result["model"]["travel_pace"]} for travel pace.')
        st.write(f'**Held-out daily MAE:** {result["model"]["daily_test_mae"]:,.0f} vehicles/day. **Travel-pace MAE:** {result["model"]["travel_pace_test_mae"]:.3f} minutes/km.')
        st.caption(result['model']['evaluation_protocol'] + ' Hourly vehicle-count accuracy has not been measured against hourly counts.')
        responses = result['model_responses']
        st.write(f'Daily model input contrasts: weather {responses["weather_daily_vehicles"]:+,.0f} vehicles; calendar {responses["calendar_daily_vehicles"]:+,.0f} vehicles. These describe model responses, not causal effects.')
        st.caption(f'Travel source: {route_names}. Latest count input: {result["latest_volume_input"] or "none"}. Model version: {result["model"]["version"]}.')
    for warning in result['warnings']:
        st.caption(warning)
    if record_id:
        st.success('Estimate saved locally with its inputs, model version and anomaly scores.')
    st.download_button('Download this result', json.dumps(result, indent=2),
                       f'traffic_{target:%Y%m%d_%H%M}.json', 'application/json', key='download_result')


def predictor(bundle, volume, journey):
    st.markdown('<div class="eyebrow">BENGALURU / URBAN TRAFFIC INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title('Estimate traffic before you leave.')
    st.markdown('<p class="note">Choose the road and time. Weather, holidays, past traffic and travel patterns are connected automatically.</p>', unsafe_allow_html=True)
    previous = st.session_state.get('traffic_result')
    default = pd.Timestamp(previous['requested_at']) if previous else pd.Timestamp.now(tz=TIMEZONE).ceil('h')
    roads = sorted(volume.entity.unique())
    road_index = roads.index(previous['road']) if previous and previous['road'] in roads else 0
    with st.form('traffic_request'):
        a, b, c = st.columns([2, 1, 1])
        road = a.selectbox('Road name', roads, index=road_index,
                           format_func=lambda name: name + (' · airport corridor' if road_context(name)['is_airport_corridor'] else ''))
        date = b.date_input('Date', value=default.date(), min_value=volume.timestamp.min().date(),
                            max_value=(pd.Timestamp.now(tz=TIMEZONE) + pd.DateOffset(years=3)).date())
        time = c.time_input('Time · IST', value=default.time().replace(tzinfo=None), step=900)
        submitted = st.form_submit_button('Estimate traffic', type='primary', width='stretch')
    if submitted:
        target = pd.Timestamp(datetime.combine(date, time))
        with st.spinner('Connecting weather and calendar, estimating traffic and checking anomalies…'):
            try:
                weather_frame, metadata = automatic_weather(date.isoformat(), pd.Timestamp.now(tz=TIMEZONE).date().isoformat())
            except Exception:
                weather_frame = pd.DataFrame()
                metadata = {'provider': 'Open-Meteo', 'status': 'unavailable', 'kind': 'connection_unavailable',
                            'message': 'Weather service is unavailable. This estimate uses seasonal weather context and learned time patterns.'}
            try:
                with threadpool_limits(limits=2):
                    result = estimate_traffic(bundle, volume, journey, road, target, weather_frame=weather_frame, weather_metadata=metadata)
                st.session_state['traffic_result'] = result
                st.session_state['traffic_record_id'] = None
                record_id = save_estimate(result)
                st.session_state['traffic_record_id'] = record_id
            except (ValueError, OSError, sqlite3.Error) as error:
                st.error(f'The estimate could not be completed or saved: {error}')
    if 'traffic_result' in st.session_state:
        show_result(st.session_state['traffic_result'], st.session_state.get('traffic_record_id'))
    else:
        st.markdown('**Recorded history for past dates. ML estimates for upcoming traffic.**')
        st.caption('Estimates are saved automatically. Road and airport-corridor patterns come from the available archive; no camera or manual weather entry is needed.')


def saved_estimates(volume):
    st.title('Saved traffic estimates')
    st.caption('Each submitted estimate keeps its weather, calendar context, model version and anomaly scores. Measured traffic stays in the source datasets.')
    records = compare_with_observations(list_estimates(), volume)
    if records.empty:
        st.info('Your first traffic estimate will appear here.')
        return
    a, b, c = st.columns(3)
    a.metric('Saved estimates', len(records))
    b.metric('Daily forecasts awaiting observations', int(records.evaluation_status.eq('Awaiting measured daily count').sum()))
    errors = records.daily_absolute_error.dropna()
    c.metric('Daily forecast MAE', f'{errors.mean():,.0f}' if len(errors) else 'Awaiting actuals', help='Only forecasts made before the target time with newly available measured daily counts are scored.')
    view = records.drop(columns='id').copy()
    st.dataframe(view, hide_index=True, width='stretch')
    st.download_button('Export saved estimates (.csv)', view.to_csv(index=False), 'saved_traffic_estimates.csv', 'text/csv')
    with st.expander('Open a saved result'):
        record_id = st.selectbox('Saved request', records.id.tolist(), format_func=lambda value:
                                 f'{records.loc[records.id.eq(value), "road"].iloc[0]} · {records.loc[records.id.eq(value), "target_local"].iloc[0]} · {value[:8]}')
        if st.button('Open result'):
            st.session_state['opened_result'] = (record_id, get_estimate(record_id))
    if 'opened_result' in st.session_state:
        record_id, result = st.session_state['opened_result']
        show_result(result, record_id)


st.sidebar.markdown('### URBAN\nTRAFFIC INTELLIGENCE')
st.sidebar.caption('Your road. Your time. An informed estimate.')
workspace = st.sidebar.radio('Workspace', ['Traffic estimate', 'Saved estimates', 'Research explorer'])
st.sidebar.divider()
st.sidebar.caption('Weather connected through Open-Meteo. Calendar: Karnataka holidays and festivals. All times IST.')
if workspace == 'Research explorer':
    from traffic_intelligence.research_ui import render_research
    render_research()
else:
    paths = [ARTIFACTS / 'estimator' / 'model.joblib', ARTIFACTS / 'volume' / 'history.parquet', ARTIFACTS / 'journey' / 'history.parquet']
    if not all(path.exists() for path in paths):
        st.info('Prepare the data and trained estimator to start.')
        st.code('python -m traffic_intelligence train', language='bash')
        st.stop()
    volume, journey = load_histories(paths[1].stat().st_mtime_ns, paths[2].stat().st_mtime_ns)
    if workspace == 'Traffic estimate':
        predictor(load_product(paths[0].stat().st_mtime_ns), volume, journey)
    else:
        saved_estimates(volume)
    st.divider()
    st.caption('Urban Traffic Intelligence · Traffic Monitor Lizard data: Mahesh Shantaram / thecont1, CC BY 4.0 · Weather: Open-Meteo · Estimates support planning; unusual traffic does not confirm an incident.')
