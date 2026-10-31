"""Run with: python -m streamlit run app.py"""
from html import escape
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from traffic_intelligence.calendar import calendar_ics, custom_events, holiday_table
from traffic_intelligence.settings import ARTIFACTS, DATASETS, SOURCES, TIMEZONE
from traffic_intelligence.weather import fetch_outlook

COLORS = ['#0D8075', '#DB9B46', '#476C9B', '#9D697C']
PAGES = ['Main estimator evaluation', 'Overview', 'Festival calendar', 'Weather & research', 'Anomaly explorer', 'Model evaluation', 'Data audit']


def read_json(path):
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False)
def load_tables(dataset, version):
    folder = ARTIFACTS / dataset
    return (pd.read_parquet(folder / 'history.parquet'), pd.read_parquet(folder / 'predictions.parquet'),
            read_json(folder / 'report.json'), read_json(folder / 'audit.json'),
            read_json(folder / 'effects.json'), read_json(folder / 'feature_importance.json'),
            read_json(folder / 'synthetic_anomalies.json'))


@st.cache_resource(show_spinner=False)
def load_bundle(dataset, version):
    return joblib.load(ARTIFACTS / dataset / 'model.joblib')


@st.cache_data(ttl=3600, show_spinner=False)
def outlook():
    return fetch_outlook()


def chart(fig, height=350):
    fig.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='DM Sans, sans-serif', color='#365260'), margin=dict(l=10,r=10,t=35,b=10),
                      height=height, legend=dict(orientation='h', y=1.14, x=0), hovermode='x unified')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#E4EBEF', zeroline=False)
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


def heading(title, detail):
    st.markdown('<div class="eyebrow">BENGALURU / URBAN TRAFFIC INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="note">{escape(detail)}</p>', unsafe_allow_html=True)




def render_research():
    source = st.sidebar.radio('Analysis branch', ['Journey time', 'Daily volume'])
    dataset = 'journey' if source == 'Journey time' else 'volume'
    folder = ARTIFACTS / dataset
    if not (folder / 'model.joblib').exists():
        st.info('Prepare the local research data and trained models to open the dashboard.')
        st.code('python -m traffic_intelligence fetch --volume "/path/to/Banglore_traffic_Dataset.csv"\npython -m traffic_intelligence train', language='bash')
        st.stop()
    version = (folder / 'model.joblib').stat().st_mtime_ns
    history, predictions, report, audit, effects, importance, synthetic = load_tables(dataset, version)
    bundle = load_bundle(dataset, version)
    unit = DATASETS[dataset]['unit']
    period = pd.Timedelta(DATASETS[dataset]['frequency'])
    options = history[['entity', 'label']].drop_duplicates().sort_values('label')
    labels = options.set_index('entity').label.to_dict()
    entity = st.sidebar.selectbox('Route' if dataset == 'journey' else 'Road / intersection', options.entity.tolist(), format_func=lambda x: labels[x])
    page = st.sidebar.selectbox('Research view', PAGES)
    st.sidebar.divider()
    st.sidebar.caption(f'Snapshot through {history.timestamp.max():%d %b %Y}. Times: Asia/Kolkata.')
    st.sidebar.caption('Google Maps journey estimates' if dataset == 'journey' else 'Daily counts: collection provenance unverified')
    route_history = history.loc[history.entity.eq(entity)].sort_values('timestamp')
    route_predictions = predictions.loc[predictions.entity.eq(entity)].sort_values('timestamp')
    selected_metric = next(m for m in report['models'] if m['model'] == report['winner'])

    if page == 'Main estimator evaluation':
        heading('Evaluate the road and time estimator.', 'The conditional models used by the main form. Their results are separate from the advance-forecast experiments below.')
        product = read_json(ARTIFACTS / 'estimator' / 'report.json')
        for name, title in [('daily_volume', 'Daily vehicle volume'), ('travel_pace', 'Travel patterns')]:
            evaluation = product['models'][name]
            st.subheader(title)
            st.caption(f'Model: {evaluation["model"]} · unit: {evaluation["unit"]}')
            a, b, c = st.columns(3)
            a.metric('Test MAE', f'{evaluation["selected_test"]["mae"]:,.3f}')
            b.metric('Test RMSE', f'{evaluation["selected_test"]["rmse"]:,.3f}')
            c.metric('Test R²', f'{evaluation["selected_test"]["r2"]:.3f}')
            rows = [{'Model': m['model'], 'Selection MAE': m['selection']['mae'], 'Test MAE': m['test']['mae'],
                     'Test RMSE': m['test']['rmse'], 'Test R²': m['test']['r2']} for m in evaluation['models']]
            rows.append({'Model': 'Historical weekday/hour baseline', 'Selection MAE': None,
                         'Test MAE': evaluation['baseline_test']['mae'], 'Test RMSE': evaluation['baseline_test']['rmse'],
                         'Test R²': evaluation['baseline_test']['r2']})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            omitted = 'weather category' if name == 'daily_volume' else 'time and corridor features'
            st.write(f'Adding {omitted}: **{evaluation["ablation"]["mae_improvement_pct"]:+.2f}% test MAE reduction** in a matched-model comparison.')
            st.caption('Negative reduction means the added features made this experiment worse. The main model is selected among the ML candidates on validation dates; a baseline can perform better on test dates.')
        st.info(product['evaluation_protocol'])
        st.write(product['hourly_volume_protocol'])
        st.caption('Hourly count accuracy is not reported without hourly count labels. Stored predictions remain estimates.')
        st.download_button('Download main estimator evaluation', json.dumps(product, indent=2), 'estimator_evaluation.json', 'application/json')

    elif page == 'Overview':
        heading('A clearer view of how the city moves.', f'{labels[entity]} · {DATASETS[dataset]["title"]}')
        st.markdown('<div class="hero"><h2>Find the pattern. Understand the exception.</h2><p>Explore recorded traffic, compare forecasts with what happened, and investigate the calendar and weather conditions around unusual journeys.</p></div>', unsafe_allow_html=True)
        a,b,c,d = st.columns(4)
        a.metric(f'Typical recorded value ({unit})', f'{route_history.y.median():,.1f}')
        local_mae = float((route_predictions.y - route_predictions.prediction).abs().mean())
        b.metric(f'This route test MAE ({unit})', f'{local_mae:,.2f}')
        c.metric('Monitored routes' if dataset == 'journey' else 'Monitored roads', f'{history.entity.nunique()}')
        d.metric('Completed observations', f'{len(history):,}')
        st.subheader('The recent picture')
        days = st.select_slider('History window', [7, 14, 30, 90], value=14)
        recent = route_history.loc[route_history.timestamp >= route_history.timestamp.max() - pd.Timedelta(days=days)]
        fig = px.line(recent, x='timestamp', y='y', color_discrete_sequence=COLORS, labels={'timestamp':'Local period start', 'y':unit.capitalize()})
        fig.update_traces(line_width=2)
        chart(fig, 330)
        left,right = st.columns([1.35,1])
        with left:
            st.subheader('Weekly rhythm')
            data = route_history.assign(weekday=route_history.timestamp.dt.day_of_week, hour=route_history.timestamp.dt.hour)
            if dataset == 'journey':
                grid = data.pivot_table(index='weekday', columns='hour', values='y', aggfunc='median').reindex(range(7))
                fig = px.imshow(grid, labels={'x':'Hour of day','y':'Weekday','color':'Minutes'}, color_continuous_scale=['#EDF4F2','#74B7A8','#0D6B63'], aspect='auto')
                fig.update_yaxes(tickvals=list(range(7)), ticktext=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
                chart(fig, 310)
            else:
                grouped = data.groupby('weekday').y.median().reindex(range(7)).rename_axis('weekday').reset_index()
                grouped['day'] = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
                chart(px.bar(grouped, x='day', y='y', color_discrete_sequence=COLORS, labels={'y':'Median vehicles','day':'Weekday'}),310)
        with right:
            st.subheader('What this forecast knows')
            st.write(f'**Selected model:** {report["winner"]}')
            if report['winner'] == 'Road / route median':
                st.write('This validation-selected research baseline uses the road/route median. The main estimator uses conditional ML with weather, calendar and travel patterns.')
            else:
                st.write('Past traffic, weekday and season; Karnataka holiday and festival context; weather available before the forecast period.')
            st.caption('Models compete against simple historical baselines. The winner is selected using validation dates.')
            if dataset == 'journey':
                st.info(f'Weather accompanies {audit["weather_observed_fraction"]:.1%} of hourly records. Missing weather is explicitly represented.')
            else:
                st.info(f'Road-date coverage is {audit["coverage_fraction"]:.1%}. Daily totals cannot reveal rush-hour patterns.')

    elif page == 'Festival calendar':
        heading('Put the date in context.', 'Karnataka holidays and festival windows, connected to traffic through calendar features.')
        year = st.selectbox('Calendar year', list(range(2022, 2029)), index=4)
        calendar = holiday_table(year, year).copy()
        a,b,c = st.columns(3)
        a.metric('Public holiday dates', int(calendar.is_holiday.sum()))
        b.metric('Festival dates', int(calendar.is_festival.sum()))
        c.metric('Window around festivals', '±3 days')
        view = st.radio('Show', ['Festivals', 'All public holidays'], horizontal=True)
        selected = calendar.loc[calendar.is_festival.eq(1) if view == 'Festivals' else calendar.is_holiday.eq(1)].copy()
        selected['date'] = selected.date.dt.date
        st.dataframe(selected[['date','holiday_name','is_festival']], hide_index=True, width='stretch', column_config={'is_festival':st.column_config.CheckboxColumn('Festival')})
        st.download_button('Export calendar (.ics)', calendar_ics(year), f'karnataka_holidays_{year}.ics', 'text/calendar')
        st.caption('Import the ICS into Google Calendar, Apple Calendar or Outlook. Calendar dates come from python-holidays; lunar dates and local observances may require confirmation.')
        st.subheader('Did calendar features help the held-out forecast?')
        effect = next(x for x in report['ablations'] if x['feature_group']=='Festival calendar')
        a,b,c = st.columns(3)
        a.metric(f'Without festival features · MAE ({unit})', f'{effect["without_mae"]:,.3f}')
        b.metric(f'With festival features · MAE ({unit})', f'{effect["with_mae"]:,.3f}')
        c.metric('Error reduction', f'{effect["mae_improvement_pct"]:+.2f}%')
        st.caption('Same boosting algorithm and chronological test rows. Negative error reduction means these features made this experiment worse.')
        st.subheader('Custom local events')
        st.write('Add announced events to `data/events.csv`, then retrain. Use the exact road name or route code, or `*` for all entities. `known_from` prevents a later announcement from becoming an earlier forecast input.')
        st.code('date,name,entity,source,known_from\n2026-11-08,Community event,*,https://organizer.example/event,2026-10-20', language='csv')
        st.caption('The row above illustrates the input format; it is not an actual scheduled event.')
        try:
            events = custom_events()
            if not events.empty:
                st.dataframe(events, hide_index=True, width='stretch')
        except ValueError as error:
            st.error(str(error))

    elif page == 'Weather & research':
        heading('Test the weather hypothesis.', 'Weather is an input to investigate. Rain is not assigned a universal congestion multiplier.')
        a,b = st.columns([1.4,1])
        with a:
            st.subheader('With weather / without weather')
            effect = next(x for x in report['ablations'] if x['feature_group']=='Weather history')
            comparison = pd.DataFrame({'Model':['Without weather','With weather'], 'MAE':[effect['without_mae'],effect['with_mae']]})
            chart(px.bar(comparison, x='Model', y='MAE', color='Model', color_discrete_sequence=COLORS, text_auto='.3f'),280)
            st.metric('Held-out error reduction', f'{effect["mae_improvement_pct"]:+.2f}%')
            st.caption('Negative means weather did not improve this experiment. Only weather from earlier completed periods is used; stale weather becomes missing after 3 hours (journeys) or 48 hours (daily counts).')
        with b:
            st.subheader('Why this experiment exists')
            st.write('Kendre et al. compare weather-aware traffic prediction algorithms. Their study motivates our controlled comparison; its reported best algorithm and effect sizes are not transferred to Bengaluru data.')
            st.link_button('Read the weather paper', 'https://doi.org/10.1109/ICWITE59797.2024.10503349')
            st.caption('Publisher abstract checked. The paper’s full text was not supplied.')
            st.write(f'Historical weather available for **{audit["weather_observed_fraction"]:.1%}** of this branch’s records.')
            st.info('Model input effects and historical associations do not establish that weather caused a traffic change.')
        st.subheader('Historical association by weather')
        st.caption('All roads/routes in this analysis branch, using training dates only.')
        observed = pd.DataFrame(effects['weather'])
        observed['difference_from_context_pct'] = (observed.median_relative - 1)*100
        st.dataframe(observed[['weather','rows','dates','median_value','difference_from_context_pct']].round({'median_value':2,'difference_from_context_pct':2}), hide_index=True, width='stretch')
        st.caption(effects['note'])
        st.subheader('Optional seven-day weather outlook')
        st.write('Fetch Bengaluru’s public city-centre forecast to explore near-term weather scenarios. This makes one request to Open-Meteo and needs no API key.')
        if st.button('Fetch weather outlook'):
            try:
                with st.spinner('Loading Bengaluru weather outlook…'):
                    weather_data, metadata = outlook()
                st.session_state['weather_outlook'] = weather_data
                st.session_state['weather_metadata'] = metadata
            except Exception as error:
                st.error(f'Weather outlook could not be loaded. The offline models remain available. {error}')
        if 'weather_outlook' in st.session_state:
            weather_data = st.session_state['weather_outlook']
            chart(px.line(weather_data, x='timestamp', y='temperature', color_discrete_sequence=COLORS, labels={'temperature':'Temperature (°C)'}),260)
            fetched_local = pd.Timestamp(st.session_state['weather_metadata']['retrieved_at_utc']).tz_convert(TIMEZONE).strftime('%d %b %Y, %H:%M IST')
            st.caption(f'Open-Meteo · fetched {fetched_local}. The main Traffic estimate form fetches weather automatically.')
        with st.expander('Research sources and attribution'):
            for source_item in SOURCES:
                st.markdown(f'**[{source_item["title"]}]({source_item["url"]})** — {source_item["role"]}')

    elif page == 'Anomaly explorer':
        heading('Investigate the unusual.', 'An alert appears after a measurement is available. It identifies a deviation, not a confirmed accident.')
        detector = st.selectbox('Detection method', ['MAD','Isolation Forest','Forecast residual'])
        flag = 'alert_' + detector.lower().replace(' ','_')
        route = route_predictions.copy()
        a,b,c = st.columns(3)
        a.metric('Held-out observations', len(route))
        b.metric('Alerts', int(route[flag].sum()))
        c.metric('Alert rate', f'{route[flag].mean():.1%}')
        fig = go.Figure(go.Scatter(x=route.timestamp, y=route.y, name='Observed', line=dict(color=COLORS[0],width=1.5)))
        flagged = route.loc[route[flag]]
        fig.add_trace(go.Scatter(x=flagged.timestamp, y=flagged.y, name='Flagged', mode='markers', marker=dict(color='#C65E4B',size=7)))
        chart(fig, 360)
        st.dataframe(flagged[['timestamp','y','prediction','residual','mad_score','weather','holiday_name']].tail(100), hide_index=True, width='stretch')
        st.download_button('Download route alerts', flagged.to_csv(index=False), 'traffic_alerts.csv', 'text/csv')
        st.subheader('Synthetic stress test')
        st.caption(synthetic['experiment'])
        st.dataframe(pd.DataFrame(synthetic['results']), hide_index=True, width='stretch')
        st.warning(synthetic['limitation'])

    elif page == 'Model evaluation':
        heading('Make every result earn its place.', 'Branch-wide scores across all monitored roads/routes. Independent model-selection, calibration and test dates; the chart below follows the selected route.')
        a,b,c = st.columns(3)
        a.metric('Validation-selected model', report['winner'])
        b.metric(f'Test MAE ({unit})', f'{selected_metric["test"]["mae"]:,.3f}')
        c.metric('Measured interval coverage', f'{report["test_interval_coverage"]:.1%}', help='90% nominal band')
        scores = pd.DataFrame([{'Model':m['model'], 'Selection MAE':m['selection']['mae'], 'Test MAE':m['test']['mae'],
                                'Test RMSE':m['test']['rmse'], 'Test R²':m['test']['r2'], 'Training seconds':m['training_seconds']} for m in report['models']])
        st.dataframe(scores, hide_index=True, width='stretch')
        st.caption('The selected model minimises selection-period MAE. Test scores are displayed only for the final comparison; they do not select the model.')
        st.subheader('Prediction versus observation')
        tail = route_predictions.tail(240 if dataset=='journey' else 90)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tail.timestamp,y=tail.upper,name='Upper band',line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=tail.timestamp,y=tail.lower,name='90% band',line=dict(width=0),fill='tonexty',fillcolor='rgba(13,128,117,.10)'))
        fig.add_trace(go.Scatter(x=tail.timestamp,y=tail.y,name='Observed',line=dict(color=COLORS[0])))
        fig.add_trace(go.Scatter(x=tail.timestamp,y=tail.prediction,name='Forecast',line=dict(color=COLORS[1])))
        chart(fig, 330)
        left,right = st.columns(2)
        with left:
            st.subheader('Calendar-date partitions')
            st.dataframe(pd.DataFrame(report['splits']).T, width='stretch')
        with right:
            st.subheader('What the full model uses')
            ranked = pd.DataFrame(importance).head(10).sort_values('mae_increase')
            chart(px.bar(ranked, x='mae_increase', y='feature', orientation='h', color_discrete_sequence=COLORS, labels={'mae_increase':'MAE increase when shuffled','feature':''}),320)
            st.caption('Permutation importance on validation data. It describes model reliance, not causal influence.')
        st.caption(report['evaluation'] + ' ' + report['interval_note'])
        st.download_button('Download evaluation report', json.dumps(report, indent=2), f'{dataset}_evaluation.json', 'application/json')
        st.download_button('Download held-out predictions', (folder/'predictions.csv').read_bytes(), f'{dataset}_predictions.csv', 'text/csv')

    else:
        heading('Know the limits of the evidence.', 'Sources, missing observations and measurement definitions remain visible alongside results.')
        st.subheader('This branch')
        st.json(audit, expanded=False)
        a,b,c = st.columns(3)
        a.metric('First recorded period', f'{history.timestamp.min():%d %b %Y}')
        b.metric('Latest recorded period', f'{history.timestamp.max():%d %b %Y}')
        c.metric('Weather coverage', f'{audit["weather_observed_fraction"]:.1%}')
        all_audit = read_json(ARTIFACTS/'audit.json')
        st.subheader('Can the two datasets be joined?')
        st.info(f'{all_audit["fusion"]["overlapping_dates"]} overlapping dates. {all_audit["fusion"]["reason"]} Vehicle counts and journey durations remain independent targets.')
        if dataset == 'volume':
            st.warning('Environmental Impact is exactly 50 + 0.002 × Traffic Volume. It is excluded from the predictors. Same-period speed, congestion, incidents and weather are also excluded from advance forecasting.')
        else:
            st.warning('These are Google Maps estimates. Route distances vary over time, and weather is absent in much of the archive. Unusual readings need investigation before being called incidents.')
        st.subheader('Reproducible snapshot')
        st.json(bundle['manifest'], expanded=False)
        st.dataframe(history[['entity','label']].drop_duplicates(), hide_index=True, width='stretch')
        st.download_button('Download data audit', json.dumps(audit, indent=2), f'{dataset}_audit.json', 'application/json')

    st.divider()
    st.caption('Urban Traffic Intelligence · Offline research and clearly labelled scenarios · Traffic Monitor Lizard data: Mahesh Shantaram / thecont1, CC BY 4.0 · All times Asia/Kolkata')
