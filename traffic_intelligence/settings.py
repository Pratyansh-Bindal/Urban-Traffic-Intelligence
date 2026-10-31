from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data' / 'raw'
ARTIFACTS = ROOT / 'artifacts'
EVENTS = ROOT / 'data' / 'events.csv'
PREDICTIONS_DB = ROOT / 'data' / 'predictions.sqlite3'
TML_REPOSITORY = 'thecont1/traffic-monitor-lizard'
TML_REVISION = 'fd4b32fb4c7507437fb7f5711d64b986e78bb2bb'
SEED = 42
TIMEZONE = 'Asia/Kolkata'
DATASETS = {
    'volume': {'title': 'Daily vehicle volume', 'unit': 'vehicles', 'frequency': '1D', 'lookback': '28D'},
    'journey': {'title': 'Hourly journey duration', 'unit': 'minutes', 'frequency': '1h', 'lookback': '28D'},
}

SOURCES = [
    {'title': 'Traffic Monitor Lizard', 'url': 'https://github.com/thecont1/traffic-monitor-lizard',
     'role': 'Google Maps route-duration estimates and associated weather. Data: CC BY 4.0; attribute Mahesh Shantaram / thecont1. No upstream application code is incorporated.'},
    {'title': 'Bangalore City Traffic Dataset', 'url': 'https://www.kaggle.com/datasets/preethamgouda/banglore-city-traffic-dataset',
     'role': 'User-supplied daily counts. Collection provenance and redistribution licence remain unverified.'},
    {'title': 'Kendre et al. (2024): Traffic Volume Prediction Based on Weather Parameters',
     'url': 'https://doi.org/10.1109/ICWITE59797.2024.10503349',
     'role': 'Motivates the with-weather / without-weather experiment. Publisher abstract verified; full text was not supplied. No published effect size is assumed.'},
    {'title': 'Du et al. (2025): Traffic prediction in time series, spatialtemporal, and OD data',
     'url': 'https://doi.org/10.1016/j.jtte.2025.03.001', 'role': 'Task definitions, temporal structure and data limitations.'},
    {'title': 'Qin et al. (2026): Modern Deep Learning Techniques for Traffic Volume Prediction',
     'url': 'https://doi.org/10.1007/s41019-025-00340-8', 'role': 'Model families and benchmark methodology; advanced architectures are future work.'},
    {'title': 'Ravichandran et al. (2025): Gaussian dual adjacency graph traffic prediction',
     'url': 'https://doi.org/10.1038/s41598-025-25075-4', 'role': 'Bengaluru modelling reference. Shuffled K-fold and area-classification metrics prevent a direct comparison with our future-date regression.'},
    {'title': 'Djenouri et al. (2019): Urban Traffic Anomalies Detection Algorithms',
     'url': 'https://doi.org/10.1109/ACCESS.2019.2893124', 'role': 'Flow versus trajectory anomalies and statistical detection taxonomy.'},
    {'title': 'Chew & Asli (2024): Vehicular Traffic Flow Anomaly Detection Using ML',
     'url': 'https://doi.org/10.1051/itmconf/20246301023', 'role': 'Preprocessing, detector comparison and evaluation limitations.'},
    {'title': 'Python holidays: Karnataka calendar', 'url': 'https://holidays.readthedocs.io/',
     'role': 'Versioned India / Karnataka holiday dates. Lunar and Islamic dates may be estimated; confirm local observances when using future scenarios.'},
    {'title': 'Karnataka DTE: Kannada Rajyotsava observance',
     'url': 'https://dtek.karnataka.gov.in/storage/pdf-files/C-20%20syllabus/Balake%20Kannada-2.pdf',
     'role': 'Annual 1 November state celebration is retained as traffic context, including Sundays; this does not assert an additional statutory office closure.'},
    {'title': 'Open-Meteo forecast API', 'url': 'https://open-meteo.com/en/docs',
     'role': 'Automatically fetched Bengaluru weather for a submitted road/date/time request. Forecasts, recent model history and historical reanalysis are labelled separately from traffic measurements.'},
]
