"""Append-only estimate records, kept separate from the measured training datasets."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import pandas as pd

from .settings import PREDICTIONS_DB


@contextmanager
def _connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute('''CREATE TABLE IF NOT EXISTS estimates (
        id TEXT PRIMARY KEY, created_at_utc TEXT NOT NULL, road TEXT NOT NULL,
        target_local TEXT NOT NULL, result_kind TEXT NOT NULL,
        hourly_estimate REAL NOT NULL, daily_estimate REAL NOT NULL,
        observed_daily REAL, payload TEXT NOT NULL)''')
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def save_estimate(result, path=None):
    record_id = str(uuid4())
    created = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(result, allow_nan=False)
    with _connect(path or PREDICTIONS_DB) as connection:
        connection.execute('INSERT INTO estimates VALUES (?,?,?,?,?,?,?,?,?)',
                           (record_id, created, result['road'], result['requested_at'], result['kind'],
                            result['hourly_estimate'], result['daily_estimate'], result['observed_daily'], payload))
    return record_id


def list_estimates(path=None, limit=1000, *, record_ids=None):
    path = Path(path or PREDICTIONS_DB)
    columns = ['id', 'created_at_utc', 'road', 'target_local', 'result_kind', 'hourly_estimate', 'daily_estimate', 'observed_daily']
    if not path.exists() or record_ids == []:
        return pd.DataFrame(columns=columns)
    clause, params = '', []
    if record_ids is not None:
        clause = ' WHERE id IN (' + ','.join('?' for _ in record_ids) + ')'
        params.extend(record_ids)
    params.append(int(limit))
    with _connect(path) as connection:
        return pd.read_sql_query('SELECT ' + ','.join(columns) + ' FROM estimates' + clause + ' ORDER BY created_at_utc DESC LIMIT ?',
                                 connection, params=params)


def get_estimate(record_id, path=None, *, record_ids=None):
    if record_ids is not None and record_id not in record_ids:
        raise ValueError('Saved estimate not found.')
    path = Path(path or PREDICTIONS_DB)
    if not path.exists():
        raise ValueError('Saved estimate not found.')
    with _connect(path) as connection:
        row = connection.execute('SELECT payload FROM estimates WHERE id=?', (record_id,)).fetchone()
    if row is None:
        raise ValueError('Saved estimate not found.')
    return json.loads(row[0])


def compare_with_observations(records, history, *, as_of=None):
    """Reconcile daily forecasts only; predicted hourly counts have no observed label."""
    if records.empty:
        return records.copy()
    out = records.copy()
    out['date'] = pd.to_datetime(out.target_local).dt.normalize()
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz='Asia/Kolkata')
    if now.tzinfo is not None:
        now = now.tz_convert('Asia/Kolkata').tz_localize(None)
    observed = history.loc[history.timestamp + pd.Timedelta('1D') <= now, ['entity', 'timestamp', 'y']].rename(columns={'entity': 'road', 'timestamp': 'date', 'y': 'actual_daily'})
    out = out.merge(observed, how='left', on=['road', 'date'])
    future_at_creation = pd.to_datetime(out.target_local).dt.tz_localize('Asia/Kolkata').dt.tz_convert('UTC') >= pd.to_datetime(out.created_at_utc, utc=True)
    out['evaluation_status'] = 'Historical estimate / lookup'
    out.loc[future_at_creation, 'evaluation_status'] = 'Awaiting measured daily count'
    scored = future_at_creation & out.actual_daily.notna() & out.observed_daily.isna()
    out.loc[scored, 'evaluation_status'] = 'Measured daily count available'
    out['daily_absolute_error'] = (out.daily_estimate - out.actual_daily).abs().where(scored)
    return out.drop(columns='date')
