import hashlib

import numpy as np
import pandas as pd
import pytest

from traffic_intelligence.anomalies import rolling_reference
from traffic_intelligence.features import make_features
from traffic_intelligence.forecast import predict_one
from traffic_intelligence.models import build_model, conformal_radius
from traffic_intelligence.settings import EVENTS


def test_preprocessing_handles_unseen_categories_without_refitting(daily_history):
    features=make_features(daily_history,'volume')
    train=features.loc[features.timestamp<'2023-03-01']
    model=build_model('Boosting: full',quick=True).fit(train)
    unknown=features.tail(1).copy()
    unknown['weather_last']='Previously unseen weather'
    prediction=model.predict(unknown)
    assert len(prediction)==1 and np.isfinite(prediction[0]) and prediction[0]>=0
    category=model.estimator.named_steps['preprocess'].named_transformers_['category'].named_steps['encode'].categories_[1]
    assert 'Previously unseen weather' not in category


def test_anomaly_reference_excludes_observation_being_scored(daily_history):
    features=make_features(daily_history,'volume')
    original=rolling_reference(features,'volume')
    changed=features.copy()
    mask=changed.entity.eq('A') & changed.timestamp.eq('2023-03-01')
    changed.loc[mask,'y']=100_000
    scored=rolling_reference(changed,'volume')
    row=original.entity.eq('A') & original.timestamp.eq('2023-03-01')
    assert scored.loc[row,'reference_median'].iloc[0]==original.loc[row,'reference_median'].iloc[0]
    assert bool(scored.loc[row,'mad_alert'].iloc[0])


def test_constant_history_has_finite_fallback_scale(daily_history):
    data=daily_history.copy()
    data['y']=100.
    features=make_features(data,'volume')
    scored=rolling_reference(features,'volume')
    usable=scored.loc[scored.reference_median.notna()]
    assert usable.reference_scale.gt(0).all()
    assert np.isfinite(usable.mad_score).all()
    assert not usable.mad_alert.any()


def test_conformal_quantile_accounts_for_finite_sample_rank():
    assert conformal_radius(np.arange(1,11),coverage=.9)==10
    with pytest.raises(ValueError):
        conformal_radius([])


def test_forecast_respects_horizon_and_separates_manual_uplift(daily_history):
    features=make_features(daily_history,'volume')
    median=build_model('Road / route median').fit(features.loc[features.timestamp<'2023-03-01'])
    full=build_model('Boosting: full',quick=True).fit(features.loc[features.timestamp<'2023-03-01'])
    bundle={'dataset':'volume','winner':'Road / route median','models':{'Road / route median':median,'Boosting: full':full},
            'radius_by_entity':{},'global_radius':10.,'report':{'splits':{'train':{'last_date':'2023-02-28'}}},
            'manifest':{'custom_events_sha256':hashlib.sha256(EVENTS.read_bytes()).hexdigest()}}
    one=predict_one(bundle,daily_history,'A','2023-04-11',uplift_pct=25)
    assert one['history_age_hours']==0
    assert one['lower'] is not None
    assert one['scenario_prediction']==pytest.approx(one['scenario_model_prediction']*1.25)
    stale=predict_one(bundle,daily_history,'A','2024-04-11')
    assert stale['lower'] is None and stale['upper'] is None and stale['warnings']
    with pytest.raises(ValueError,match='not represented'):
        predict_one(bundle,daily_history,'Unknown','2023-04-11')
    with pytest.raises(ValueError,match='after the training'):
        predict_one(bundle,daily_history,'A','2023-02-01')
