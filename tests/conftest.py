import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_history():
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    pieces = []
    for name, offset in [('A',0),('B',1000)]:
        pieces.append(pd.DataFrame({'timestamp':dates,'entity':name,'label':name,'area':'Test',
                                    'y':100 + offset + np.arange(100)*2 + (np.arange(100)%7)*5,
                                    'weather':np.where(np.arange(100)%3==0,'Rain','Clear')}))
    return pd.concat(pieces, ignore_index=True)
