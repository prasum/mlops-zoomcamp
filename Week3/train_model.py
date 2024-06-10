from typing import Dict, List, Optional, Tuple

import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_exporter
def export(data) :
    
    df = data.iloc[:,:]
    target = 'duration'

    print('Start dict vectorization')
    dv = DictVectorizer()

    train_dicts = df[['PULocationID','DOLocationID']].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    y_train = df[target]
    print('Model training started')

    lr = LinearRegression()
    lr.fit(X_train, y_train)    
    print(lr.intercept_)

    print('Model training done')
   
    return  lr, dv

@test
def test_output(lr, dv):
    assert lr is not None
    assert dv is not None