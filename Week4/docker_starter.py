import pickle
import pandas as pd
import numpy as np
import sys

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

year = int(sys.argv[1])
month = int(sys.argv[2])

#with open('/app/model2.bin', 'rb') as f_in:
    #dv = pickle.load(f_in)

with open('/app/model.bin', 'rb') as f_in:
    dv,model = pickle.load(f_in)
categorical = ['PULocationID', 'DOLocationID']
df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)
print(f"The mean is {np.mean(y_pred)}")
df_result = pd.DataFrame()
df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
y_pred_std = np.std(y_pred)
df_result['pred']=y_pred
df_result.to_parquet(
    f'test_duration_{year:04d}_{month:02d}.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)

