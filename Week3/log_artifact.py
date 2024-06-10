if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import mlflow
import mlflow.sklearn
import joblib
@data_exporter
def export_data(artifacts):
    
    print('Get Artifacts')
    lr , dv = artifacts
    mlflow.set_tracking_uri("http://mlflow:5000")
    print('MLFlow Tracking URI started')
    
    mlflow.set_experiment('Log_Mage_Experiment')
    print('The Mage Experiment is set up')
    with mlflow.start_run():
        print('MLFlow run started')    
        mlflow.sklearn.log_model(lr,'lr_model')
        dict_vectorizer_path = "dv.pkl"
        joblib.dump(dv, dict_vectorizer_path)
        mlflow.log_artifact(dict_vectorizer_path)
    print('MLFlow Logging finished')