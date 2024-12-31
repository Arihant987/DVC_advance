import os 
import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path:str):
    '''
    Load the rrained model from the file path
    '''
    try:
        # rb is read binary mode
        with open(file_path,'rb') as file:
            model=pickle.load(file)
        logger.debug("Model loaded successfully from %s",file_path)
        return model
    except FileNotFoundError as e:
        logger.error("File not found: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading the model: %s",e)
        raise

def load_data(file_path:str)->pd.DataFrame:
    '''
    Load data from csv file
    '''
    try:
        df=pd.read_csv(file_path)
        logger.debug("Data loaded successfully from %s with shape %s",file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s",e)
        raise
    except FileNotFoundError as e:
        logger.error("File not found: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading the data: %s",e)
        raise

def evaluate_model(model,x_test:np.ndarray,y_test:np.ndarray)->dict:
    '''
    Evaluate the model on the test data and 
    return evaluation metrices
    in form of dictionary
    '''
    try:
        y_pred=model.predict(x_test)
        y_pred_prob=model.predict_proba(x_test)[:,1]

        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        roc_auc=roc_auc_score(y_test,y_pred_prob)

        metrics_dict={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'roc_auc':roc_auc
        }

        logger.debug("Model evaluated successfully")
        return metrics_dict
    except Exception as e:
        logger.error("Unexpected error while evaluating the model: %s",e)
        raise

def save_metrics(metrics:dict,file_path:str)->None:
    '''
    Save the evaluation metrices to a JSON file
    '''
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        # write mode
        with open(file_path,'w') as file:
            json.dump(metrics,file)
        logger.debug("Metrics saved successfully at %s",file_path)
    except Exception as e:
        logger.error("Unexpected error while saving the metrics: %s",e)
        raise

def main():
    try:
        model=load_model('models/model.pkl')
        test_data=load_data('data/processed/test_tfidf.csv')

        x_test=test_data.iloc[:,:-1].values
        y_test=test_data['target'].values

        metrics=evaluate_model(model,x_test,y_test)
        save_metrics(metrics,'reports/metrics.json')
    except Exception as e:
        logger.error("Unexpected error during evaluation of model: %s",e)
        raise

if __name__=='__main__':   
    main()