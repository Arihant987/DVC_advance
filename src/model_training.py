import os
import numpy as np
import pandas as pd 
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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

def train_model(x_train:np.ndarray,y_train:np.ndarray,params:dict)->RandomForestClassifier:
    '''                                      
    Train the model and in params {n_estimators,random_state}   
    '''
    try:
        # No. of rows should be same
        if(x_train.shape[0]!=y_train.shape[0]):
            raise ValueError("Shape mismatch between x_train and y_train")
        logger.debug("Initializing the model with parameters: %s",params)
        model=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])

        logger.debug("Model training started with %d samples",x_train.shape[0])
        model.fit(x_train,y_train)
        logger.debug("Model trained successfully")
        return model
    except ValueError as e:
        logger.error("Value error while training the model: %s",e)
    except Exception as e:
        logger.error("Unexpected error while training the model: %s",e)
        raise

def save_model(model,file_path:str)->None:
    '''
    Save the model
    '''
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        # wb written in binary mode
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug("Model saved successfully at %s",file_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error while saving the model: %s",e)
        raise

def main():
    try:
        params={'n_estimators':50,"random_state":2}
        train_data=load_data('data/processed/train_tfidf.csv')
        test_data=load_data('data/processed/test_tfidf.csv')
        x_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values

        model=train_model(x_train,y_train,params)
        save_model(model,'models/model.pkl')
    except Exception as e:  
        logger.error("Failed to train the model: %s",e)
        raise

if __name__=='__main__':
    main()
