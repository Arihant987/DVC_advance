import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
    '''
    Load the parameters from the yaml file
    '''
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Parameters loaded successfully from %s",params_path)
        return params 
    except FileNotFoundError as e:
        logger.error("File not found: %s",params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Failed to parse the yaml file: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading the parameters: %s",e)
        raise

def load_data(file_path:str)->pd.DataFrame:
    '''
    Load data from csv file
    '''
    try:
        df=pd.read_csv(file_path)
        # replace nan values with empty str ""
        df.fillna("",inplace=True)
        logger.debug("Data loaded successfully")
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading the data: %s",e)
        raise   

def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int)->tuple:
    '''
    Apply tfidf on the text data
    '''

    try:
        vectorizer=TfidfVectorizer(max_features=max_features)

        x_train=train_data['text'].values
        y_train=train_data['target'].values
        x_test=test_data['text'].values
        y_test=test_data['target'].values

        x_train_tfidf=vectorizer.fit_transform(x_train)
        x_test_tfidf=vectorizer.transform(x_test)   

        train_df=pd.DataFrame(x_train_tfidf.toarray())
        train_df['target']=y_train
        test_df=pd.DataFrame(x_test_tfidf.toarray())
        test_df['target']=y_test

        logger.debug("Tfidf applied successfully") 

        return train_df,test_df
    except Exception as e: 
        logger.error("Unexpected error while applying tfidf: %s",e)
        raise    

def save_data(df:pd.DataFrame,file_path:str)->None:
    '''
    Save the data to csv file
    '''
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug("Data saved successfully")
    except Exception as e:
        logger.error("Unexpected error while saving the data: %s",e)
        raise

def main():
    try:
        # max_features=500
        params=load_params('params.yaml')
        max_features=params['feature_engineering']['max_features']
        train_data=load_data('data/interim/train_preprocessed.csv')
        test_data=load_data('data/interim/test_preprocessed.csv')   
        train_df,test_df=apply_tfidf(train_data,test_data,max_features)

        save_data(train_df,os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df,os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.error("Unexpected error during feature engineering: %s",e)
        raise

if __name__=='__main__':
    main() 