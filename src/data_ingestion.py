import  pandas as pd 
import os
import logging
from sklearn.model_selection import train_test_split

log_dirs='logs'
os.makedirs(log_dirs,exist_ok=True)

logger=logging.getLogger('data_ingestion') # name of logger
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dirs,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -> return type
def load_data(data_url:str)->pd.DataFrame:
    '''
    Load data from csv file
    '''
    try:
        df=pd.read_csv(data_url)
        logger.debug("Data loaded from %s",data_url)
        return df
    except pd.errors.ParseError as e:
        logger.error("Failed to parse the csv file: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected erorr while loading the data: %s",e)
        raise

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    '''
    Preprocess the data
    '''
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug("Data preprocessed successfully")
        return df
    except KeyError as e:
        logger.error("Missing columns in the df: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error during pre-processing: %s",e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    '''
    Save train and test data.
    '''
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logger.debug("Train and test data saved to: %s",raw_data_path)
    except Exception as e:
        logger.error("Unexpected error while saving the data: %s",e)
        raise  

def main():
    try:
        test_size=0.2
        data_path='https://raw.githubusercontent.com/Arihant987/DVC_advance/refs/heads/main/experiments/spam.csv'
        df=load_data(data_url=data_path)
        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=2)
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.error("Failed to run the data ingestion pipeline: %s",e)
        raise

if __name__=='__main__':
    main()
