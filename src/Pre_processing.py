import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    '''
    Transforms the input txt into lowercase,tokenizing,
    removing stopwords and punctuation and stemming
    '''
    ps=PorterStemmer()
    text=text.lower()
    tokens=nltk.word_tokenize(text)
    words=[]
    for word in tokens:
        if(word.isalnum()):
            words.append(word)
    text=words[:] 
    words.clear()   
    for word in text:
        if(word not in stopwords.words('english') and word not in string.punctuation):
            words.append(word) 
    text=words[:]
    words.clear()
    for word in text:
        words.append(ps.stem(word))               
    return " ".join(words)

def preprocess_df(df,text_column='text',target_column='target'):
    '''
    Preprocess the df by encoding,removing duplicates
    and then doing text transformation
    '''
    try:
        logger.debug("Started pre-processing the data")
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug("Target column encoded")

        df=df.drop_duplicates(keep='first')
        logger.debug("Duplicates removed") 

        # text transformation
        # updates text_column
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug("Text column transformed")
        return df
    
    except KeyError as e:
        logger.error("Missing columns in the df: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error during pre-processing: %s",e)
        raise

def main(text_column='text',target_column='target'):
    '''
    Main func to load raw data,preprocess and save the data
    '''
    try:
        train_data=pd.read_csv('data/raw/train.csv')
        test_data=pd.read_csv('data/raw/test.csv')
        logger.debug("Data loaded successfully")

        train_processed_data=preprocess_df(train_data,text_column,target_column)
        test_processed_data=preprocess_df(test_data,text_column,target_column)

        data_path=os.path.join('data','interim')
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,'train_preprocessed.csv'),index=False)    
        test_processed_data.to_csv(os.path.join(data_path,'test_preprocessed.csv'),index=False)  

        logger.debug("Data preprocessed saved to %s",data_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s",e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("Empty data: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s",e)
        raise   

if __name__=='__main__':    
    main()