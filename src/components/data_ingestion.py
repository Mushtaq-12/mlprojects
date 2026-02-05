import os
import sys  
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTranformation
# from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifact","train.csv")   #used to store in artifact folder and file name as train.csv
    test_data_path:str=os.path.join("artifact","test.csv")
    raw_data_path:str=os.path.join("artifact","raw.csv")

class DataIngestion:
    '''Data Ingestion = Collecting + Loading + Preparing raw data'''
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()   ## calling the dataIngestionconfig class
    
    def initate_data_ingestion(self):
        logging.info("Entered into the Data Ingestion Method or component")
        try:
            df=pd.read_csv('Notebook/data/stud.csv')  #reading the dataset
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)  # creates artifacts folder exist_ok=True → prevents error if folder already exists.
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)  #Saves original dataset copy.

            logging.info("Train Test split initiated")

            #Train Test Split
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            #Save Train Data and train data 
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data completed")
            #Return Paths of train,test dataset
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initate_data_ingestion()

    data_tranformation=DataTranformation()
    train_arr,test_arr,_=data_tranformation.initiate_data_transformer(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

