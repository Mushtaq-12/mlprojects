# Imputer--> fills in missing (NaN/null) values in a dataset

import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import os
from src.utils import save_object


'''
SimpleImputer	  :  Fill missing values
OneHotEncoder	  :  Convert category → numbers
StandardScaler	  :  Normalize data
Pipeline	      :  Chain preprocessing steps
ColumnTransformer :	 Apply pipelines to specific columns

'''

@dataclass
class DataTranformationConfig:
    preprocessor_obj_file=os.path.join('artifacts','preprocessor.pkl')

'''
Why save preprocessor?
Because:
✔ Need same transformation during prediction
✔ Avoid data leakage
✔ Reproducibility

'''

class DataTranformation:
    def __init__(self):
        self.data_transformation_config=DataTranformationConfig()
    
    def get_data_tranformer_object(self):
        '''this function is responsible for the data transformation'''
        try:
            numerical_columns=["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            cat_pipline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("One_hot_Encoder",OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipline,categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformer(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_tranformer_object()   #it does the column transfer
            target_column_name='math_score'

            #for training data
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            #for testing data
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            #now will do standard scaling for both test and train
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("saved preprocessing object")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessor_obj
            )
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file)
        except Exception as e:
            raise CustomException(e,sys)