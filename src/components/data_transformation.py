import sys
import os
from dataclasses import dataclass
import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTranformationConfig():
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTranformation():
    def __init__(self):
        self.data_tranformation_config = DataTranformationConfig()

    def get_data_tranformer_object(self):
        '''This function is responsible for data transformation'''
        try:
            numerical_columns= ["writing score","reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )
            cat_pipline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling completed")
            logging.info("categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                ("num_pipline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipline,categorical_columns)
                ]

            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

            pass    