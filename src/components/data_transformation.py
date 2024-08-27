import sys
import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

# Feature Engineering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

## Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Tranformation Initiated")

            # Define which columns should be label encoded, one hot encoded and which should be scaled
            numerical_cols =  ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            ohe_cols =  ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

            logging.info("Pipeline Initiated")

            ## Numerical Pipeline

            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ( "scaler", StandardScaler())

                ]
            )



            ## Ohe Pipeline

            ohe_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehotencoder", OneHotEncoder(sparse_output=False)),
                ("scaler", StandardScaler())   
                
                ]
            )


            # Combine all
            preprocessor = ColumnTransformer([
            ("num_pipeline", num_pipeline, numerical_cols),
            ("ohe_pipeline", ohe_pipeline, ohe_cols)
            ])


            logging.info("Pipeline Completed")

            return preprocessor


        except Exception as e:
            logging.info("Error in Data Tranformation")
            raise CustomException(e, sys)
        
    
    def initiate_data_tranformation(self, train_path, test_path):
        try:
            # Reading train and test data

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head : \n {train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n {test_df.head().to_string()}")

            logging.info("Obtaining Preprocesing Object")

            preprocessing_obj = self.get_data_transformation_obj()

            target_column_name = "income"
            drop_columns = [target_column_name, "fnlwgt", "education"]

            input_feature_train_df  = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df  = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Transforming using preprocessing object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying pre processing object on training and testing datasets")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Preprocessor Pickle file saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation")
            raise CustomException(e, sys)