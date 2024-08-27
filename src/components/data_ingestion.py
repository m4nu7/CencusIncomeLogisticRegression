import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


# Initialise Data Ingestion configaration

@dataclass
class DataIngestionConfig:
    raw_data_path : str = os.path.join("artifacts", "raw.csv")
    train_data_path : str = os.path.join("artifacts", "train.csv")
    test_data_path : str = os.path.join("artifacts", 'test.csv')


# Create a class for Data Ingestion
class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")

        try:
            df1 = pd.read_csv(os.path.join("./notebooks/data", "adult_train.csv"))
            df2 = pd.read_csv(os.path.join("./notebooks/data", "adult_test.csv"))
            df = pd.concat([df1, df2])
            logging.info("Dataset read as pandas Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Train Test split data")
            df1.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            df2.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Error occured at Data Ingestion Stage")
            raise CustomException(e, sys)