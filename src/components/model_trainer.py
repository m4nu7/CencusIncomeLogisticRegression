import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


from src.utils import save_object, evaluate_model



@dataclass
class ModelTrainerConfig:
    trainer_model_file_path : str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array) :
        try:
            logging.info("Splitting independent and dependent features from train and test data")

            X_train, X_test, y_train, y_test = (
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )

            # Parameters for GridSearchCV
            parameters = {
            'C' : [0.5,1,2,3,5,6,7,9,10,30,45],
            'penalty' : ['l1', 'l2', 'elasticnet', None],
            "solver" : ['liblinear','saga']
            }
        

            models = {
            "LogisticRegression" : LogisticRegression(),
            "GridSearchCV_logRegressor" : GridSearchCV(estimator = LogisticRegression(), param_grid = parameters, scoring = "accuracy", cv=5)

            }


            model_report : dict = evaluate_model(X_train, X_test, y_train, y_test, models)
            
            print(model_report)
            print("\n============================================================================================")
            logging.info(f"Model_report : {model_report}")

            # Best model score
            best_model_score = max(sorted(model_report.values()))

            # Best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_name == "GridSearchCV_logRegressor":
                print(f"Best Model Found, Model Name : {best_model_name}, accuracy_score : {best_model_score}")
                print(f"Best Parameters : {best_model.best_params_}")
                print(f"Best Score : {best_model.best_score_}")
                print('\n====================================================================================\n')
                logging.info(print(f"Best Model Found, Model Name : {best_model_name}, accuracy_score : {best_model_score}"))

            else :
                print(f"Best Model Found, Model Name : {best_model_name}, accuracy_score : {best_model_score}")
                print('\n====================================================================================\n')
                logging.info(print(f"Best Model Found, Model Name : {best_model_name}, accuracy_score : {best_model_score}"))

            save_object(file_path=self.model_trainer_config.trainer_model_file_path,
                        obj=best_model
            )


        except Exception as e:
            logging.info("Exception occured in Model Training")
            raise CustomException(e, sys)