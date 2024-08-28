import os
import sys
import pickle

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import accuracy_score, roc_auc_score



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Error while creating Pickle file")
        raise CustomException(e, sys)
    

def evaluate_model(X_train, X_test, y_train, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]

            # Train the model
            model.fit(X_train, y_train)

            # Predict testing data
            y_test_pred = model.predict(X_test)

            # Get the accuary scores for the test data
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        
        return report

    except Exception as e:
        logging.info("Exception occured during Model Training")
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "wb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occured in load_object function utils")
        raise CustomException(e, sys)