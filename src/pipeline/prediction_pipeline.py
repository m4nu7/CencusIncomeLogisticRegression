import os
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.utils import load_object


class PreditPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Exception occured in Prediction")
            raise CustomException(e, sys)
        
    
class CustomData:
    def __init__(self,
                 age : int,
                 education_num : int,
                 capital_gain : int,
                 capital_loss : int,
                 hours_per_week : int,
                 workclass : str,
                 marital_status : str,
                 occupation : str,
                 relationship : str,
                 race : str,
                 sex : str,
                 native_country : str):
        
        self.age = age
        self.education_num = education_num
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.workclass = workclass
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.native_country = native_country
    

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':self.age,
                'education-num':self.education_num,
                'capital-gain':self.capital_gain,
                'capital-loss':self.capital_loss,
                'hours-per-week':self.hours_per_week,
                'workclass':self.workclass,
                'marital-status':self.marital_status,
                'occupation':self.occupation,
                'relationship':self.relationship,
                'race':self.race,
                'sex':self.sex,
                'native-country':self.native_country 
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame gathered")
            return df
        
        except Exception as e:
            logging.info("Exception occured in prediction pipeline")
            raise CustomException(e, sys)
