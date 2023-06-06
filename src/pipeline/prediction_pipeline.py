import sys
import os
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # set the preprocessor and model pickle file path
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            # get the preprocessor and model pickle file
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            # transform the data
            data_scaled = preprocessor.transform(features)
            # prediction
            prediction = model.predict(data_scaled)
            return prediction
        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)


# to get the new data for prediction
class CustomData:
    def __init__(self, tv, radio, newspaper):
        self.tv = tv
        self.radio = radio
        self.newspaper = newspaper

    # to convert new data to dataframe
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'TV': [self.tv],
                'Radio': [self.radio],
                'Newspaper': [self.newspaper],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)
