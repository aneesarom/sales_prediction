# Basic Import
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import lightgbm as ltb
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.utils import save_object
from src.utils.utils import evaluate_model
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
import sys
import os
import numpy as np


@dataclass  # Used when we want to create class without init function
class ModelTrainerConfig:
    # we are defining path for final model pickle file
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        # to create path object
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            # split the train test array
            X_train, y_train, X_test, y_test = train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[
                                                                                                            :, -1]
            logging.info('Split Dependent and Independent variables from train and test data')

            models = {
                "linear_regression": LinearRegression(),
                "catboost": CatBoostRegressor(random_state=123),
                "etr": ExtraTreesRegressor(random_state=123)
            }

            r2_accuracy = []
            trained_models_list = []
            rmse_list = []

            # looping through dictionary, create model and evaluates it
            for model in list(models.values()):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2_score = evaluate_model(X_test, y_test, model)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2_accuracy.append(r2_score)
                rmse_list.append((rmse))
                trained_models_list.append(model)
                print(rmse)

            logging.info("Model Training completed")

            # finding best model based on accuracy
            max_value = max(r2_accuracy)
            max_index = r2_accuracy.index(max_value)
            best_model = trained_models_list[max_index]
            best_model_name = list(models.keys())[max_index]
            best_model_rmse = rmse_list[max_index]
            best_model_accuracy = r2_accuracy[max_index]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best Model pickle file saved")
            print(f"MODEL: {best_model_name} ACCURACY: {best_model_accuracy}, RMSE: {best_model_rmse}")
            return trained_models_list, r2_accuracy

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)
