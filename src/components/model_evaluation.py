import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger.logging import logging
from src.exception.exception import customexception

class ModelEvaluation:
    def __init__(self):
        logging.info("evaluation started")
        
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_squared_error(actual, pred)
        r2 = r2_score(actual, pred)
        logging.info("evaluation metrics captured")
        
        return rmse, mae, r2
    
    def initiate_model_evaluation(self, train_array, teat_array):
        
        try:
            pass
        
        except Exception as e:
            raise customexception(e, sys)
