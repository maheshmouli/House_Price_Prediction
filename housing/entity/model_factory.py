from housing.exception import Exception_Handling
from collections import namedtuple
from typing import List
from housing.logger import logging
from sklearn.metrics import r2_score, mean_squared_error
from cmath import log
import importlib
from pyexpat import model
import numpy as np
import yaml
import os
import sys

GRID_SEARCH_KEY = "grid_search"
CLASS_KEY = "class"
MODULE_KEY = "module"
PARAMETER_KEY = "params"
MODEL_SELECTION_KEY = "model_selection"
SEARCH_PARAM_GRID_KEY = "search_param_grid"

InitializedModelDetail = namedtuple("InitializedModelDetail",["model_serial_number", "model", "param_grid_search", "model_name"])
GridSearchedBestModel = namedtuple("GridSearchedBestModel",["model_serial_number", "model","best_model","best_parameters","best_score"])
BestModel = namedtuple("BestModel",["model_serial_number","model","best_model","best_parameters","best_score"])
MetricInfoArtifact = namedtuple("MetricInfoArtifact",["model_name","model_object","train_rmse","test_rmse","train_accuracy",
                                                      "test_accuracy", "model_accuracy", "index_number"])

def evaluate_regression_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray,
                              base_accuracy:float=0.7) -> MetricInfoArtifact:
    """
    Description:
    It compares multiple regression models & returns a best model

    Params:
    model_list: List of model
    X_train: Training dataset input feature
    y_train: Training dataset target feature
    X_test: Testing dataset input feature
    y_test: Testing dataset input feature

    return
    It retured a named tuple of 
    
    MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])
    """
    try:
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)
            logging.info(f"{'>>'*20}Started Evaluating Model: [{type(model).__name__}] {'<<'*20}")

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy = r2_score(y_train, y_train_pred)
            test_accuracy = r2_score(y_test, y_test_pred)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            model_accuracy = (2 * (train_accuracy * test_accuracy)) / (train_accuracy + test_accuracy)
            diff_test_train_accuracy = abs(test_accuracy - train_accuracy)
            #logging all important metric
            logging.info(f"{'>>'*20} Score {'<<'*20}")
            logging.info(f"Train Score\t\t Test Score\t\t Average Score")
            logging.info(f"{train_accuracy}\t\t {test_accuracy}\t\t{model_accuracy}")

            logging.info(f"{'>>'*20} Loss {'<<'*20}")
            logging.info(f"Diff test train accuracy: [{diff_test_train_accuracy}].") 
            logging.info(f"Train root mean squared error: [{train_rmse}].")
            logging.info(f"Test root mean squared error: [{test_rmse}].")

            if model_accuracy >= base_accuracy and diff_test_train_accuracy < 0.05:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                model_object=model,train_rmse=train_rmse,
                test_rmse=test_rmse,train_accuracy=train_accuracy,
                test_accuracy=test_accuracy, model_accuracy=model_accuracy,
                index_number=index_number)
                logging.info(f"Acceptable Model Found {metric_info_artifact}.")
            index_number += 1
        if metric_info_artifact is None:
            logging.info(f"No Model found with Higher Accuracy than base accuracy")
        return metric_info_artifact
    except Exception as e:
        raise Exception_Handling(e,sys) from e
    

class ModelFactory:
    def __init__(self, model_config_path: str = None):
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)
            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data: dict = self.config[GRID_SEARCH_KEY][PARAMETER_KEY]
            self.models_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list = None
            self.grid_searched_best_model_list = None
        except Exception as e:
            raise Exception_Handling(e, sys) from e
    
    @staticmethod
    def update_property_of_class(instance_ref: object, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("Property_data parameter required to dictionary")
            for key, value in property_data.items():
                logging.info(f"Executing:$ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise Exception_Handling(e, sys) from e
        
    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config: dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise Exception_Handling(e, sys) from e
        
    @staticmethod
    def class_for_name(module_name: str, class_name: str):
        try:
            module = importlib.import_module(module_name)
            logging.info(f"Executing command: from {module} import {class_name}")
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise Exception_Handling(e, sys) from e
        
    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail, input_feature, output_feature) -> GridSearchedBestModel:
        """
        It will perform each parameter search operation, and returns the best optimistic model with best parameter.
        estimator: Model Object
        param_grid: Dictionary of parameter to perform search operation
        input_feature: Input labels
        output_feature: Target/Dependent features

        returns: It returns GridSearchedOperation Object
        """
        try:
            # Instantiating GridSearchCV class
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module, class_name=self.grid_search_class_name)
            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model, param_grid = initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv, self.grid_search_property_data)  

            message = f'{">>"* 20} f"Training {type(initialized_model.model).__name__} Started." {"<<"*20}'
            logging.info(message)
            grid_search_cv.fit(input_feature, output_feature)
            message = f'{">>"* 20} f"Training {type(initialized_model.model).__name__}" completed {"<<"*20}'
            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
            model=initialized_model.model, best_model=grid_search_cv.best_estimator_, best_parameters=grid_search_cv.best_params_,
            best_score=grid_search_cv.best_score_)
            return grid_searched_best_model
        except Exception as e:
            raise Exception_Handling(e,sys) from e
        

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """
        Returns a list of Model details List[ModelDetail]
        """
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():
                model_initialization_config = self.models_initialization_config[model_serial_number]
                model_object_reference = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY],
                                                                     class_name=model_initialization_config[CLASS_KEY])
                model = model_object_reference()
                if PARAMETER_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAMETER_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref= model, property_data=model_obj_property_data)
                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"
                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number, model=model,
                param_grid_search=param_grid_search, model_name=model_name)
                initialized_model_list.append(model_initialization_config)
            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise Exception_Handling(e,sys) from e

    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail, input_feature, output_feature) -> GridSearchedBestModel:
        """
        Calls & returns the execute_grid_search_operation()
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model, input_feature=input_feature, output_feature=output_feature)
        except Exception as e:
            raise Exception_Handling(e,sys) from e
        
    def initiate_best_parameter_search_for_initialized_models(self, initialized_model_list: List[InitializedModelDetail],
    input_feature, output_feature) -> List[GridSearchedBestModel]:
        try:
            self.grid_searched_best_model_list = []
            for initialized_model_list in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model_list,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise Exception_Handling(e,sys) from e
        
    @staticmethod
    def get_best_model_from_grid_search_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel], base_accuracy=0.7) -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable Model Found:{grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score
                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            logging.info(f"Best Model: {best_model}")
            return best_model
        except Exception as e:
            raise Exception_Handling(e,sys) from e

    def get_best_model(self, X, y, base_accuracy = 0.7) -> BestModel:
        try:
            logging.info("Started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized Model: {initialized_model_list}")
            grid_search_best_model_list = self.initiate_best_parameter_search_for_initialized_models(initialized_model_list=initialized_model_list,
            input_feature=X,
            output_feature=y)
            return ModelFactory.get_best_model_from_grid_search_best_model_list(grid_searched_best_model_list=grid_search_best_model_list,base_accuracy=base_accuracy)
        except Exception as e:
            raise Exception_Handling(e, sys) from e