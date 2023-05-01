from housing.logger import logging
from housing.exception import Exception_Handling
from housing.entity.config_entity import ModelEvaluationConfig
from housing.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from housing.util.util import write_yaml_file, load_data, load_object, read_yaml_file
from housing.entity.model_factory import evaluate_regression_model
from housing.constant import *
import numpy as np
import os
import sys

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20}Model Evaluation log started.{'<<'*20}")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise Exception_Handling(e,sys) from e
        
    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path
            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path)
                return model
            model_evaluation_file_content = read_yaml_file(file_path=model_evaluation_file_path)
            model_evaluation_file_content = dict() if model_evaluation_file_content is None else model_evaluation_file_content
            if BEST_MODEL_KEY not in model_evaluation_file_content:
                return model
            
            model = load_object(file_path=model_evaluation_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise Exception_Handling(e,sys) from e
        
    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_evaluation_content = read_yaml_file(file_path=evaluation_file_path)
            model_evaluation_content = dict() if model_evaluation_content is None else model_evaluation_content

            previous_best_model = None
            if BEST_MODEL_KEY in model_evaluation_content:
                previous_best_model = model_evaluation_content[BEST_MODEL_KEY]
            logging.info(f"Previous Evaluation Result: {model_evaluation_content}")
            evaluation_result = {
                BEST_MODEL_KEY:{
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path
                }
            }

            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_evaluation_content:
                    history = {HISTORY_KEY: model_history}
                    evaluation_result.update(history)
                else:
                    model_evaluation_content[HISTORY_KEY].update(model_history)
            
            model_evaluation_content.update(evaluation_result)
            logging.info(f"Updated Evaluation Result: {model_evaluation_content}")
            write_yaml_file(file_path=evaluation_file_path, data=model_evaluation_content)
        except Exception as e:
            raise Exception_Handling(e,sys) from e
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_filepath
            trained_model_object = load_object(file_path=trained_model_file_path)
            
            # Getting the Train & Test File path
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Gathering the Schema File Path
            schema_file_path = self.data_validation_artifact.schema_file_path

            # Loading the Train & Test datasets
            train_dataframe = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            test_dataframe = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema_content = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema_content[TARGET_COLUMN_KEY]

            # Target Column
            logging.info(f"Converting target column into numpy array.")
            train_target_array = np.array(train_dataframe[target_column_name])
            test_target_array = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # Drop target column 
            logging.info(f"Dropping target column from the dataframe")
            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")

            model = self.get_best_model()

            if model is None:
                logging.info("No Existing Model found. Accepting the trained Model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path, is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model Accepted. Model Evaluation Artifact {model_evaluation_artifact} created.")
                return model_evaluation_artifact
            
            model_list = [model, trained_model_object]

            metric_info_artifact = evaluate_regression_model(
                model_list=model_list, 
                X_train=train_dataframe, 
                y_train=train_target_array,
                X_test=test_dataframe, 
                y_test=test_target_array,
                base_accuracy=self.model_trainer_artifact.model_accuracy
                )
            logging.info(f"Model Evaluation Completed. Model Metric Artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False, evaluated_model_path=trained_model_file_path)
                logging.info(response)
                return response
            
            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path, is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model Accepted. Model Evaluation Artifact {model_evaluation_artifact} created.")
            else:
                logging.info("Trained Model is no better than existing Model. Hence Not accepting trained Model.")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path, is_model_accepted=False)
            
            return model_evaluation_artifact
        except Exception as e:
            raise Exception_Handling(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'='*20} Model Evaluation Log completed.{'='*20}")
