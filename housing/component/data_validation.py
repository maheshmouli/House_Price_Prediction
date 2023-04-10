from housing.logger import logging
from housing.exception import Exception_Handling
from housing.entity.config_entity import DataValidationConfig
from housing.entity.artifact_entity import DataIngestionArtifact
import os,sys

class DataValidation:
    def __init__(self, data_validation_config:DataValidationConfig, data_ingestion_artifact:DataIngestionArtifact) -> None:
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise Exception_Handling(e, sys) from e
        
    def is_train_test_file_exists(self):
        try:
            logging.info("Checking if training and testing file exists")
            is_train_file_exist = False
            is_test_file_exist = False
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exist = os.path.exists(train_file_path)
            is_test_file_exist = os.path.exists(test_file_path)

            is_exists = is_train_file_exist and is_test_file_exist
            logging.info(f"Is Train and Test file exists ? -> {is_exists}")
            return is_exists
        except Exception as e:
            raise Exception_Handling(e, sys) from e
    def initiate_data_validation(self):
        try:
            pass
        except Exception as e:
            raise Exception_Handling(e, sys) from e