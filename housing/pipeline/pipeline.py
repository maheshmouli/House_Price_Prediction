from housing.config.configuration import Configuration
from housing.logger import logging
from housing.exception import Exception_Handling
from housing.entity.config_entity import *
from housing.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPushArtifact
from housing.entity.config_entity import DataIngestionConfig
from housing.component.data_ingestion import DataIngestion
from housing.component.data_validation import DataValidation
from housing.component.data_transformation import DataTransformation
from housing.component.model_evaluation import ModelEvaluation
from housing.component.model_pusher import ModelPusher
from housing.component.model_training import ModelTrainer
import os, sys
from threading import Thread
from collections import namedtuple


class Pipeline(Thread):
    experiment: Experiment = Experiment(*([None]*11))
    experiment_file_path = None

    def __init__(self, config: Configuration = Configuration()) -> None:
        try:
            super().__init__(daemon=False, name="pipeline")
            self.config = config 
        except Exception as e:
            raise Exception_Handling(e, sys) from e
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        
        except Exception as e:
            raise Exception_Handling(e, sys) from e
        
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(), data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise Exception_Handling(e,sys) from e
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, 
                                  data_validation_artifact: DataValidationArtifact)-> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise Exception_Handling(e,sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_train_config(), data_transformation_artifact=data_transformation_artifact)
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise Exception_Handling(e,sys) from e

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                                data_validation_artifact: DataValidationArtifact,
                                model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        try:
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.config.get_model_evaluation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise Exception_Handling(e,sys) from e

    def start_model_push(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPushArtifact:
        try:
            model_push = ModelPusher(
                model_push_config=self.config.get_push_model_config(),
                model_evaluation_artifact=model_evaluation_artifact
            )
            return model_push.initiate_model_pusher()
        except Exception as e:
            raise Exception_Handling(e,sys) from e
    
    def run_pipeline(self):
        try:
            
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                                          data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact, data_validation_artifact=data_validation_artifact, model_trainer_artifact=model_trainer_artifact)
            if model_evaluation_artifact.is_model_accepted:
                model_push_artifact = self.start_model_push(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f"Model Push Artifact: {model_push_artifact}")
            else:
                logging.info("Trained Model Rejected.")
            
        except Exception as e:
            raise Exception_Handling(e, sys) from e
        
    def run(self):
        """
        It will be called by start function
        """
        try:
            self.run_pipeline()
        except Exception as e:
            raise Exception_Handling(e,sys) from e