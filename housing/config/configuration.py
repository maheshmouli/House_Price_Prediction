from housing.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainConfig,     ModelEvaluationConfig, PushModelConfig, TrainingPipelineConfig
from housing.util.util import read_yaml_file
from housing.constant import *
from housing.exception import Exception_Handling
from housing.logger import logging
import sys,os


class Configuration:
    def __init__(self, config_file_path:str = CONFIG_FILE_PATH, current_time_stamp:str = CURRENT_TIME_STAMP) -> None:
        self.config_info = read_yaml_file(file_path = config_file_path)
        self.training_pipeline_config = self.get_training_pipeline_config()
        self.time_stamp = current_time_stamp

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_ingestion_artifact_dir = os.path.join(
                artifact_dir, 
                DATA_INGESTION_ARTIFACT_DIR,
                self.time_stamp
            )
            data_ingestion_info=self.config_info[DATA_INGESTION_CONFIG_KEY]
            dataset_download_url = data_ingestion_info[DATA_INGESTION_DONWLOAD_URL_KEY]
            tgz_download_dir = os.path.join(data_ingestion_artifact_dir, 
                                            data_ingestion_info[DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY]
                                            )
            raw_data_dir = os.path.join(data_ingestion_artifact_dir,
                                        data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY]
                                        )
            ingested_data_dir = os.path.join(data_ingestion_artifact_dir, 
                                             data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY]
                                             )
            ingested_train_dir = os.path.join(ingested_data_dir, 
                                              data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY]
                                              )
            ingested_test_dir = os.path.join(ingested_data_dir, 
                                             data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY]
                                             )

            data_ingestion_config = DataIngestionConfig(
                dataset_download_url=dataset_download_url, 
                tgz_download_dir=tgz_download_dir, 
                raw_data_dir=raw_data_dir, 
                ingested_train_dir=ingested_train_dir, 
                ingested_test_dir=ingested_test_dir)
            logging.info("Data Ingestion Config: {}".format(data_ingestion_config))
            return data_ingestion_config
        except Exception as e:
            raise Exception_Handling(e, sys) from e
        

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_validation_artifact_dir = os.path.join(
                artifact_dir, DATA_VALIDATION_ARTIFACT_DIR_KEY, self.time_stamp
                )
            data_validation_info = self.config_info[DATA_VALIDATION_CONFIG_KEY]
            schema_file_path = os.path.join(
                ROOT_DIR, data_validation_info[DATA_VALIDATION_SCHEMA_DIR_KEY], data_validation_info[DATA_VALIDATION_SCHEMA_NAME]
                )
            report_file_path = os.path.join(data_validation_artifact_dir, data_validation_info[DATA_VALIDATION_REPORT_FILE_NAME_KEY])
            report_page_file_path = os.path.join(data_validation_artifact_dir, 
                                                 data_validation_info[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY])
            data_validation_config = DataValidationConfig(
                schema_file_path=schema_file_path, report_file_path= report_file_path, report_page_file_path=report_page_file_path
            ) 
            logging.info("Data Validation Config: {}".format(data_validation_config))
            return data_validation_config
        except Exception as e:
            raise Exception_Handling(e, sys) from e

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_transformation_artifact_dir = os.path.join(
                artifact_dir, DATA_TRANSFORMATION_ARTIFACT_DIR_KEY, self.time_stamp
            )
            data_transformation_info = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            add_bedroom_per_room = data_transformation_info[DATA_TRANSFORMATION_ADD_BEDROOM_PER_ROOM_KEY]
            transformed_train_dir = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY],
                data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY]
            )
            transformed_test_dir = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY],
                data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY]
            )
            preprocessed_object_file_path = os.path.join(
                data_transformation_artifact_dir,
                data_transformation_info[DATA_TRANSFORMATION_PREPROCESS_DIR_KEY],
                data_transformation_info[DATA_TRANSFORMATION_PREPROCESS_OBJECT_FILENAME]
            )
            data_transformation_config = DataTransformationConfig(
                add_bedroom_per_room=add_bedroom_per_room,
                transformed_train_dir=transformed_train_dir ,
                transformed_test_dir= transformed_test_dir,
                preprocessed_object_file_path= preprocessed_object_file_path
            )
            logging.info("Data Transformation Config: {}".format(data_transformation_config))
            return data_transformation_config
        except Exception as e:
            raise Exception_Handling(e, sys) from e

    def get_model_train_config(self) -> ModelTrainConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            model_training_artifact_dir = os.path.join(
                artifact_dir,
                MODEL_TRAINING_ARTIFACT_DIR_KEY, self.time_stamp

            )
            model_training_info = self.config_info[MODEL_TRAINING_CONFIG_KEY]
            trained_model_file_path = os.path.join(
                model_training_artifact_dir,
                model_training_info[MODEL_TRAINING_MODEL_DIR_KEY],
                model_training_info[MODEL_TRAINING_MODEL_FILENAME_KEY]
            )

            model_config_file_path = os.path.join(model_training_info[MODEL_TRAINING_MODEL_CONFIG_DIR_KEY],
            model_training_info[MODEL_TRAINING_MODEL_CONFIG_FILE_NAME])
            base_accuracy = model_training_info[MODEL_TRAINING_BASE_ACCURACY_KEY]
            model_training_config = ModelTrainConfig(
                trained_model_file_path= trained_model_file_path,
                base_accuracy= base_accuracy,
                model_config_file_path=model_config_file_path
            )
            logging.info("Model Training Config: {}".format(model_training_config))
            return model_training_config
        except Exception as e:
            raise Exception_Handling(e, sys) from e

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            model_evaluation_artifact_dir = os.path.join(
                artifact_dir,
                MODEL_EVALUATION_ARTIFACT_DIR_KEY
            )
            model_evaluation_info = self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            model_evaluation_filepath = os.path.join(
                model_evaluation_artifact_dir, 
                model_evaluation_info[MODEL_EVALUATION_MODEL_FILENAME_KEY]
            )
            model_evaluation_config = ModelEvaluationConfig(
                model_evaluation_file_path=model_evaluation_filepath,
                time_stamp=self.time_stamp
            )
            logging.info("Model Evaluation Config: {}".format(model_evaluation_config))
            return model_evaluation_config
        except Exception as e:
            raise Exception_Handling(e, sys) from e

    def get_push_model_config(self) -> PushModelConfig :
        try:
            # artifact_dir = self.training_pipeline_config.artifact_dir
            model_push_info = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            model_export_dir = os.path.join(ROOT_DIR, model_push_info[MODEL_PUSHER_MODEL_EXPORT_KEY], self.time_stamp)
            model_pusher_config = PushModelConfig(
                export_dir_path=model_export_dir
            )
            logging.info("Model Push Config: {}".format(model_pusher_config))
            return model_pusher_config
        except Exception as e:
            raise Exception_Handling(e, sys) from e

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(ROOT_DIR, training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                                        training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])
            
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info("Training Pipeline Config: {}".format(training_pipeline_config))
            return training_pipeline_config
        
        except Exception as e:
            raise Exception_Handling(e, sys) from e
        