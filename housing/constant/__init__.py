import os
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

ROOT_DIR = os.getcwd() # Get Current Working Directory

CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE_NAME)

CURRENT_TIME_STAMP = get_current_time_stamp()

# Training Pipeline related Variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"

# Data Ingestion variables
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_DONWLOAD_URL_KEY = "dataset_download_url"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY = "tgz_download_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"

# Data Validation variables
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_ARTIFACT_DIR_KEY = "data_validation"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_SCHEMA_NAME = "schema_file_name"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"

# Data Transformation variables
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_ARTIFACT_DIR_KEY = "data_transformation"
DATA_TRANSFORMATION_ADD_BEDROOM_PER_ROOM_KEY = "add_bedroom_per_room"
DATA_TRANSFORMATION_TRANSFORMED_DIR_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESS_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESS_OBJECT_FILENAME = "preprocessed_object_file_name"

COLUMN_TOTAL_ROOMS = "total_rooms"
COLUMN_POPULATION = "population"
COLUMN_HOUSEHOLDS = "households"
COLUMN_TOTAL_BEDROOM = "total_bedrooms"
DATASET_SCHEMA_COLUMNS_KEY = "columns"

NUMERICAL_COLUMN_KEY = "numerical_columns"
CATEGORICAL_COLUMN_KEY = "categorical_columns"

TARGET_COLUMN_KEY = "target_column"

# Model Train variables
MODEL_TRAINING_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINING_ARTIFACT_DIR_KEY = "model_trainer"
MODEL_TRAINING_MODEL_DIR_KEY = "trained_model_dir"
MODEL_TRAINING_MODEL_FILENAME_KEY = "model_file_name"
MODEL_TRAINING_BASE_ACCURACY_KEY = "base_accuracy"
MODEL_TRAINING_MODEL_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINING_MODEL_CONFIG_FILE_NAME = "model_config_file_name"

# Model Evaluation Variables
MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
MODEL_EVALUATION_ARTIFACT_DIR_KEY = "model_evaluation"
MODEL_EVALUATION_MODEL_DIR_KEY = "model_evaluation_dir"
MODEL_EVALUATION_MODEL_FILENAME_KEY = "model_evaluation_file_name"

# Best Model Variables
BEST_MODEL_KEY = "best_model"
HISTORY_KEY = "history"
MODEL_PATH_KEY = "model_path"

# Model Pusher Variables
MODEL_PUSHER_CONFIG_KEY = "push_model_config"
MODEL_PUSHER_MODEL_EXPORT_KEY = "model_export_dir"

# Experiment Variables
EXPERIMENT_DIR_NAME = "experiment"
EXPERIMENT_FILE_NAME = "experiment.csv"