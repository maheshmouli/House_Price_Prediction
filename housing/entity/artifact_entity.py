"""It is for the output from the component"""

from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestion", ["train_file_path", "test_file_path", "is_ingested", "message"])

DataValidationArtifact = namedtuple("DataValidation",["schema_file_path","report_file_path", "report_page_file_path","is_validated","message"])

DataTransformationArtifact = namedtuple("DataTransformation",["is_transformed","message","transformed_train_file_path","transformed_test_file_path", "preprocessed_object_file_path"])

ModelTrainerArtifact = namedtuple("ModelTrainer",["is_trained","message","trained_model_filepath","train_rmse","test_rmse","train_accuracy","test_accuracy","model_accuracy"])

ModelEvaluationArtifact = namedtuple("ModelEvaluation",["is_model_accepted", "evaluated_model_path"])

ModelPushArtifact = namedtuple("ModelPush",["is_model_push","export_model_file_path"])
