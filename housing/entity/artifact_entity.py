"""It is for the output from the component"""

from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestion", ["train_file_path", "test_file_path", "is_ingested", "message"])

DataValidationArtifact = namedtuple("DataValidation",["schema_file_path","report_file_path", "report_page_file_path","is_validated","message"])