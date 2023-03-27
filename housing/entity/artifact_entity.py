"""It is for the output from the component"""

from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestion", ["train_file_path", "test_file_path", "is_ingested", "message"])