from housing.entity.config_entity import DataIngestionConfig
from housing.exception import Exception_Handling
from housing.logger import logging
import os, sys

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'='*20} Data Ingestion log starte.{'='*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise Exception_Handling(e, sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            pass
        except Exception as e:
            raise Exception_Handling(e, sys)