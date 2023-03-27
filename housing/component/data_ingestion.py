from housing.entity.config_entity import DataIngestionConfig
from housing.exception import Exception_Handling
from housing.logger import logging
from housing.entity.artifact_entity import DataIngestionArtifact
import tarfile
from six.moves import urllib
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'='*20} Data Ingestion log starte.{'='*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise Exception_Handling(e, sys) # type: ignore
        
    def download_housing_data(self) -> str:
        try:
            # Extract remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url

            # Folder location to download file
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            
            # Checks whether folder exists, if not create
            if os.path.exists(tgz_download_dir):
                os.remove(tgz_download_dir)
            os.makedirs(tgz_download_dir, exist_ok=True)

            housing_file_name = os.path.basename(download_url)
            tgz_filepath  = os.path.join(tgz_download_dir, housing_file_name)

            logging.info("Downloading File from: [{}] into :[{}]".format(download_url, tgz_filepath))
            urllib.request.urlretrieve(download_url, tgz_filepath)
            logging.info("File :[{}] has been Donwloaded Successfully".format(tgz_filepath))

            return tgz_filepath
        except Exception as e:
            raise Exception_Handling(e, sys) from e


    def extract_tgz_file(self, tgz_file_path):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
            os.makedirs(raw_data_dir, exist_ok=True)
            logging.info("Extracting tgz file:[{}] into dir:[{}]".format(tgz_file_path, raw_data_dir))
            with tarfile.open(tgz_file_path) as housing_tgz_file_obj:
                housing_tgz_file_obj.extractall(path = raw_data_dir)
            logging.info("Extraction Completed")

        except Exception as e:
            raise Exception_Handling(e, sys) from e

    def split_data_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            file_name = os.listdir(raw_data_dir)[0]
            
            housing_file_path = os.path.join(raw_data_dir, file_name)

            logging.info(f"Reading CSV File: [{housing_file_path}]")
            housing_df = pd.read_csv(housing_file_path)
            # For splitting the data equally
            housing_df['income_cat'] = pd.cut(
                housing_df['median_income'],
                bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                labels=[1,2,3,4,5]
            )

            logging.info(f"Splitting data into train and test")
            strat_train_set = None
            strat_test_set = None
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in split.split(housing_df, housing_df["income_cat"]):
                strat_train_set = housing_df.loc[train_index].drop(["income_cat"],axis=1)
                strat_test_set = housing_df.loc[test_index].drop(["incomce_cat"],axis=1)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, file_name)

            # Saving the data in CSV File Format
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                logging.info(f"Exporting training dataset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path, index=False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path, index=False)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                  test_file_path=test_file_path,
                                  is_ingested=True,
                                  message=f"Data Ingestion Completed Successfully")
            logging.info(f"Data Ingestion Artifact: [{data_ingestion_artifact}]")
            return data_ingestion_artifact
        
        except Exception as e:
            raise Exception_Handling(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            tgz_file_path = self.download_housing_data()
            self.extract_tgz_file(tgz_file_path=tgz_file_path)
            return self.split_data_train_test()
        except Exception as e:
            raise Exception_Handling(e, sys) from e
        
    def __del__(self):
        logging.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n\n")