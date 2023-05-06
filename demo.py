from housing.pipeline.pipeline import Pipeline
from housing.exception import Exception_Handling
from housing.logger import logging
from housing.config.configuration import Configuration
from housing.component.data_transformation import DataTransformation
import os

def main():
    try:
        config_file_path = os.path.join("config", "config.yaml")
        pipeline = Pipeline(Configuration(config_file_path=config_file_path))
        pipeline.start()
        logging.info("Main Execution Completed")
    except Exception as e:
        logging.error(f"{e}")
        print(e)
        
if __name__=="__main__":
    main()
