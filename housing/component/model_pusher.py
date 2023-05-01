from housing.logger import logging
from housing.exception import Exception_Handling
from housing.entity.artifact_entity import ModelPushArtifact, ModelEvaluationArtifact
from housing.entity.config_entity import PushModelConfig
import os, sys
import shutil

class ModelPusher:
    def __init__(self, model_push_config: PushModelConfig, 
                 model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            logging.info(f"{'>>'*20}Model Push log started.{'<<'*20}")
            self.model_push_config = model_push_config
            self.model_evaluation_artifact = model_evaluation_artifact

        except Exception as e:
            raise Exception_Handling(e,sys) from e
        
    
    def export_model(self) -> ModelPushArtifact:
        try:
            evaluated_model_file_path = self.model_evaluation_artifact.evaluated_model_path
            export_dir = self.model_push_config.export_dir_path
            model_file_name = os.path.basename(evaluated_model_file_path)
            export_model_file_path = os.path.join(export_dir, model_file_name)
            logging.info(f"Exporting Model File: [{export_model_file_path}]")
            os.makedirs(export_dir, exist_ok=True)

            shutil.copy(src=evaluated_model_file_path, dst=export_model_file_path)
            logging.info(f"Trained Model: {evaluated_model_file_path} is copied in export dir:[{export_model_file_path}]")
            model_push_artifact = ModelPushArtifact(is_model_push=True, 
                                                    export_model_file_path=export_model_file_path)
            logging.info(f"Model Push Artifact: [{model_push_artifact}]")
            return model_push_artifact
        except Exception as e:
            raise Exception_Handling(e,sys) from e
        
    def initiate_model_pusher(self) -> ModelPushArtifact:
        try:
            return self.export_model()
        except Exception as e:
            raise Exception_Handling(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20} Model Push Log Completed.{'<<'*20}")
        
