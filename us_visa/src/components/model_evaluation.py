import sys
import pandas as pd
from typing import Optional
from sklearn.metrics import f1_score
from dataclasses import dataclass

from us_visa.logger import logging
from us_visa.exception import CustomException
from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from us_visa.constants import TARGET_COLUMN, CURRENT_YEAR, SCHEMA_FILE_PATH
from us_visa.entity.s3_estimator import USVisaEstimator
from us_visa.entity.estimator import USVisaModel, TargetValueMapping
from us_visa.utils.main_utils import drop_columns, read_yaml_file

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference_in_score: float

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, 
                 data_ingestion_artifact:DataIngestionArtifact, 
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(path_to_yaml=SCHEMA_FILE_PATH)

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def get_best_model(self) -> Optional[USVisaEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try: 
            bucket_name = self.model_evaluation_config.bucket_name
            model_path = self.model_evaluation_config.s3_model_key_path
            usvisa_estimator = USVisaEstimator(bucket_name=bucket_name, model_path=model_path)
            if usvisa_estimator.is_model_present(model_path=model_path):
                return usvisa_estimator
            return None
        except Exception as e:
            raise CustomException(e, sys)
        
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']

            drop_cols = self._schema_config['drop_columns']
            logging.info("drop the columns in drop_cols of Training dataset inside model evaluation")

            test_df = drop_columns(df=test_df, cols = drop_cols)

            X, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y = y.replace(TargetValueMapping()._asdict())

            # trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            best_model_f1_score=None
            best_model = self.get_best_model()

            if best_model is not None:
                y_hat_best_model = best_model.predict(X)
                best_model_f1_score = f1_score(y, y_hat_best_model)
            
            temp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score>temp_best_model_score,
                                           difference_in_score=trained_model_f1_score - temp_best_model_score)
            logging.info(f"Result of Model After Evaluation: {result}")
            return result

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_evaluation_config.s3_model_key_path
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference_in_score
            )
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        
        except Exception as e:
            raise CustomException(e, sys) from e
