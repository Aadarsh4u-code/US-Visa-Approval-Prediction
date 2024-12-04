import sys
from typing import Tuple
import numpy as np

from us_visa.exception import CustomException
from us_visa.logger import logging
from us_visa.src.components.model_factory import ModelFactory, ModelTraining
from us_visa.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from us_visa.entity.estimator import USVisaModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model trainer
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using ModelTraining class to train model.")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            model_training = ModelTraining(model_factory=model_factory, base_accuracy=self.model_trainer_config.expected_accuracy)
            
            # Run the training and evaluation pipeline
            best_model_details = model_training.train_and_evaluate(train, test)
    
            # Extract all necessary details from the results
            metric_artifact = ClassificationMetricArtifact(
                accuracy=best_model_details.metrics_after.accuracy,
                f1_score=best_model_details.metrics_after.f1_score,
                precision_score=best_model_details.metrics_after.precision_score,
                recall_score=best_model_details.metrics_after.recall_score,
                roc_auc_score=best_model_details.metrics_after.roc_auc_score)

            return best_model_details, metric_artifact
        
        except Exception as e:
            raise CustomException(e, sys) from e
        
        
    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"{self.data_transformation_artifact.transformed_train_file_path} self.data_transformation_artifact.transformed_train_file_path")
            train_arr = load_numpy_array_data(path_to_npdata=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(path_to_npdata=self.data_transformation_artifact.transformed_test_file_path)
            
            best_model_details ,metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            preprocessing_obj = load_object(path_to_obj=self.data_transformation_artifact.transformed_object_file_path)

            if best_model_details.metrics_after.roc_auc_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            usvisa_model = USVisaModel(preprocessing_object=preprocessing_obj,
                                       trained_model_object=best_model_details.best_model_object)
            logging.info("Created usvisa model object with preprocessor and model")
            logging.info("Created best model file path.")
            save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)

            model_trainer_artifact = ModelTrainerArtifact(
                best_model_name = usvisa_model.__str__(),
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact= metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e




