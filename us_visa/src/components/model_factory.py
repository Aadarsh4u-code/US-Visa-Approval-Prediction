import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import importlib

from us_visa.constants import *
from us_visa.entity.artifact_entity import ModelEvaluationResult, ClassificationMetricArtifact
from us_visa.utils.main_utils import load_numpy_array_data, read_yaml_file, write_yaml_file
from us_visa.logger import logging
from us_visa.exception import CustomException

class ModelFactory:
    def __init__(self, model_config_path: str = None):
        """
        model_config_path: model yaml configuration file
        """
        logging.info(f"{self.__class__.__name__} initiated.")
        try:
            self.config: dict = read_yaml_file(model_config_path)

            self.grid_search_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_params: dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])
            
            # Models listed with keys like module_0, module_1, etc.
            self.models_config: dict = dict(self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list = None
            self.grid_searched_best_model_list = None
            logging.info(f"Model attribute from YAML file loaded.")

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def get_model(self, model_name):
        model_config = self.models_config[model_name]
        module = importlib.import_module(model_config[MODULE_KEY])
        model_class = getattr(module, model_config[CLASS_KEY])

        # Validate solver and penalty
        params = model_config['params']
        if model_class == 'LogisticRegression':
            solver = params.get('solver', 'lbfgs')
            penalty = params.get('penalty', 'l2')
            if solver == 'lbfgs' and penalty == 'l1':
                raise ValueError("Solver 'lbfgs' supports only 'l2' or None penalties. Fix your configuration.")

        model = model_class(**params)
        return model
    
    def grid_search_cv(self, model, param_grid, X_train, y_train):
        logging.info(f"inside grid_search_cv method.")
        # Check if the model is LogisticRegression
        from sklearn.linear_model import LogisticRegression

        # Convert 'null' to None in the parameter grid
        param_grid = {k: [(None if v == 'null' else v) for v in vals] for k, vals in param_grid.items()}

        # Handle solver and penalty compatibility for LogisticRegression
        if isinstance( model, LogisticRegression):
            if 'solver' in param_grid and 'penalty' in param_grid:
                param_grid['penalty'] = [
                    p for p in param_grid['penalty'] if (p == 'l2' or p is None)
                ]
                param_grid['solver'] = [
                    s for s in param_grid['solver'] if s in ['lbfgs', 'sag', 'newton-cg']
                ]

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, **self.grid_search_params)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def evaluate_classification(self, y_train, y_pred):
        logging.info("Inside evaluate_classification method to calculate classification matrices.")
        return ClassificationMetricArtifact(
            accuracy=accuracy_score(y_train, y_pred),
            f1_score=f1_score(y_train, y_pred),
            precision_score=precision_score(y_train, y_pred),
            recall_score=recall_score(y_train, y_pred),
            roc_auc_score=roc_auc_score(y_train, y_pred)
        )
    
    def get_best_model(self, X_train, y_train, X_test, y_test, base_accuracy):
        logging.info("Inside get_best_model method to evaluate best model.")
        best_model = None
        best_roc_auc = 0 
        models_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        roc_auc_score_list = []

        for model_key, model_config in self.models_config.items():
            model_class_name = model_config['class']
            model = self.get_model(model_key)
            param_grid = model_config[SEARCH_PARAM_GRID_KEY]
            best_model_estimator, best_params, best_score = self.grid_search_cv(model, param_grid, X_train, y_train)

            # Train and evaluate model before tuning
            model_before_tuning = model.fit(X_train, y_train)
            y_train_pred_before = model_before_tuning.predict(X_train)
            y_test_pred_before = model_before_tuning.predict(X_test)

            train_metrics_before = self.evaluate_classification(y_train, y_train_pred_before)
            test_metrics_before = self.evaluate_classification(y_test, y_test_pred_before)

            # Train and evaluate the best model after tuning
            best_model_estimator.fit(X_train, y_train)  # Re-train with the best hyperparameters
            y_train_pred_after = best_model_estimator.predict(X_train)
            y_test_pred_after = best_model_estimator.predict(X_test)

            train_metrics_after = self.evaluate_classification(y_train, y_train_pred_after)
            test_metrics_after = self.evaluate_classification(y_test, y_test_pred_after)

            # Log model details and metrics
            models_list.append(model_class_name)
            accuracy_list.append(test_metrics_after.accuracy)
            precision_list.append(test_metrics_after.precision_score)
            recall_list.append(test_metrics_after.recall_score)
            f1_score_list.append(test_metrics_after.f1_score)
            roc_auc_score_list.append(test_metrics_after.roc_auc_score)

            print('--------------------------------------------------------------')

            print(f'{model_class_name} performance for Testing set before Hyperparameter Tunning')
            print("- Accuracy: {:.4f}".format(test_metrics_before.accuracy))
            print('- F1 score: {:.4f}'.format(test_metrics_before.f1_score)) 
            print('- Precision: {:.4f}'.format(test_metrics_before.precision_score))
            print('- Recall: {:.4f}'.format(test_metrics_before.recall_score))
            print('- Roc Auc Score: {:.4f}'.format(test_metrics_before.roc_auc_score))
            print('\n')

            print(f'{model_class_name} performance for Testing set After Hyperparameter Tunning')
            print("- Accuracy: {:.4f}".format(test_metrics_after.accuracy))
            print('- F1 score: {:.4f}'.format(test_metrics_after.f1_score)) 
            print('- Precision: {:.4f}'.format(test_metrics_after.precision_score))
            print('- Recall: {:.4f}'.format(test_metrics_after.recall_score))
            print('- Roc Auc Score: {:.4f}'.format(test_metrics_after.roc_auc_score))
            print('='*50)
            print('\n')
        

            # If it's the best model based on AUC, store the model details
            if test_metrics_after.roc_auc_score > best_roc_auc:
                best_roc_auc = test_metrics_after.roc_auc_score
            # if best_score > base_accuracy:
                best_model = {
                    'model_name': model_class_name,
                    'best_model': best_model_estimator,
                    'best_params': best_params,
                    'metrics_before': test_metrics_before,
                    'metrics_after': test_metrics_after,
                    'best_score': best_score
                }

        return best_model, models_list, accuracy_list, precision_list, recall_list, f1_score_list, roc_auc_score_list


class ModelTraining:
    def __init__(self, model_factory: ModelFactory, base_accuracy: float):
        self.model_factory = model_factory
        self.base_accuracy = base_accuracy

    def train_and_evaluate(self, train, test):
        logging.info("Inside train_and_evaluate method.")
        # Split the data into train and test sets
        X_train, y_train, X_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

        # Get the best model from the ModelFactory
        best_model, models_list, accuracy_list, precision, recall, f1_score, roc_auc_score = self.model_factory.get_best_model(
            X_train, y_train, X_test, y_test, self.base_accuracy
        )

        # Update YAML with best model details
        best_model_name = best_model['model_name']
        best_params = best_model['best_params']
        best_model_obj = best_model['best_model']
        metrics_before = best_model['metrics_before']
        metrics_after = best_model['metrics_after']

        # Return the results
        return ModelEvaluationResult(
            best_model_name=best_model_name,
            best_model_object= best_model_obj,
            best_params=best_params,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            models_list=models_list,
            accuracy_list=accuracy_list,
            precision= precision,
            recall=recall,
            f1_score=f1_score,
            roc_auc_score=roc_auc_score,
            best_score = best_model['best_model']
        )
    

# # Define the path to the YAML file
# yaml_path = "/Users/aadarsh/Desktop/Data Scientist/Projects/US-Visa-Approval-Prediction/config/model.yaml"

# # Initialize ModelFactory with the YAML file path
# model_factory = ModelFactory(model_config_path=yaml_path)

# # Initialize ModelTrainingPipeline
# pipeline = ModelTraining(model_factory=model_factory, base_accuracy = 0.6)

# train_arr = load_numpy_array_data(path_to_npdata='/Users/aadarsh/Desktop/Data Scientist/Projects/US-Visa-Approval-Prediction/artifact/02_12_2024_22_21_05/data_transformation/transformed/train.npy')
# test_arr = load_numpy_array_data(path_to_npdata='/Users/aadarsh/Desktop/Data Scientist/Projects/US-Visa-Approval-Prediction/artifact/02_12_2024_22_21_05/data_transformation/transformed/test.npy')

# # Run the training and evaluation pipeline
# best_model_results = pipeline.train_and_evaluate(train_arr, test_arr)

# # Output the best model results
# print("Best Model Results:")
# print(best_model_results)