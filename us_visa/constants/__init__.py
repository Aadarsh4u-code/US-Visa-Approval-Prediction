import os
from datetime import date

# Database and AWS related constant
# ----------------------------------------------------
DATABASE_NAME: str = "US_VISA"
COLLECTION_NAME: str = "visa_data"
MONGODB_URL_KEY: str = "MONGODB_URL_KEY"

# AWS
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID_ENV_KEY"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY_ENV_KEY"
REGION_NAME = "REGION_NAME"

# File and Directory related constant
# ----------------------------------------------------
PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR: str = "artifact"

# Training and Testing  File and Directory related constant
# ----------------------------------------------------
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# Data and Model File related constant
# ----------------------------------------------------
DATA_FILE_NAME: str = "usvisa.csv"
TARGET_COLUMN = "case_status"
MODEL_FILE_NAME: str = "us_visa_model.pkl"

# Preprocessing and yaml file related constant
# ----------------------------------------------------
CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# Data Ingestion related constant start with DATA_INGESTION VAR NAME
# ----------------------------------------------------
DATA_INGESTION_COLLECTION_NAME: str = "visa_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data Validation realted contant start with DATA_VALIDATION VAR NAME
# ----------------------------------------------------
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME_JSON: str = "report.json"


# Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
# ----------------------------------------------------
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


# MODEL TRAINER related constant start with MODEL_TRAINER var name
# ----------------------------------------------------
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

# LOAD  MODEL YAML CONFIGURATION related constant
GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"


# MODEL EVALUATION related constant 
# ----------------------------------------------------
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "usvisa-model2024-exp-1"
MODEL_PUSHER_S3_KEY = "usvisa-model-registry"

# App Host related constant 
# ----------------------------------------------------
APP_HOST = "0.0.0.0"
APP_PORT = 8080