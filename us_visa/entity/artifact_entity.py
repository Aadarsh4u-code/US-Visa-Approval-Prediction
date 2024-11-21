#******************************************************************
"""
Artifact entity is the output comming from the component and treated as a input for next component in pipeline.
"""
#******************************************************************

from dataclasses import dataclass

# Step 1. Data Ingestion
#------------------------------------------------------------------
@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str 


# Step 2. Data Validation
#------------------------------------------------------------------
@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    drift_report_file_path: str


# Step 3. Data Transformation
#------------------------------------------------------------------
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str


# Step 4. Model Training
#------------------------------------------------------------------
@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetricArtifact


# Step 5. Model Evaluation
#------------------------------------------------------------------
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    s3_model_path:str 
    trained_model_path:str


# Step 6. Model Pusher
#------------------------------------------------------------------
@dataclass
class ModelPusherArtifact:
    bucket_name:str
    s3_model_path:str