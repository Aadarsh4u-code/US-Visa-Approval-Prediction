#******************************************************************
"""
Artifact entity is the output comming from the component and treated as a input for next component in pipeline.
"""
#******************************************************************

from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

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
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    message: str
    drift_report_file_path: str
    drift_report_json_file_path: str


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
    accuracy: float
    f1_score:float
    precision_score:float
    recall_score:float
    roc_auc_score: float

@dataclass
class ModelEvaluationResult:
    best_model_name: Optional[str] = None
    best_params: Optional[Dict[str, Any]] = None
    metrics_before: Optional[Dict[str, float]] = None
    metrics_after: Optional[Dict[str, float]] = None
    models_list: List[str] = field(default_factory=list)
    accuracy_list: List[float] = field(default_factory=list)
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    f1_score: List[float] = field(default_factory=list)
    roc_auc_score: List[float] = field(default_factory=list)
    best_model_object: Any = None
    best_score: float= None

@dataclass
class ModelTrainerArtifact:
    best_model_name: str
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