import os
from pathlib import Path


REPO_DIR = Path(os.path.realpath(""))
INFERENCE_DATA_PATH = REPO_DIR / "data/sample_for_inference.parquet"
TRAINING_DATA_PATH = REPO_DIR / "data/dry_bean.parquet"


class PreprocessConfig:
    train_path = REPO_DIR / "data/preprocessed/train.parquet"
    test_path = REPO_DIR / "data/preprocessed/test.parquet"
    batch_path = REPO_DIR / "data/preprocessed/batch.parquet"

class TrainerConfig:
    model_name ="gradient-boosting"
    random_state = 42
    train_size = 0.2
    shuffle = True
    params = {
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
    metric_params = {
        'average': 'macro',
    }

class ConditionConfig:
    criteria = 0.05
    metric = "roc_auc"

class MlFlowConfig:
    uri = "http://0.0.0.0:8000"
    experiment_name = "dry_bean_classifier"
    artifact_path = "model-artifact"
    registered_model_name = "dry_bean_classifier"

class FeatureEngineeringConfig:
    train_path = REPO_DIR / "data/features_store/train.parquet"
    test_path = REPO_DIR / "data/features_store/test.parquet"
    batch_path = REPO_DIR / "data/features_store/batch.parquet"
    normalizers_path = REPO_DIR / "artifacts/normalizers.joblib"
    base_features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
       'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent',
       'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
       'ShapeFactor3', 'ShapeFactor4'
       ]
    # ordinal_features = [
    #     "arrival_date_month",
    #     "meal",
    #     "market_segment",
    #     "distribution_channel",
    #     "reserved_room_type",
    #     "assigned_room_type",
    #     "customer_type"
    # ]
    # target_features = [
    #     "country",
    #     "booking_changes",
    #     "agent",
    #     "company"
    # ]
    target = "Class"