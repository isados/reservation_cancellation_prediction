import logging
from typing import Tuple, Optional
import joblib
from pathlib import Path

# from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
from sklearn.preprocessing import Normalizer
import pandas as pd

from steps.utils.data_classes import FeaturesNormalizer, FeaturesEngineeringData
from steps.config import FeatureEngineeringConfig


LOGGER = logging.getLogger(__name__)


class FeatureEngineeringStep:
    """Feature engineering: transform features for model training and inference.
    
    Args:
        inference_mode (bool): Whether the step is used in the training or inference pipeline. 
        feature_engineering_data (FeaturesEngineeringData): Paths relative to the FeatureEngineeringStep
    """

    def __init__(
        self, 
        inference_mode: bool, 
        feature_engineering_data: FeaturesEngineeringData
    ) -> None:
        self.inference_mode = inference_mode
        self.feature_engineering_data = feature_engineering_data

    def __call__(
        self,
        train_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        batch_path: Optional[Path] = None,
    ) -> None:
        """
        Input data paths depending on whether it's training (train, test) or inference (batch)

        Args:
            train_path (Optional[Path], optional): Input train path. Defaults to None.
            test_path (Optional[Path], optional): Input test path. Defaults to None.
            batch_path (Optional[Path], optional): input batch path. Defaults to None.
        """
        if not self.inference_mode:
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
            self.fit_transform(
                df=train_df, 
                output_path=self.feature_engineering_data.train_path
            )
            self.transform(
                df=test_df,
                output_path=self.feature_engineering_data.test_path
            )

        if self.inference_mode:
            batch_df = pd.read_parquet(batch_path)
            self.transform(
                batch_df, 
                output_path=self.feature_engineering_data.batch_path
            )

    def fit_transform(
            self, 
            df: pd.DataFrame, 
            output_path: Path
        ) -> None:
        """Fit encoders on data and store the encoder into the features store
        The processed data is then stored.

        Args:
            df (pd.DataFrame): Data to train encoders and to transform.
            output_path (Path): Data path after encoding.
        """
        LOGGER.info("Start features engineering 'fit_transform'.")
        feature_normalizers = self._init_features_normalizer()
        base_df, target_col = self._get_dfs(
            df=df, 
            features_normalizer=feature_normalizers
        )

        normalized_data = feature_normalizers.normalizer.fit_transform(base_df)


        base_df[feature_normalizers.base_features] = normalized_data

        # Don't forget to add the target
        base_df[feature_normalizers.target] = target_col

        base_df.to_parquet(path=output_path)
        feature_normalizers.to_joblib(path=self.feature_engineering_data.normalizers_path)
        LOGGER.info(
            f"Features and normalizers successfully saved respectively to {str(output_path)} and {str(self.feature_engineering_data.normalizers_path)}"
        )

    def transform(
            self, 
            df: pd.DataFrame, 
            output_path: Path
        ) -> None:
        """Transform data based on trained encoders.

        Args:
            df (pd.DataFrame): Data to transform.
            output_path (Path): Transformed data path.
        """
        LOGGER.info("Start features engineering 'transform'.")
        features_normalizer = self._load_features_normalizer()
        base_df, target_col = self._get_dfs(
            df, features_normalizer=features_normalizer
        )

        normalized_data = features_normalizer.normalizer.transform(base_df)

        base_df[features_normalizer.base_features] = normalized_data

        if target_col is not None:
            # Inference
            base_df[features_normalizer.target] = target_col

        base_df.to_parquet(path=output_path)
        LOGGER.info(f"Features successfully saved to {str(output_path)}")

    def _init_features_normalizer(self) -> FeaturesNormalizer:
        """Init encoders for fit_transform()

        Return:
            feature_encoders (FeatureEncoders): Encoders artifact
        """
        normalizer = Normalizer()
        return FeaturesNormalizer(
            normalizer=normalizer,
            base_features=FeatureEngineeringConfig.base_features,
            target=FeatureEngineeringConfig.target,
        )

    def _load_features_normalizer(self) -> FeaturesNormalizer:
        """Load encoders artifact

        Returns:
            FeaturesEncoder: Encoders artifact
        """
        features_normalizer = joblib.load(self.feature_engineering_data.normalizers_path)
        return features_normalizer

    def _get_dfs(
        self, 
        df: pd.DataFrame, 
        features_normalizer: FeaturesNormalizer
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Extract the relevant columns based on features for respectively: 
        no transformation - ordinal categories - target categories"""
        base_df = df[features_normalizer.base_features]
        if not self.inference_mode:
            target_col = df[features_normalizer.target]
            return base_df, target_col
        elif self.inference_mode:
            return base_df, None
