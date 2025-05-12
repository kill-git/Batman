"""Project pipelines registry."""
from __future__ import annotations

from kedro.pipeline import Pipeline

# Import manuel des pipelines
from batman.pipelines.data_download.pipeline import create_pipeline as create_data_download_pipeline
from batman.pipelines.data_preprocessing.pipeline import create_pipeline as create_data_preprocessing_pipeline
from batman.pipelines.feature_engineering.pipeline import create_pipeline as create_feature_engineering_pipeline
from batman.pipelines.training.pipeline import create_pipeline as create_training_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Manually register all pipelines."""
    data_download = create_data_download_pipeline()
    data_preprocessing = create_data_preprocessing_pipeline()
    feature_engineering = create_feature_engineering_pipeline()
    training = create_training_pipeline()

    return {
        "data_download": data_download,
        "data_preprocessing": data_preprocessing,
        "feature_engineering": feature_engineering,
        "training": training,
        "__default__": (
            data_download
            + data_preprocessing
            + feature_engineering
            + training
        ),
    }