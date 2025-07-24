# src/customer_analytics/pipeline_registry.py

"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

# Import your preprocessing pipeline
from cltv_base.pipeline import create_pipeline as create_preprocessing_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Create an instance of your preprocessing pipeline
    preprocessing_pipeline = create_preprocessing_pipeline()

    # Register the pipeline with a name
    return {
        "preprocessing_pipeline": preprocessing_pipeline,
        "__default__": preprocessing_pipeline, # Optionally set it as the default pipeline
    }

