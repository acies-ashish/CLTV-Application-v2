"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from cltv_base.pipeline import create_pipeline as create_full_project_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Create an instance of the full assembled pipeline
    # This single call now represents your entire data workflow
    full_project_pipeline = create_full_project_pipeline()
    

    # Register the full pipeline with a descriptive name
    return {
        "full_pipeline": full_project_pipeline,
        "__default__": full_project_pipeline, # Set it as the default pipeline
    }

