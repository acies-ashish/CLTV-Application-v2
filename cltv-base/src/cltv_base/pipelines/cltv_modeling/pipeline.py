# src/cltv_base/pipelines/cltv_modeling/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import predict_cltv_bgf_ggf

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline for CLTV modeling using BG/NBD and Gamma-Gamma models.
    """
    return Pipeline(
        [
            node(
                func=predict_cltv_bgf_ggf,
                inputs="transactions_typed", # Input from data_processing pipeline
                outputs="predicted_cltv_df",
                name="predict_cltv_bgf_ggf",
            ),
        ],
        tags="cltv_modeling" # Optional: Add a tag for this pipeline
    )

