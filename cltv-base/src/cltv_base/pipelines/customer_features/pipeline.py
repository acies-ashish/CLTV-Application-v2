from kedro.pipeline import Pipeline, node
from .nodes import (
    calculate_customer_level_features,
    perform_rfm_segmentation,
    calculate_historical_cltv
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline for calculating customer-level features,
    performing RFM segmentation, and computing historical CLTV.
    """
    return Pipeline(
        [
            node(
                func=calculate_customer_level_features,
                inputs="transactions_typed",
                outputs="customer_level_features",
                name="calculate_customer_level_features",
            ),
            node(
                func=perform_rfm_segmentation,
                inputs="customer_level_features",
                outputs="rfm_segmented_df",
                name="perform_rfm_segmentation",
            ),
            node(
                func=calculate_historical_cltv,
                inputs="rfm_segmented_df",
                outputs="historical_cltv_customers",
                name="calculate_historical_cltv",
            ),
        ],
        tags="customer_features"
    )