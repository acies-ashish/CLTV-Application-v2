# src/cltv_base/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    standardize_columns,
    convert_data_types,
    merge_orders_transactions
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a data pipeline for initial data preprocessing.
    """
    return pipeline(
        [
            # 1. Data Standardization
            node(
                func=standardize_columns,
                inputs=["orders_raw", "params:mapping_orders_expected_cols", "params:df_name_orders"], # Referencing the new parameter
                outputs="orders_standardized",
                name="standardize_orders_columns",
            ),
            node(
                func=standardize_columns,
                inputs=["transactions_raw", "params:mapping_transactions_expected_cols", "params:df_name_transactions"], # Referencing the new parameter
                outputs="transactions_standardized",
                name="standardize_transactions_columns",
            ),

            # 2. Data Type Conversion
            node(
                func=convert_data_types,
                inputs=["orders_standardized", "transactions_standardized"],
                outputs=["orders_typed", "transactions_typed"],
                name="convert_data_types",
            ),

            # 3. Merge Orders with User ID from Transactions
            node(
                func=merge_orders_transactions,
                inputs=["orders_typed", "transactions_typed"],
                outputs="orders_merged_with_user_id",
                name="merge_orders_with_user_id",
            ),
        ],
        inputs=["orders_raw", "transactions_raw"],
        outputs=["orders_merged_with_user_id", "transactions_typed"],
    )
