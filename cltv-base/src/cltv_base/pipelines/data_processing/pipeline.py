from kedro.pipeline import Pipeline, node
from .nodes import (
    standardize_columns, 
    convert_data_types, 
    merge_orders_transactions, 
    aggregate_behavioral_customer_level, 
    aggregate_orders_transactions_customer_level, 
    merge_customer_behavioral_data

)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline for initial data processing, including column standardization,
    data type conversion, and merging orders with user IDs from transactions.
    """
    return Pipeline(
        [
            node(
                func=standardize_columns,
                inputs=["orders_raw", "params:expected_orders_cols", "params:orders_df_name"],
                outputs="orders_standardized",
                name="standardize_orders_columns",
            ),
            node(
                func=standardize_columns,
                inputs=["transactions_raw", "params:expected_transaction_cols", "params:transactions_df_name"],
                outputs="transactions_standardized",
                name="standardize_transactions_columns",
            ),
            node(
                func=standardize_columns,
                inputs=["behavioral_raw", "params:expected_behavioral_cols", "params:behavioral_df_name"],
                outputs="behavioral_standardized",
                name="standardize_behavioral_columns",
            ),
            node(
                func=convert_data_types,
                inputs=["orders_standardized", "transactions_standardized","behavioral_standardized"],
                outputs=["orders_typed", "transactions_typed","behavioral_typed"],
                name="convert_raw_data_types",
            ),
            node(
                func=merge_orders_transactions,
                inputs=["orders_typed", "transactions_typed"],
                outputs="orders_merged_with_user_id",
                name="merge_orders_and_transactions",
            ),
            node(
                func=aggregate_behavioral_customer_level,
                inputs=["behavioral_typed"],
                outputs="customer_aggregated_behavioral_data",
                name="customer_aggregated_behavioral_data_node"
            ),
            node(
                func=aggregate_orders_transactions_customer_level,
                inputs=["orders_merged_with_user_id"],
                outputs="customer_aggregated_orders_transaction_data",
                name="customer_aggregated_orders_transaction_node"
            ),
            node(
                func=merge_customer_behavioral_data,
                inputs=["customer_aggregated_orders_transaction_data", "customer_aggregated_behavioral_data"],
                outputs="customer_level_merged_data",
                name="customer_level_merged_data_node"
            ),
        ],
        tags="data_processing" # Optional: Add a tag for this pipeline
    )

