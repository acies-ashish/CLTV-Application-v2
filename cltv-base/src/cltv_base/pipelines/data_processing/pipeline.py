from kedro.pipeline import Pipeline, node
from .nodes import standardize_columns, convert_data_types, merge_orders_transactions

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
                func=convert_data_types,
                inputs=["orders_standardized", "transactions_standardized"],
                outputs=["orders_typed", "transactions_typed"],
                name="convert_raw_data_types",
            ),
            node(
                func=merge_orders_transactions,
                inputs=["orders_typed", "transactions_typed"],
                outputs="orders_merged_with_user_id",
                name="merge_orders_and_transactions",
            ),
        ],
        tags="data_processing" # Optional: Add a tag for this pipeline
    )

