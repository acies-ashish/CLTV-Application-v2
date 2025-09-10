from kedro.pipeline import Pipeline, node
from .nodes import (
    prepare_kpi_data,
    prepare_segment_summary_data,
    prepare_segment_counts_data,
    prepare_top_products_by_segment_data,
    prepare_predicted_cltv_display_data,
    prepare_cltv_comparison_data,
    calculate_realization_curve_data,
    prepare_churn_summary_data,
    prepare_churn_detailed_view_data,
    # 
    # **- NEW IMPORT -**
    # 
    prepare_churn_performance_data,
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline for preparing all data required for the Streamlit UI.
    """
    return Pipeline(
        [
            node(
                func=prepare_kpi_data,
                inputs=["orders_merged_with_user_id", "final_rfm_cltv_churn_data", "transactions_typed"], # Changed input from historical_cltv_customers
                outputs="kpi_data_for_ui",
                name="prepare_kpi_data",
            ),
            node(
                func=prepare_segment_summary_data,
                inputs="historical_cltv_customers",
                outputs="segment_summary_data_for_ui",
                name="prepare_segment_summary_data",
            ),
            node(
                func=prepare_segment_counts_data,
                inputs="historical_cltv_customers",
                outputs="segment_counts_data_for_ui",
                name="prepare_segment_counts_data",
            ),
            node(
                func=prepare_top_products_by_segment_data,
                inputs=["orders_merged_with_user_id", "transactions_typed", "historical_cltv_customers"],
                outputs="top_products_by_segment_data_for_ui",
                name="prepare_top_products_by_segment_data",
            ),
            node(
                func=prepare_predicted_cltv_display_data,
                inputs=["historical_cltv_customers", "predicted_cltv_df"],
                outputs="predicted_cltv_display_data_for_ui",
                name="prepare_predicted_cltv_display_data",
            ),
            node(
                func=prepare_cltv_comparison_data,
                inputs="predicted_cltv_display_data_for_ui",
                outputs="cltv_comparison_data_for_ui",
                name="prepare_cltv_comparison_data",
            ),
            node(
                func=calculate_realization_curve_data,
                inputs=["orders_merged_with_user_id", "historical_cltv_customers"],
                outputs="realization_curve_data_for_ui",
                name="calculate_realization_curve_data",
            ),
            node(
                func=prepare_churn_summary_data,
                inputs="final_rfm_cltv_churn_data", # This will be the combined customer data
                outputs=["churn_summary_data_for_ui", "active_days_summary_data_for_ui"],
                name="prepare_churn_summary_data",
            ),
            node(
                func=prepare_churn_detailed_view_data,
                inputs="final_rfm_cltv_churn_data", # This will be the combined customer data
                outputs="churn_detailed_view_data_for_ui",
                name="prepare_churn_detailed_view_data",
            ),
            node(
                func=prepare_churn_performance_data,
                inputs=["churn_classification_report", "churn_labels_test", "churn_features_test"],
                outputs="churn_performance_metrics_for_ui",
                name="prepare_churn_performance_data",
            )
        ],
        tags="ui_data_preparation" # Optional: Add a tag for this pipeline
    )