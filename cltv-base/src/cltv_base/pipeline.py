# src/cltv_base/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    standardize_columns,
    convert_data_types,
    merge_orders_transactions,
    calculate_customer_level_features,
    perform_rfm_segmentation,
    calculate_historical_cltv,
    get_customers_at_risk,
    label_churned_customers,
    get_churn_features_labels,
    prepare_survival_data,
    predict_cltv_bgf_ggf,
    train_churn_prediction_model,
    predict_churn_probabilities,
    assign_predicted_churn_labels,
    train_cox_survival_model,
    # New UI data preparation nodes
    prepare_kpi_data,
    prepare_segment_summary_data,
    prepare_segment_counts_data, # NEW NODE IMPORT
    prepare_top_products_by_segment_data,
    prepare_predicted_cltv_display_data,
    prepare_cltv_comparison_data,
    calculate_realization_curve_data,
    prepare_churn_summary_data,
    prepare_churn_detailed_view_data
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a data pipeline for initial data preprocessing, core operations, and UI data preparation.
    """
    return pipeline(
        [
            # 1. Data Ingestion and Standardization
            node(
                func=standardize_columns,
                inputs=["orders_raw", "params:mapping_orders_expected_cols", "params:df_name_orders"],
                outputs="orders_standardized",
                name="standardize_orders_columns",
            ),
            node(
                func=standardize_columns,
                inputs=["transactions_raw", "params:mapping_transactions_expected_cols", "params:df_name_transactions"],
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

            # 3. Merge Orders with User ID from Transactions (used for Top Products and Realization Curve)
            node(
                func=merge_orders_transactions,
                inputs=["orders_typed", "transactions_typed"],
                outputs="orders_merged_with_user_id",
                name="merge_orders_with_user_id",
            ),

            # --- Core Operations Nodes ---
            # 4. Calculate Customer Level Features
            node(
                func=calculate_customer_level_features,
                inputs="transactions_typed",
                outputs="customer_level_features",
                name="calculate_customer_level_features",
            ),

            # 5. Perform RFM Segmentation
            node(
                func=perform_rfm_segmentation,
                inputs="customer_level_features",
                outputs="rfm_segmented_df",
                name="perform_rfm_segmentation",
            ),

            # 6. Calculate Historical CLTV
            node(
                func=calculate_historical_cltv,
                inputs="rfm_segmented_df",
                outputs="rfm_segmented_with_historical_cltv",
                name="calculate_historical_cltv",
            ),
            
            # 7. Identify Customers at Risk
            node(
                func=get_customers_at_risk,
                inputs=["rfm_segmented_with_historical_cltv", "params:at_risk_threshold_days"],
                outputs="customers_at_risk_df",
                name="identify_customers_at_risk",
            ),

            # 8. CLTV Prediction (BG/NBD + Gamma-Gamma)
            node(
                func=predict_cltv_bgf_ggf,
                inputs="transactions_typed",
                outputs="predicted_cltv_df",
                name="predict_cltv_bgf_ggf",
            ),

            # 9. Churn Modeling (Random Forest)
            node(
                func=label_churned_customers,
                inputs=["rfm_segmented_with_historical_cltv", "params:churn_inactive_days_threshold"],
                outputs="rfm_segmented_labeled_churn",
                name="label_churned_customers",
            ),
            node(
                func=get_churn_features_labels,
                inputs="rfm_segmented_labeled_churn",
                outputs=["churn_features_X", "churn_labels_y"],
                name="get_churn_features_labels",
            ),
            node(
                func=train_churn_prediction_model,
                inputs=[
                    "churn_features_X",
                    "churn_labels_y",
                    "params:churn_model_n_estimators",
                    "params:churn_model_random_state",
                ],
                outputs=["churn_model", "churn_report", "churn_feature_importances", "churn_X_test", "churn_y_test"],
                name="train_churn_prediction_model",
            ),
            node(
                func=predict_churn_probabilities,
                inputs=["churn_model", "churn_features_X"], # Use the full X for prediction on all customers
                outputs="predicted_churn_probabilities",
                name="predict_churn_probabilities",
            ),
            node(
                func=assign_predicted_churn_labels,
                inputs=["predicted_churn_probabilities", "params:predicted_churn_probability_threshold"],
                outputs="predicted_churn_labels",
                name="assign_predicted_churn_labels",
            ),

            # 10. Survival Analysis (Cox Model)
            node(
                func=prepare_survival_data,
                inputs=["rfm_segmented_labeled_churn", "params:cox_model_churn_threshold"],
                outputs="rfm_segmented_survival_data",
                name="prepare_survival_data",
            ),
            node(
                func=train_cox_survival_model,
                inputs=["rfm_segmented_survival_data", "params:cox_feature_cols"],
                outputs=["cox_model", "rfm_segmented_with_cox_predictions"],
                name="train_cox_survival_model",
            ),

            # --- UI Data Preparation Nodes ---
            node(
                func=prepare_kpi_data,
                inputs=["orders_merged_with_user_id", "rfm_segmented_with_historical_cltv", "transactions_typed"],
                outputs="kpi_data_for_ui",
                name="prepare_kpi_data",
            ),
            node(
                func=prepare_segment_summary_data,
                inputs="rfm_segmented_with_historical_cltv",
                outputs="segment_summary_data_for_ui",
                name="prepare_segment_summary_data",
            ),
            node(
                func=prepare_segment_counts_data, # NEW NODE
                inputs="rfm_segmented_with_historical_cltv",
                outputs="segment_counts_data_for_ui",
                name="prepare_segment_counts_data",
            ),
            node(
                func=prepare_top_products_by_segment_data,
                inputs=["orders_merged_with_user_id", "transactions_typed", "rfm_segmented_with_historical_cltv"],
                outputs="top_products_by_segment_data_for_ui",
                name="prepare_top_products_by_segment_data",
            ),
            node(
                func=prepare_predicted_cltv_display_data,
                inputs=["rfm_segmented_with_historical_cltv", "predicted_cltv_df"],
                outputs="predicted_cltv_display_data_for_ui",
                name="prepare_predicted_cltv_display_data",
            ),
            node(
                func=prepare_cltv_comparison_data,
                inputs="predicted_cltv_display_data_for_ui", # This DF now contains predicted_cltv_3m
                outputs="cltv_comparison_data_for_ui",
                name="prepare_cltv_comparison_data",
            ),
            node(
                func=calculate_realization_curve_data,
                inputs=["orders_merged_with_user_id", "rfm_segmented_with_historical_cltv"],
                outputs="realization_curve_data_for_ui",
                name="calculate_realization_curve_data",
            ),
            node(
                func=prepare_churn_summary_data,
                inputs=["rfm_segmented_with_cox_predictions"], # This DF now contains predicted_churn_prob and expected_active_days
                outputs=["churn_summary_data_for_ui", "active_days_summary_data_for_ui"],
                name="prepare_churn_summary_data",
            ),
            node(
                func=prepare_churn_detailed_view_data,
                inputs="rfm_segmented_with_cox_predictions", # This DF now contains all necessary churn columns
                outputs="churn_detailed_view_data_for_ui",
                name="prepare_churn_detailed_view_data",
            ),

            # Final Data Assembly for UI (Combines all relevant customer-level data into a single DF)
            node(
                func=lambda rfm, cltv_pred_df, churn_prob_df, churn_labels_df, cox_pred_df: rfm
                    .merge(cltv_pred_df, on='User ID', how='left')
                    .merge(churn_prob_df, left_index=True, right_index=True, how='left')
                    .merge(churn_labels_df, left_index=True, right_index=True, how='left')
                    .merge(cox_pred_df[['User ID', 'expected_active_days']], on='User ID', how='left'),
                inputs=[
                    "rfm_segmented_with_historical_cltv",
                    "predicted_cltv_df",
                    "predicted_churn_probabilities",
                    "predicted_churn_labels",
                    "rfm_segmented_with_cox_predictions"
                ],
                outputs="final_rfm_cltv_churn_data",
                name="assemble_final_customer_data",
            ),
        ],
        inputs=["orders_raw", "transactions_raw"],
        outputs=[
            "final_rfm_cltv_churn_data",
            "kpi_data_for_ui",
            "segment_summary_data_for_ui",
            "segment_counts_data_for_ui", # NEW OUTPUT
            "top_products_by_segment_data_for_ui",
            "predicted_cltv_display_data_for_ui",
            "cltv_comparison_data_for_ui",
            "realization_curve_data_for_ui",
            "churn_summary_data_for_ui",
            "active_days_summary_data_for_ui",
            "churn_detailed_view_data_for_ui",
            "customers_at_risk_df"
        ],
    )
