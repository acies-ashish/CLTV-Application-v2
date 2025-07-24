# src/cltv_base/pipeline.py

import pandas as pd
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
    prepare_segment_counts_data,
    prepare_top_products_by_segment_data,
    prepare_predicted_cltv_display_data,
    prepare_cltv_comparison_data,
    calculate_realization_curve_data,
    prepare_churn_summary_data,
    prepare_churn_detailed_view_data
)

# New named function to encapsulate the final data assembly logic
def _assemble_final_customer_data(rfm: pd.DataFrame, cltv_pred_df: pd.DataFrame, 
                                  churn_prob_df: pd.DataFrame, churn_labels_df: pd.DataFrame, 
                                  cox_pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assembles the final customer-level DataFrame for UI display by merging various predictions.
    Includes diagnostic prints to check data integrity after merges.
    """
    # Aggressive cleaning and type conversion for 'User ID' in all relevant DataFrames
    # and deep dive diagnostics for churn dataframes
    for df_name, df in [
        ("rfm_segmented_with_historical_cltv", rfm),
        ("predicted_cltv_df", cltv_pred_df),
        ("predicted_churn_probabilities", churn_prob_df),
        ("predicted_churn_labels", churn_labels_df),
        ("rfm_segmented_with_cox_predictions", cox_pred_df)
    ]:
        if 'User ID' in df.columns:
            # Convert to string, strip whitespace, and then ensure unique IDs are handled
            df['User ID'] = df['User ID'].astype(str).str.strip()
            # If there are duplicate User IDs after stripping, we need to handle them.
            # In a real scenario, you might need a more sophisticated deduplication strategy.
            # For now, drop duplicates to ensure unique merge keys.
            df.drop_duplicates(subset=['User ID'], inplace=True)
        else:
            print(f"Warning: 'User ID' column not found in {df_name} during pre-merge cleaning.")

    print("\n--- Diagnostic: User ID counts and samples AFTER cleaning and BEFORE final merges ---")
    print(f"rfm_segmented_with_historical_cltv User IDs: {rfm['User ID'].nunique()} unique. Sample: {rfm['User ID'].head().tolist()}")
    print(f"predicted_cltv_df User IDs: {cltv_pred_df['User ID'].nunique()} unique. Sample: {cltv_pred_df['User ID'].head().tolist()}")
    print(f"predicted_churn_probabilities User IDs: {churn_prob_df['User ID'].nunique()} unique. Sample: {churn_prob_df['User ID'].head().tolist()}")
    print(f"predicted_churn_labels User IDs: {churn_labels_df['User ID'].nunique()} unique. Sample: {churn_labels_df['User ID'].head().tolist()}")
    print(f"rfm_segmented_with_cox_predictions User IDs: {cox_pred_df['User ID'].nunique()} unique. Sample: {cox_pred_df['User ID'].head().tolist()}")
    print("--- End User ID diagnostics ---")

    # Check for User ID intersection before merging
    base_users = set(rfm['User ID'].unique())
    churn_prob_users = set(churn_prob_df['User ID'].unique())
    churn_labels_users = set(churn_labels_df['User ID'].unique())

    print(f"\n--- Diagnostic: User ID Intersection Check ---")
    print(f"Users in base RFM but not in churn_prob_df: {len(base_users - churn_prob_users)}")
    print(f"Users in churn_prob_df but not in base RFM: {len(churn_prob_users - base_users)}")
    print(f"Users in base RFM but not in churn_labels_df: {len(base_users - churn_labels_users)}")
    print(f"Users in churn_labels_df but not in base RFM: {len(churn_labels_users - base_users)}")
    print("--- End Intersection Check ---")

    # --- DEEP DIVE DIAGNOSTICS FOR CHURN DATA BEFORE MERGE ---
    print("\n--- Deep Dive Diagnostic: predicted_churn_probabilities BEFORE MERGE ---")
    print(churn_prob_df.info())
    print("\nHead of predicted_churn_probabilities:")
    print(churn_prob_df.head())
    print("\nNull counts in predicted_churn_probabilities:")
    print(churn_prob_df.isnull().sum())
    print("\nColumns in predicted_churn_probabilities:", churn_prob_df.columns.tolist())

    print("\n--- Deep Dive Diagnostic: predicted_churn_labels BEFORE MERGE ---")
    print(churn_labels_df.info())
    print("\nHead of predicted_churn_labels:")
    print(churn_labels_df.head())
    print("\nNull counts in predicted_churn_labels:")
    print(churn_labels_df.isnull().sum())
    print("\nColumns in predicted_churn_labels:", churn_labels_df.columns.tolist())
    # --- END DEEP DIVE DIAGNOSTICS ---

    # Start with the base RFM DataFrame and chain merges
    final_df = rfm.merge(cltv_pred_df, on='User ID', how='left') \
                  .merge(churn_prob_df, on='User ID', how='left') \
                  .merge(churn_labels_df, on='User ID', how='left') \
                  .merge(cox_pred_df[['User ID', 'expected_active_days']], on='User ID', how='left')
    
    # --- Diagnostic: final_rfm_cltv_churn_data after merges ---
    print("\n--- Diagnostic: final_rfm_cltv_churn_data after merges ---")
    print(final_df.info())
    print("\nNull counts after merges:")
    print(final_df.isnull().sum())
    print("\nValue counts for predicted_churn after merges:")
    print(final_df['predicted_churn'].value_counts(dropna=False)) # Check counts including NaN
    print("\nDescribe for predicted_churn_prob after merges:")
    print(final_df['predicted_churn_prob'].describe())
    # --- End Diagnostic ---

    return final_df


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
                func=prepare_segment_counts_data,
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
                inputs="predicted_cltv_display_data_for_ui",
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
                inputs=["final_rfm_cltv_churn_data"], # Use the final combined data for churn summary
                outputs=["churn_summary_data_for_ui", "active_days_summary_data_for_ui"],
                name="prepare_churn_summary_data",
            ),
            node(
                func=prepare_churn_detailed_view_data,
                inputs="final_rfm_cltv_churn_data", # Use the final combined data for detailed view
                outputs="churn_detailed_view_data_for_ui",
                name="prepare_churn_detailed_view_data",
            ),

            # Final Data Assembly for UI (Combines all relevant customer-level data into a single DF)
            node(
                func=_assemble_final_customer_data, # Call the new named function
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
            "segment_counts_data_for_ui",
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
