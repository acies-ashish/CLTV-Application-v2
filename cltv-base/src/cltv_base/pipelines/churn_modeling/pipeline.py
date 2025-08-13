from kedro.pipeline import Pipeline, node
from .nodes import (
    calculate_at_risk_threshold,
    get_customers_at_risk,
    label_churned_customers,
    get_churn_features_labels,
    train_churn_prediction_model,
    predict_churn_probabilities,
    assign_predicted_churn_labels,
    prepare_survival_data,
    train_cox_survival_model
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Churn modeling pipeline:
    - Calculates dynamic at-risk threshold
    - Flags at-risk customers
    - Labels churned customers
    - Trains churn model
    - Predicts churn probabilities and labels
    - Trains survival model for churn time prediction
    """
    return Pipeline([
        node(
            func=calculate_at_risk_threshold,
            inputs="historical_cltv_customers",
            outputs="calculated_at_risk_threshold",
            name="calculate_at_risk_threshold"
        ),
        node(
            func=get_customers_at_risk,
            inputs=["historical_cltv_customers", "calculated_at_risk_threshold"],
            outputs="customers_at_risk_df",
            name="get_customers_at_risk"
        ),
        node(
            func=label_churned_customers,
            inputs=["historical_cltv_customers", "params:churn_inactive_days_threshold"],
            outputs="churn_labeled_customers",
            name="label_churned_customers"
        ),
        node(
            func=get_churn_features_labels,
            inputs="churn_labeled_customers",
            outputs=["churn_features", "churn_labels"],
            name="get_churn_features_labels"
        ),
        node(
            func=train_churn_prediction_model,
            inputs=[
                "churn_features", "churn_labels",
                "params:churn_model_n_estimators", "params:churn_model_random_state"
            ],
            outputs=[
                "churn_prediction_model", "churn_classification_report",
                "churn_feature_importances", "churn_features_test", "churn_labels_test"
            ],
            name="train_churn_prediction_model"
        ),
        node(
            func=predict_churn_probabilities,
            inputs=["churn_prediction_model", "churn_features"],
            outputs="predicted_churn_probabilities",
            name="predict_churn_probabilities"
        ),
        node(
            func=assign_predicted_churn_labels,
            inputs=["predicted_churn_probabilities", "params:predicted_churn_probability_threshold"],
            outputs="predicted_churn_labels",
            name="assign_predicted_churn_labels"
        ),
        node(
            func=prepare_survival_data,
            inputs=["historical_cltv_customers", "params:churn_inactive_days_threshold"],
            outputs="survival_data",
            name="prepare_survival_data"
        ),
        node(
            func=train_cox_survival_model,
            inputs=["survival_data", "params:cox_feature_cols"],
            outputs=["cox_survival_model", "cox_predicted_active_days"],
            name="train_cox_survival_model"
        ),
    ])
