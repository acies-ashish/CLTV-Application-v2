from kedro.pipeline import Pipeline, node
from .nodes import (
    train_xgboost_model,
    predict_xgboost_probabilities
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_xgboost_model,
            inputs=[
                "churn_features", 
                "churn_labels",
                "params:churn_model_random_state"
            ],
            outputs=[
                "xgboost_churn_model", 
                "xgboost_classification_report", 
                "xgboost_feature_importances", 
                "xgboost_features_test", 
                "xgboost_labels_test"
            ],
            name="train_xgboost_model",
        ),
        node(
            func=predict_xgboost_probabilities,
            inputs=["xgboost_churn_model", "churn_features"],
            outputs="xgboost_predicted_churn_probabilities",
            name="predict_xgboost_probabilities",
        ),
    ], tags="xgboost_churn")