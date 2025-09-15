# src/cltv_base/pipelines/lightgbm_churn/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import (
    train_lightgbm_model,
    predict_lightgbm_probabilities
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_lightgbm_model,
            inputs=[
                "churn_features",
                "churn_labels",
                "params:churn_model_random_state"
            ],
            outputs=[
                "lightgbm_churn_model",
                "lightgbm_classification_report",
                "lightgbm_feature_importances",
                "lightgbm_features_test",
                "lightgbm_labels_test"
            ],
            name="train_lightgbm_model",
        ),
        node(
            func=predict_lightgbm_probabilities,
            inputs=["lightgbm_churn_model", "churn_features"],
            outputs="lightgbm_predicted_churn_probabilities",
            name="predict_lightgbm_probabilities",
        ),
    ], tags="lightgbm_churn")