# src/cltv_base/pipelines/model_comparison_ui/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import combine_model_reports

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=combine_model_reports,
            inputs=[
                "churn_classification_report", # Random Forest report
                "xgboost_classification_report" # XGBoost report
            ],
            outputs="model_comparison_metrics_for_ui",
            name="combine_model_reports_for_ui",
        ),
    ], tags="model_comparison_ui")