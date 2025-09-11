# src/cltv_base/pipelines/model_comparison_ui/nodes.py

import pandas as pd
from typing import Dict

def combine_model_reports(
    rf_report: Dict, xgb_report: Dict
) -> pd.DataFrame:
    """
    Combines classification reports from multiple models into a single DataFrame
    for UI display.
    """
    if not rf_report and not xgb_report:
        print("[WARN] No model reports available for comparison.")
        return pd.DataFrame()

    reports = {}
    if rf_report:
        reports["Random Forest"] = rf_report
    if xgb_report:
        reports["XGBoost"] = xgb_report

    data = []
    for model_name, report in reports.items():
        if "accuracy" in report:
            accuracy = report["accuracy"]
            data.append({
                "Model": model_name,
                "Metric": "Accuracy",
                "Value": accuracy
            })
        if "macro avg" in report:
            macro_avg = report["macro avg"]
            data.append({
                "Model": model_name,
                "Metric": "Macro Precision",
                "Value": macro_avg["precision"]
            })
            data.append({
                "Model": model_name,
                "Metric": "Macro Recall",
                "Value": macro_avg["recall"]
            })
            data.append({
                "Model": model_name,
                "Metric": "Macro F1-Score",
                "Value": macro_avg["f1-score"]
            })

    df = pd.DataFrame(data)
    return df