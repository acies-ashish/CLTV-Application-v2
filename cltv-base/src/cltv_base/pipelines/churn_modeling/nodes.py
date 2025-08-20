import pandas as pd
from typing import Tuple, Dict, List, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lifelines import CoxPHFitter

# -------------------------
# Supported Metrics
# -------------------------
METRICS_SUPPORTED = ['recency', 'frequency', 'monetary', 'rfm_score']

# -------------------------
# Distribution-based Threshold Calculation
# -------------------------
def calculate_all_distribution_thresholds(customer_df: pd.DataFrame, metrics: list = None) -> dict:
    """
    Calculate 75th percentile after removing top/bottom 10% outliers for all specified metrics.
    Returns a dictionary with {metric}_threshold for each metric.
    """
    if metrics is None:
        metrics = ['recency', 'frequency', 'monetary', 'rfm_score']
    thresholds = {}
    for metric in metrics:
        if customer_df.empty or metric not in customer_df.columns:
            print(f"[WARN] No data or '{metric}' missing. Using threshold=0")
            thresholds[f"{metric}_threshold"] = 0
        else:
            values = customer_df[metric]
            lower = values.quantile(0.10)
            upper = values.quantile(0.90)
            filtered = values[(values >= lower) & (values <= upper)]
            threshold_value = float(filtered.quantile(0.75))
            print(f"[INFO] {metric} threshold calculated: {threshold_value:.2f}")
            thresholds[f"{metric}_threshold"] = threshold_value
    return thresholds

# -------------------------
# User-Defined Value Threshold
# -------------------------
def calculate_user_value_threshold(metric: str, user_value: float) -> dict:
    """
    Use user-selected value as threshold for the chosen metric.
    """
    print(f"[INFO] User set {metric} threshold: {user_value:.2f}")
    return {f"{metric}_threshold": user_value}

# -------------------------
# ML-Based Threshold Calculation (Stub for future)
# -------------------------
def calculate_ml_based_threshold(customer_df: pd.DataFrame, metric: str) -> dict:
    """
    Placeholder—future ML-based thresholding.
    """
    # Integration: implement ML threshold selection here
    print(f"[WARN] ML threshold for '{metric}' not implemented. Using default value = 0.")
    return {f"{metric}_threshold": 0}

# -------------------------
# Get Customers At Risk (Generic for any metric)
# -------------------------
def get_customers_at_risk(customer_df: pd.DataFrame, threshold_data, metric: str) -> pd.DataFrame:
    """
    Select customers above/below the threshold based on metric.
    """
    # Safe extraction logic
    if isinstance(threshold_data, dict):
        threshold = threshold_data.get(f"{metric}_threshold", 0)
    elif isinstance(threshold_data, (int, float)):
        threshold = threshold_data
    elif isinstance(threshold_data, str):
        try:
            threshold = float(threshold_data)
        except ValueError:
            threshold = 0
    else:
        threshold = 0

    df = customer_df.copy()
    if metric not in df.columns:
        print(f"[WARN] Metric '{metric}' missing. Can't flag at risk.")
        return pd.DataFrame()
    if metric in ['recency']:
        at_risk_df = df[df[metric] > threshold].copy()
    else:
        at_risk_df = df[df[metric] < threshold].copy()
    at_risk_df['status'] = 'At Risk'
    print(f"[INFO] {len(at_risk_df)} customers flagged as 'At Risk' by '{metric}'.")
    return at_risk_df

# -------------------------
# Label Churned Customers (for any metric)
# -------------------------
def label_churned_customers(customer_df: pd.DataFrame, metric: str, inactive_days_threshold: float) -> pd.DataFrame:
    """
    Adds 'is_churned' column based on the metric and threshold.
    For 'recency' and 'rfm_score', above threshold = churned.
    For 'frequency' and 'monetary', below threshold = churned.
    """
    df = customer_df.copy()
    if df.empty or metric not in df.columns:
        print(f"[WARN] Missing '{metric}'. Setting is_churned=0 for all.")
        df['is_churned'] = 0
        return df
    if metric in ['recency', 'rfm_score']:
        df['is_churned'] = (df[metric] > inactive_days_threshold).astype(int)
    else:
        df['is_churned'] = (df[metric] < inactive_days_threshold).astype(int)
    churned_count = df['is_churned'].sum()
    print(f"[INFO] Labeled {churned_count} customers as churned by '{metric}'.")
    return df

# -------------------------
# Feature & Label Extraction, Model Training, Prediction — (unchanged from your code)
# -------------------------
def get_churn_features_labels(customer_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if 'User ID' not in customer_df.columns:
        raise ValueError("'User ID' column is required.")
    df = customer_df.set_index('User ID', drop=False)
    feature_cols = [
        'frequency', 'monetary', 'aov',
        'avg_days_between_orders', 'CLTV_30d', 'CLTV_60d', 'CLTV_90d'
    ]
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < len(feature_cols):
        print(f"[WARN] Missing features: {set(feature_cols) - set(available_cols)}")
    X = df[available_cols]
    y = df[['is_churned']]
    return X, y

def train_churn_prediction_model(
    X: pd.DataFrame, y: pd.DataFrame, n_estimators: int, random_state: int
) -> Tuple[RandomForestClassifier, Dict, List, pd.DataFrame, pd.DataFrame]:
    if X.empty or y.empty:
        print("[WARN] Empty features or labels. Returning dummy model.")
        dummy_model = RandomForestClassifier(n_estimators=1, random_state=random_state)
        return dummy_model, {}, [], pd.DataFrame(), pd.DataFrame()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train.values.ravel())
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    importances = model.feature_importances_.tolist()
    return model, report, importances, X_test, y_test

def predict_churn_probabilities(model: RandomForestClassifier, X: pd.DataFrame) -> pd.DataFrame:
    if X.empty:
        return pd.DataFrame(columns=['User ID', 'predicted_churn_prob'])
    probs = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        'User ID': X.index.astype(str),
        'predicted_churn_prob': probs
    })

def assign_predicted_churn_labels(predicted_churn_prob: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if predicted_churn_prob.empty:
        return pd.DataFrame(columns=['User ID', 'predicted_churn'])
    predicted_churn_prob = predicted_churn_prob.copy()
    predicted_churn_prob['predicted_churn'] = (
        predicted_churn_prob['predicted_churn_prob'] >= threshold
    ).astype(int)
    return predicted_churn_prob[['User ID', 'predicted_churn']]

def prepare_survival_data(customer_df: pd.DataFrame, churn_inactive_days_threshold: int) -> pd.DataFrame:
    df = customer_df.copy()
    if df.empty or 'lifespan_1d' not in df.columns or 'recency' not in df.columns:
        return pd.DataFrame()
    df['duration'] = df['lifespan_1d']
    df['event'] = (df['recency'] > churn_inactive_days_threshold).astype(int)
    return df

def train_cox_survival_model(customer_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[CoxPHFitter, pd.DataFrame]:
    df = customer_df.copy()
    required_cols = feature_cols + ['duration', 'event']
    if df.empty or not all(col in df.columns for col in required_cols):
        print("[WARN] Missing survival model columns.")
        return CoxPHFitter(), df
    cph = CoxPHFitter()
    cph.fit(df[required_cols], duration_col='duration', event_col='event')
    df['expected_active_days'] = cph.predict_expectation(df[feature_cols]).round(0).astype(int)
    return cph, df
