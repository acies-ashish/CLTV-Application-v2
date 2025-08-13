import pandas as pd
from typing import Tuple, Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lifelines import CoxPHFitter


# -------------------------
# Threshold Calculation
# -------------------------
def calculate_at_risk_threshold(customer_df: pd.DataFrame) -> dict:
    """
    Dynamically calculates the at-risk threshold (in days) based on the 'recency' column
    after removing the top and bottom 10% outliers.
    Returns:
        dict: {"at_risk_threshold_days": float}
    """
    if customer_df.empty or 'recency' not in customer_df.columns:
        print("[WARN] No data or 'recency' missing. Using threshold=0")
        return {"at_risk_threshold_days": 0}

    recency = customer_df['recency']
    lower = recency.quantile(0.10)
    upper = recency.quantile(0.90)
    recency_filtered = recency[(recency >= lower) & (recency <= upper)]

    threshold_value = float(recency_filtered.quantile(0.75))
    print(f"[INFO] At-risk threshold calculated: {threshold_value:.2f} days")
    return {"at_risk_threshold_days": threshold_value}


# -------------------------
# At-Risk Customers
# -------------------------
def get_customers_at_risk(customer_level_df: pd.DataFrame, threshold_data: dict) -> pd.DataFrame:
    """
    Returns customers with recency above the calculated at-risk threshold.
    """
    threshold = threshold_data.get("at_risk_threshold_days", 0)
    df = customer_level_df[customer_level_df['recency'] > threshold].copy()
    df['status'] = 'At Risk'
    print(f"[INFO] {len(df)} customers flagged as 'At Risk'.")
    return df


# -------------------------
# Churn Labeling
# -------------------------
def label_churned_customers(customer_df: pd.DataFrame, churn_inactive_days_threshold: int) -> pd.DataFrame:
    """
    Adds an 'is_churned' column based on recency > threshold.
    """
    df = customer_df.copy()
    if df.empty or 'recency' not in df.columns:
        print("[WARN] Missing 'recency'. Setting is_churned=0 for all.")
        df['is_churned'] = 0
        return df

    df['is_churned'] = (df['recency'] > churn_inactive_days_threshold).astype(int)
    churned_count = df['is_churned'].sum()
    print(f"[INFO] Labeled {churned_count} customers as churned.")
    return df


# -------------------------
# Feature & Label Extraction
# -------------------------
def get_churn_features_labels(customer_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits customer data into features (X) and labels (y).
    Ensures User ID is used as index.
    """
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


# -------------------------
# Model Training
# -------------------------
def train_churn_prediction_model(
    X: pd.DataFrame, y: pd.DataFrame, n_estimators: int, random_state: int
) -> Tuple[RandomForestClassifier, Dict, List, pd.DataFrame, pd.DataFrame]:
    """
    Trains a RandomForest churn prediction model.
    """
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


# -------------------------
# Predictions
# -------------------------
def predict_churn_probabilities(model: RandomForestClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts churn probability for each customer.
    """
    if X.empty:
        return pd.DataFrame(columns=['User ID', 'predicted_churn_prob'])

    probs = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        'User ID': X.index.astype(str),
        'predicted_churn_prob': probs
    })


def assign_predicted_churn_labels(predicted_churn_prob: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Converts churn probability into binary churn label.
    """
    if predicted_churn_prob.empty:
        return pd.DataFrame(columns=['User ID', 'predicted_churn'])

    predicted_churn_prob = predicted_churn_prob.copy()
    predicted_churn_prob['predicted_churn'] = (
        predicted_churn_prob['predicted_churn_prob'] >= threshold
    ).astype(int)
    return predicted_churn_prob[['User ID', 'predicted_churn']]


# -------------------------
# Survival Analysis
# -------------------------
def prepare_survival_data(customer_df: pd.DataFrame, churn_inactive_days_threshold: int) -> pd.DataFrame:
    """
    Prepares survival analysis input by adding duration and event columns.
    """
    df = customer_df.copy()
    if df.empty or 'lifespan_1d' not in df.columns or 'recency' not in df.columns:
        return pd.DataFrame()

    df['duration'] = df['lifespan_1d']
    df['event'] = (df['recency'] > churn_inactive_days_threshold).astype(int)
    return df


def train_cox_survival_model(customer_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[CoxPHFitter, pd.DataFrame]:
    """
    Trains a Cox Proportional Hazards survival model.
    """
    df = customer_df.copy()
    required_cols = feature_cols + ['duration', 'event']
    if df.empty or not all(col in df.columns for col in required_cols):
        print("[WARN] Missing survival model columns.")
        return CoxPHFitter(), df

    cph = CoxPHFitter()
    cph.fit(df[required_cols], duration_col='duration', event_col='event')
    df['expected_active_days'] = cph.predict_expectation(df[feature_cols]).round(0).astype(int)
    return cph, df
