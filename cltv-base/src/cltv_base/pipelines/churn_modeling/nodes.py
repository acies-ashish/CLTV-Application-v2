# src/cltv_base/pipelines/churn_modeling/nodes.py

import pandas as pd
from typing import Tuple, Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lifelines import CoxPHFitter


def get_customers_at_risk(customer_level_df: pd.DataFrame, at_risk_threshold_days: int) -> pd.DataFrame:
    """
    Identifies customers at risk based on recency.
    This function acts as a Kedro node.
    Args:
        customer_level_df: DataFrame with customer-level features including 'recency'.
        at_risk_threshold_days: Threshold in days for considering a customer 'at risk'.
    Returns:
        DataFrame of customers identified as at risk.
    """
    print(f"Identifying customers at risk (recency > {at_risk_threshold_days} days)...")
    if customer_level_df.empty or 'recency' not in customer_level_df.columns:
        print("Warning: Missing 'recency' or empty DataFrame for customers at risk. Returning empty DataFrame.")
        return pd.DataFrame(columns=customer_level_df.columns) # Return empty with original columns
    return customer_level_df[customer_level_df['recency'] > at_risk_threshold_days]

def label_churned_customers(customer_df: pd.DataFrame, churn_inactive_days_threshold: int) -> pd.DataFrame:
    """
    Labels customers as churned based on recency.
    This function acts as a Kedro node.
    Args:
        customer_df: DataFrame with customer-level features including 'recency'.
        churn_inactive_days_threshold: Threshold in days for labeling a customer as 'churned'.
    Returns:
        DataFrame with an added 'is_churned' column.
    """
    print(f"Labeling churned customers (recency > {churn_inactive_days_threshold} days)...")
    df = customer_df.copy()
    if df.empty or 'recency' not in df.columns:
        print("Warning: Missing 'recency' or empty DataFrame for churn labeling. Returning original DataFrame with 'is_churned' set to 0.")
        if 'is_churned' not in df.columns:
            df['is_churned'] = 0
        return df
    
    df['is_churned'] = (df['recency'] > churn_inactive_days_threshold).astype(int)
    
    # --- Diagnostic Print ---
    churned_count = df['is_churned'].sum()
    total_customers = len(df)
    print(f"Diagnostic: Labeled {churned_count} out of {total_customers} customers as churned.")
    if total_customers > 0 and churned_count == 0:
        print("WARNING: No customers were labeled as churned. Consider adjusting 'churn_inactive_days_threshold'.")
    # --- End Diagnostic Print ---

    return df

def get_churn_features_labels(customer_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts features and labels for churn prediction.
    Ensures 'User ID' is the index for X and y to maintain mapping.
    This function acts as a Kedro node.
    Args:
        customer_df: DataFrame containing customer-level features and 'is_churned' label.
    Returns:
        Tuple of DataFrames (features X, labels y).
    """
    print("Extracting churn features and labels...")
    
    # Ensure 'User ID' is present and set as index
    if 'User ID' not in customer_df.columns:
        raise ValueError("customer_df must contain 'User ID' column for churn feature extraction.")
    
    # Create a copy and set 'User ID' as the index
    df_indexed = customer_df.set_index('User ID', drop=False) 

    feature_cols = [
        'frequency', 'monetary', 'aov',
        'avg_days_between_orders', 'CLTV_30d', 'CLTV_60d', 'CLTV_90d'
    ]
    
    # Ensure all feature columns exist before selection
    existing_feature_cols = [col for col in feature_cols if col in df_indexed.columns] # Corrected: was feature_indexed.columns
    if len(existing_feature_cols) < len(feature_cols):
        missing_cols = set(feature_cols) - set(existing_feature_cols)
        print(f"Warning: Missing churn feature columns: {missing_cols}. Using available features.")
    
    X = df_indexed[existing_feature_cols]
    y = df_indexed['is_churned']
    
    # --- Diagnostic Print ---
    print(f"Diagnostic: get_churn_features_labels - X index sample:\n{X.index[:5].tolist()}")
    print(f"Diagnostic: get_churn_features_labels - y (is_churned) value counts:\n{y.value_counts()}")
    # --- End Diagnostic Print ---

    return X, y.to_frame() # y.to_frame() is fine, it will inherit the index


def train_churn_prediction_model(X: pd.DataFrame, y: pd.DataFrame, n_estimators: int, random_state: int) -> Tuple[RandomForestClassifier, Dict, List, pd.DataFrame, pd.DataFrame]:
    """
    Trains a RandomForestClassifier for churn prediction.
    This function acts as a Kedro node.
    Args:
        X: Feature DataFrame.
        y: Label DataFrame ('is_churned').
        n_estimators: Number of trees in the RandomForest.
        random_state: Random state for reproducibility.
    Returns:
        Tuple of (trained model, classification report, feature importances, X_test, y_test).
    """
    print("Training churn prediction model...")
    if X.empty or y.empty:
        print("Warning: X or y DataFrame is empty for churn model training. Returning empty results.")
        # Return dummy values to prevent downstream errors
        dummy_model = RandomForestClassifier(n_estimators=1, random_state=random_state)
        dummy_report = {"accuracy": 0.0, "precision": {"0": 0.0, "1": 0.0}, "recall": {"0": 0.0, "1": 0.0}, "f1-score": {"0": 0.0, "1": 0.0}}
        dummy_importances = []
        dummy_X_test = pd.DataFrame(columns=X.columns)
        dummy_y_test = pd.DataFrame(columns=y.columns)
        return dummy_model, dummy_report, dummy_importances, dummy_X_test, dummy_y_test

    # X and y should already have 'User ID' as index from get_churn_features_labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random_state)
    
    # --- Diagnostic Print ---
    print(f"Diagnostic: train_churn_prediction_model - y_train (is_churned) value counts:\n{y_train.value_counts()}")
    if y_train['is_churned'].sum() == 0:
        print("WARNING: No churned customers in training data. Model will likely predict no churn.")
    # --- End Diagnostic Print ---

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train.values.ravel()) # .values.ravel() flattens the DataFrame to a 1D array
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    importances = model.feature_importances_.tolist()
    return model, report, importances, X_test, y_test

def predict_churn_probabilities(model: RandomForestClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts churn probabilities using the trained model.
    This function acts as a Kedro node.
    Args:
        model: Trained RandomForestClassifier model.
        X: Feature DataFrame for prediction.
    Returns:
        DataFrame with 'User ID' and 'predicted_churn_prob'.
    """
    print("Predicting churn probabilities...")
    if X.empty:
        print("Warning: X DataFrame is empty for churn probability prediction. Returning empty DataFrame.")
        return pd.DataFrame(columns=['User ID', 'predicted_churn_prob'])

    # X's index should be 'User ID' from upstream node (get_churn_features_labels)
    prob_df = pd.DataFrame(model.predict_proba(X)[:, 1], index=X.index, columns=['predicted_churn_prob'])
    prob_df.index.name = 'User ID' # Explicitly name the index
    prob_df = prob_df.reset_index() # Convert index to column
    prob_df['User ID'] = prob_df['User ID'].astype(str) # Ensure User ID is string
    
    # --- Diagnostic Print ---
    print(f"Diagnostic: predict_churn_probabilities - predicted_churn_prob describe:\n{prob_df['predicted_churn_prob'].describe()}")
    print(f"Diagnostic: predict_churn_probabilities - User ID sample:\n{prob_df['User ID'].head().tolist()}")
    # --- End Diagnostic Print ---

    return prob_df

def assign_predicted_churn_labels(predicted_churn_prob: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Assigns binary churn labels based on a probability threshold.
    This function acts as a Kedro node.
    Args:
        predicted_churn_prob: DataFrame with 'User ID' and 'predicted_churn_prob'.
        threshold: Probability threshold to classify as churned.
    Returns:
        DataFrame with 'User ID' and 'predicted_churn'.
    """
    print("Assigning predicted churn labels...")
    if predicted_churn_prob.empty or 'predicted_churn_prob' not in predicted_churn_prob.columns:
        print("Warning: Missing 'predicted_churn_prob' or empty DataFrame for churn labels. Returning empty DataFrame.")
        return pd.DataFrame(columns=['User ID', 'predicted_churn'])

    # Use the 'User ID' column from predicted_churn_prob directly and create the new column
    churn_labels_df = predicted_churn_prob[['User ID']].copy() # Start with User ID
    churn_labels_df['predicted_churn'] = (predicted_churn_prob['predicted_churn_prob'] >= threshold).astype(int)
    churn_labels_df['User ID'] = churn_labels_df['User ID'].astype(str) # Ensure User ID is string
    
    # --- Diagnostic Print ---
    print(f"Diagnostic: assign_predicted_churn_labels - predicted_churn value counts:\n{churn_labels_df['predicted_churn'].value_counts()}")
    print(f"Diagnostic: assign_predicted_churn_labels - User ID sample:\n{churn_labels_df['User ID'].head().tolist()}")
    if churn_labels_df['predicted_churn'].sum() == 0:
        print("WARNING: No customers predicted as churned. Consider adjusting 'predicted_churn_probability_threshold'.")
    # --- End Diagnostic Print ---

    return churn_labels_df

def prepare_survival_data(customer_df: pd.DataFrame, churn_inactive_days_threshold: int) -> pd.DataFrame:
    """
    Adds 'duration' and 'event' for survival analysis.
    This function acts as a Kedro node.
    Args:
        customer_df: DataFrame with customer-level features including 'lifespan_1d' and 'recency'.
        churn_inactive_days_threshold: Threshold in days for labeling a customer as 'churned' (event).
    Returns:
        DataFrame with 'duration' and 'event' columns added.
    """
    print("Preparing survival data...")
    df = customer_df.copy()
    if df.empty or 'lifespan_1d' not in df.columns or 'recency' not in df.columns:
        print("Warning: Missing required columns or empty DataFrame for survival data. Returning empty DataFrame.")
        return pd.DataFrame(columns=df.columns.tolist() + ['duration', 'event'])
    
    df['duration'] = df['lifespan_1d']
    df['event'] = (df['recency'] > churn_inactive_days_threshold).astype(int)
    return df

def train_cox_survival_model(customer_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[CoxPHFitter, pd.DataFrame]:
    """
    Trains Cox Proportional Hazards model for churn time prediction.
    Returns trained model and predicted expected churn times.
    This function acts as a Kedro node.
    Args:
        customer_df: DataFrame with customer-level features, 'duration', and 'event'.
        feature_cols: List of feature columns to use for the Cox model.
    Returns:
        Tuple of (trained CoxPHFitter model, DataFrame with 'expected_active_days' added).
    """
    print("Training Cox survival model...")
    df = customer_df.copy()
    cph = CoxPHFitter()

    required_cols = feature_cols + ['duration', 'event']
    if df.empty or not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Warning: Missing required columns for Cox model training: {missing} or empty DataFrame. Returning original DataFrame with 'expected_active_days' set to 0.")
        if 'expected_active_days' not in df.columns:
            df['expected_active_days'] = 0
        return cph, df # Return dummy cph and original df

    survival_df = df[required_cols]
    cph.fit(survival_df, duration_col='duration', event_col='event')

    # Ensure 'User ID' is string type before returning
    df['User ID'] = df['User ID'].astype(str)
    df['expected_active_days'] = cph.predict_expectation(survival_df).round(0).astype(int)
    return cph, df
