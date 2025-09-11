import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_xgboost_model(
    X: pd.DataFrame, y: pd.DataFrame, random_state: int
) -> Tuple[xgb.XGBClassifier, Dict, List, pd.DataFrame, pd.DataFrame]:
    """
    Trains an XGBoost model for churn prediction.
    """
    print("[INFO] Training XGBoost model...")
    
    # Check for empty data or single class
    if X.empty or y.empty or y.nunique().max() < 2:
        print("[WARN] Insufficient data or single class. Returning dummy model.")
        dummy_model = xgb.XGBClassifier(random_state=random_state)
        return dummy_model, {}, [], pd.DataFrame(), pd.DataFrame()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )

    # Train the XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    model.fit(X_train, y_train.values.ravel())
    
    # Evaluate the model
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    importances = model.feature_importances_.tolist()
    
    return model, report, importances, X_test, y_test

def predict_xgboost_probabilities(model: xgb.XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts churn probabilities using the trained XGBoost model.
    """
    if X.empty:
        return pd.DataFrame(columns=['User ID', 'predicted_churn_prob'])
    
    # Predict probabilities
    probs = model.predict_proba(X)[:, 1]
    
    predictions = pd.DataFrame(data={
        'User ID': X.index,  
        'predicted_churn_prob': probs
    })
    return predictions