# src/cltv_base/pipelines/cltv_modeling/nodes.py

import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter

def predict_cltv_bgf_ggf(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits BG/NBD and Gamma-Gamma models to predict 3-month CLTV (fixed horizon).
    This function acts as a Kedro node.
    """
    print(f"Predicting CLTV using BG/NBD + Gamma-Gamma models for 3 months (fixed horizon)...")
    df = transactions_df.copy()
    
    if not all(col in df.columns for col in ['Purchase Date', 'User ID', 'Total Amount']):
        raise KeyError("Missing required columns ('Purchase Date', 'User ID', 'Total Amount') in the transaction dataset for CLTV prediction.")

    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])

    # Use the full transactions data for CLTV calculation
    observation_period_end = df['Purchase Date'].max()

    summary_df = summary_data_from_transaction_data(
        df, # Use full data
        customer_id_col='User ID',
        datetime_col='Purchase Date',
        monetary_value_col='Total Amount',
        observation_period_end=observation_period_end
    )

    summary_df = summary_df[(summary_df['frequency'] > 0) & (summary_df['monetary_value'] > 0)]

    if summary_df.empty:
        print("Warning: No valid data for CLTV prediction after filtering. Returning empty DataFrame.")
        return pd.DataFrame(columns=['User ID', 'predicted_cltv_3m'])

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary_df['frequency'], summary_df['recency'], summary_df['T'])

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(summary_df['frequency'], summary_df['monetary_value'])

    summary_df['predicted_cltv_3m'] = ggf.customer_lifetime_value(
        bgf,
        summary_df['frequency'],
        summary_df['recency'],
        summary_df['T'],
        summary_df['monetary_value'],
        time=3, # Fixed to 3 months
        freq='D', 
        discount_rate=0.01
    )

    summary_df = summary_df.reset_index()
    # Ensure 'User ID' is string type after reset_index
    summary_df['User ID'] = summary_df['User ID'].astype(str)
    
    # --- Diagnostic Print ---
    print(f"Diagnostic: predict_cltv_bgf_ggf - predicted_cltv_df head:\n{summary_df.head()}")
    print(f"Diagnostic: predict_cltv_bgf_ggf - predicted_cltv_df null counts:\n{summary_df.isnull().sum()}")
    # --- End Diagnostic Print ---

    return summary_df[['User ID', 'predicted_cltv_3m']]
