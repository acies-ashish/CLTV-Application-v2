# src/cltv_base/pipelines/customer_features/nodes.py

import pandas as pd
from typing import Tuple

def calculate_customer_level_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes customer-level RFM and other derived features.
    This function acts as a Kedro node.
    """
    print("Calculating customer level features...")
    if not all(col in transactions_df.columns for col in ['Purchase Date', 'Total Amount', 'User ID']):
        raise ValueError("transactions_df must contain 'Purchase Date', 'Total Amount', and 'User ID' columns.")
    
    transactions_df['Purchase Date'] = pd.to_datetime(transactions_df['Purchase Date'])

    today = transactions_df['Purchase Date'].max() + pd.Timedelta(days=1)

    customer_level = transactions_df.groupby('User ID').agg(
        recency=('Purchase Date', lambda x: (today - x.max()).days),
        frequency=('Purchase Date', 'count'),
        monetary=('Total Amount', 'sum'),
        last_purchase=('Purchase Date', 'max'),
        first_purchase=('Purchase Date', 'min')
    ).reset_index()

    # Ensure 'User ID' is string type after reset_index
    customer_level['User ID'] = customer_level['User ID'].astype(str)

    customer_level['aov'] = round(customer_level['monetary'] / customer_level['frequency'], 2)
    
    # Handle division by zero for avg_days_between_orders for customers with only one purchase
    customer_level['avg_days_between_orders'] = (
        (customer_level['last_purchase'] - customer_level['first_purchase']).dt.days / 
        (customer_level['frequency'] - 1).replace(0, pd.NA) # Replace 0 with NA to avoid division by zero
    )
    valid_avg = customer_level['avg_days_between_orders'][customer_level['avg_days_between_orders'].notna() & (customer_level['avg_days_between_orders'] != float('inf'))]
    median_gap = valid_avg.median() if not valid_avg.empty else 0
    customer_level['avg_days_between_orders'] = customer_level['avg_days_between_orders'].replace([float('inf'), -float('inf')], None)
    customer_level['avg_days_between_orders'] = customer_level['avg_days_between_orders'].fillna(median_gap).round(0).astype(int)

    customer_level['lifespan_1d'] = (customer_level['last_purchase'] - customer_level['first_purchase']).dt.days + 1
    customer_level['lifespan_7d'] = round(customer_level['lifespan_1d'] / 7, 2)
    customer_level['lifespan_15d'] = round(customer_level['lifespan_1d'] / 15, 2)
    customer_level['lifespan_30d'] = round(customer_level['lifespan_1d'] / 30, 2)
    customer_level['lifespan_60d'] = round(customer_level['lifespan_1d'] / 60, 2)
    customer_level['lifespan_90d'] = round(customer_level['lifespan_1d'] / 90, 2)
    
    customer_level['CLTV_1d'] = round(customer_level['monetary'] / customer_level['lifespan_1d'].replace(0, 1), 2)
    customer_level['CLTV_7d'] = round(customer_level['monetary'] / customer_level['lifespan_7d'].replace(0, 0.1), 2)
    customer_level['CLTV_15d'] = round(customer_level['monetary'] / customer_level['lifespan_15d'].replace(0, 0.1), 2)
    customer_level['CLTV_30d'] = round(customer_level['monetary'] / customer_level['lifespan_30d'].replace(0, 0.1), 2)
    customer_level['CLTV_60d'] = round(customer_level['monetary'] / customer_level['lifespan_60d'].replace(0, 0.1), 2)
    customer_level['CLTV_90d'] = round(customer_level['monetary'] / customer_level['lifespan_90d'].replace(0, 0.1), 2)
    customer_level['CLTV_total'] = customer_level['monetary']

    return customer_level

def perform_rfm_segmentation(customer_level_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs RFM segmentation on customer-level data using new, positive labels.
    This function acts as a Kedro node.
    """
    print("Performing RFM segmentation...")
    df = customer_level_df.copy()
    
    if df.empty:
        print("Warning: customer_level_df is empty. Cannot perform RFM segmentation. Returning empty DataFrame.")
        return pd.DataFrame(columns=[
            'User ID', 'recency', 'frequency', 'monetary', 'last_purchase', 
            'first_purchase', 'aov', 'avg_days_between_orders', 'lifespan_1d',
            'lifespan_7d', 'lifespan_15d', 'lifespan_30d', 'lifespan_60d',
            'lifespan_90d', 'CLTV_1d', 'CLTV_7d', 'CLTV_15d', 'CLTV_30d',
            'CLTV_60d', 'CLTV_90d', 'CLTV_total', 'R_score', 'F_score',
            'M_score', 'RFM_score', 'RFM', 'segment', 'CLTV'
        ])

    # Ensure columns exist before qcut
    required_rfm_cols = ['recency', 'frequency', 'monetary']
    if not all(col in df.columns for col in required_rfm_cols):
        print(f"Warning: Missing RFM columns {set(required_rfm_cols) - set(df.columns)}. Cannot perform RFM segmentation. Returning original DataFrame.")
        return df # Return original if essential columns are missing

    # Handle cases where qcut might fail due to insufficient unique values
    for col in required_rfm_cols:
        if df[col].nunique() < 5: # qcut needs at least 5 unique values for 5 quantiles
            print(f"Warning: Not enough unique values in '{col}' for 5-quantile segmentation. Using fewer quantiles or assigning default scores.")
            # Fallback: assign scores based on simpler logic or fewer quantiles
            # For simplicity, if less than 5 unique values, assign all to a single score for that metric
            if col == 'recency':
                df['R_score'] = 3 # Default to middle score
            elif col == 'frequency':
                df['F_score'] = 3
            elif col == 'monetary':
                df['M_score'] = 3
        else:
            if col == 'recency':
                df['R_score'] = pd.qcut(df['recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
            elif col == 'frequency':
                df['F_score'] = pd.qcut(df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
            elif col == 'monetary':
                df['M_score'] = pd.qcut(df['monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Ensure R_score, F_score, M_score exist after conditional assignment
    for score_col in ['R_score', 'F_score', 'M_score']:
        if score_col not in df.columns:
            df[score_col] = 3 # Assign a default if not created

    df['RFM_score'] = df['R_score'] + df['F_score'] + df['M_score']
    df['RFM'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)

    # Handle cases where quantiles might be undefined if RFM_score has too few unique values
    if df['RFM_score'].nunique() < 3: # Need at least 3 unique values for 3 segments
        print("Warning: Not enough unique RFM_score values for 3 segments. Assigning all to 'Active Shoppers'.")
        df['segment'] = 'Active Shoppers'
    else:
        q1 = df['RFM_score'].quantile(0.33)
        q2 = df['RFM_score'].quantile(0.66)

        def assign_segment(score):
            if score <= q1:
                return 'New Discoverers' # Changed from 'Low'
            elif score <= q2:
                return 'Active Shoppers' # Changed from 'Medium'
            else:
                return 'Loyalty Leaders' # Changed from 'High'

        df['segment'] = df['RFM_score'].apply(assign_segment)
    return df

def calculate_historical_cltv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates historical CLTV based on AOV and frequency.
    This function acts as a Kedro node.
    """
    print("Calculating historical CLTV...")
    if df.empty or 'aov' not in df.columns or 'frequency' not in df.columns:
        print("Warning: Missing 'aov' or 'frequency' or empty DataFrame for historical CLTV. Returning original DataFrame.")
        if 'CLTV' not in df.columns: # Ensure CLTV column exists even if calculation skipped
            df['CLTV'] = 0.0
        return df
    df['CLTV'] = df['aov'] * df['frequency']
    return df
