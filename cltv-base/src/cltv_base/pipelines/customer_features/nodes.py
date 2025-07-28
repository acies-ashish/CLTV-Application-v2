# src/cltv_base/pipelines/customer_features/nodes.py

import pandas as pd
from typing import Tuple, List

# --- Helper functions for new RFM scoring and segmentation ---
def assign_score(value, thresholds, reverse=False):
    """
    Assigns a score (1-5) based on a value and predefined thresholds.
    If reverse is True, higher values get lower scores (e.g., for Recency).
    """
    if reverse:
        if value <= thresholds[0]: return 5
        elif value <= thresholds[1]: return 4
        elif value <= thresholds[2]: return 3
        elif value <= thresholds[3]: return 2
        else: return 1
    else:
        if value <= thresholds[0]: return 1
        elif value <= thresholds[1]: return 2
        elif value <= thresholds[2]: return 3
        elif value <= thresholds[3]: return 4
        else: return 5

def assign_segment(row):
    """
    Assigns a customer segment based on R and FM scores.
    """
    r = row['r_score']
    # Ensure fm_score is calculated before calling this function
    # For now, assuming fm_score is available in the row
    fm = row['fm_score'] # This will be calculated as (f_score + m_score) / 2 or similar

    if (r == 5 and fm == 5) or (r == 5 and fm == 4) or (r == 4 and fm == 5):
        return 'Champions'
    elif (r == 5 and fm == 3) or (r == 4 and fm == 4) or (r == 3 and fm == 5) or (r == 3 and fm == 4):
        return 'Loyal Customers'
    elif (r == 5 and fm == 2) or (r == 4 and fm == 2) or (r == 3 and fm == 3) or (r == 4 and fm == 3):
        return 'Potential Loyalists'
    elif r == 5 and fm == 1:
        return 'Recent Customers'
    elif (r == 4 and fm == 1) or (r == 3 and fm == 1):
        return 'Promising'
    elif (r == 3 and fm == 2) or (r == 2 and fm == 3) or (r == 2 and fm == 2):
        return 'Customers Needing Attention'
    elif r == 2 and fm == 1:
        return 'About to Sleep'
    elif (r == 2 and fm == 5) or (r == 2 and fm == 4) or (r == 1 and fm == 3):
        return 'At Risk'
    elif (r == 1 and fm == 5) or (r == 1 and fm == 4):
        return "Can't Lose Them"
    elif r == 1 and fm == 2:
        return 'Hibernating'
    elif r == 1 and fm == 1:
        return 'Lost'
    else:
        return 'Unclassified'

# --- Data Processing Nodes ---

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

def perform_rfm_segmentation(
    customer_level_df: pd.DataFrame,
    rfm_recency_thresholds: List[float],
    rfm_frequency_thresholds: List[float],
    rfm_monetary_thresholds: List[float]
) -> pd.DataFrame:
    """
    Performs RFM segmentation on customer-level data using custom scoring and segment rules.
    This function acts as a Kedro node.
    """
    print("Performing RFM segmentation with new rules...")
    df = customer_level_df.copy()
    
    if df.empty:
        print("Warning: customer_level_df is empty. Cannot perform RFM segmentation. Returning empty DataFrame.")
        return pd.DataFrame(columns=[
            'User ID', 'recency', 'frequency', 'monetary', 'last_purchase', 
            'first_purchase', 'aov', 'avg_days_between_orders', 'lifespan_1d',
            'lifespan_7d', 'lifespan_15d', 'lifespan_30d', 'lifespan_60d',
            'lifespan_90d', 'CLTV_1d', 'CLTV_7d', 'CLTV_15d', 'CLTV_30d',
            'CLTV_60d', 'CLTV_90d', 'CLTV_total', 'r_score', 'f_score',
            'm_score', 'fm_score', 'rfm_segment', 'CLTV' # Updated column names
        ])

    # Ensure columns exist before applying scoring
    required_rfm_cols = ['recency', 'frequency', 'monetary']
    if not all(col in df.columns for col in required_rfm_cols):
        print(f"Warning: Missing RFM columns {set(required_rfm_cols) - set(df.columns)}. Cannot perform RFM segmentation. Returning original DataFrame.")
        return df # Return original if essential columns are missing

    # Apply custom scoring
    df['r_score'] = df['recency'].apply(lambda x: assign_score(x, rfm_recency_thresholds, reverse=True))
    df['f_score'] = df['frequency'].apply(lambda x: assign_score(x, rfm_frequency_thresholds))
    df['m_score'] = df['monetary'].apply(lambda x: assign_score(x, rfm_monetary_thresholds))

    # Calculate FM Score (average of F and M scores, rounded to nearest integer)
    df['fm_score'] = ((df['f_score'] + df['m_score']) / 2).round().astype(int)

    # Apply custom segmentation
    df['segment'] = df.apply(assign_segment, axis=1) # Renamed 'rfm_segment' to 'segment' for consistency with UI

    # Drop the individual R, F, M scores if they are not needed downstream,
    # but keep them for now as they are used in the segment assignment.
    # df = df.drop(columns=['R_score', 'F_score', 'M_score', 'RFM_score', 'RFM']) # Old columns to drop if present

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
