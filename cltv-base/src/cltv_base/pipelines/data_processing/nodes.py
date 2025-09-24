import pandas as pd
import difflib
from typing import Tuple

# --- Helper function for column mapping --- 
def _auto_map_column(column_list, candidate_names):
    
    for name in candidate_names:
        match = difflib.get_close_matches(name, column_list, n=1, cutoff=0.6)
        if match:
            return match[0]
    return None

def standardize_columns(df: pd.DataFrame, expected_mapping: dict, df_name: str) -> pd.DataFrame:
    
    print(f"Standardizing columns for {df_name} DataFrame...")
    column_map = {}
    for standard_name, candidates in expected_mapping.items():
        mapped_col = _auto_map_column(df.columns.tolist(), candidates)
        if mapped_col:
            column_map[mapped_col] = standard_name
        else:
            print(f"Warning: Could not find a suitable column for '{standard_name}' in {df_name} DataFrame.")
    
    # Rename columns that were successfully mapped
    df_standardized = df.rename(columns=column_map)
    print(df_standardized.info())
    return df_standardized

def convert_data_types(
    orders_df: pd.DataFrame, 
    transactions_df: pd.DataFrame, 
    behavioral_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    print("Converting data types...")

    # --- Transactions ---
    if 'Purchase Date' in transactions_df.columns:
        transactions_df['Purchase Date'] = pd.to_datetime(
            transactions_df['Purchase Date'], dayfirst=False, errors='coerce'
        )
    else:
        print("Warning: 'Purchase Date' not found in transactions_df.")

    if 'User ID' in transactions_df.columns:
        transactions_df['User ID'] = transactions_df['User ID'].astype(str)
    else:
        print("Warning: 'User ID' not found in transactions_df.")

    numeric_cols_txn = ['Total Amount', 'Total Payable', 'Discount Value', 'Shipping Cost']
    for col in numeric_cols_txn:
        if col in transactions_df.columns:
            transactions_df[col] = pd.to_numeric(transactions_df[col], errors='coerce')

    # --- Orders ---
    if 'Return Date' in orders_df.columns:
        orders_df['Return Date'] = pd.to_datetime(
            orders_df['Return Date'], dayfirst=True, errors='coerce'
        )
    else:
        print("Warning: 'Return Date' not found in orders_df.")

    numeric_cols_orders = ['Unit Price', 'Quantity']
    for col in numeric_cols_orders:
        if col in orders_df.columns:
            orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce')

    # --- Behavioral ---
    if 'Visit Timestamp' in behavioral_df.columns:
        behavioral_df['Visit Timestamp'] = pd.to_datetime(
            behavioral_df['Visit Timestamp'], errors='coerce'
        )
    else:
        print("Warning: 'Visit Timestamp' not found in behavioral_df.")

    numeric_cols_behavioral = [
        'Session Total Cost',
        'Session Duration',
        'Page Views'
    ]
    for col in numeric_cols_behavioral:
        if col in behavioral_df.columns:
            behavioral_df[col] = pd.to_numeric(behavioral_df[col], errors='coerce')

    bool_like_cols = [
        'Sponsored Listing Viewed',
        'Banner Viewed',
        'Homepage Promo Seen',
        'Product Search View',
        'Bounce Flag'
    ]
    for col in bool_like_cols:
        if col in behavioral_df.columns:
            # Convert truthy/falsy values into proper booleans
            behavioral_df[col] = behavioral_df[col].astype(bool)

    # Ensure ID-like fields are strings
    id_cols_behavioral = [
        'Visit ID', 'Customer ID', 'Session ID', 'Device ID', 'Cookie ID', 'Ad Campaign ID'
    ]
    for col in id_cols_behavioral:
        if col in behavioral_df.columns:
            behavioral_df[col] = behavioral_df[col].astype(str)

    # String/categorical type (optional: keep free text as str)
    str_cols_behavioral = [
        'Channel', 'Geo Location', 'Device Type', 'OS', 'Entry Page', 'Exit Page'
    ]
    for col in str_cols_behavioral:
        if col in behavioral_df.columns:
            behavioral_df[col] = behavioral_df[col].astype(str)

    return orders_df, transactions_df, behavioral_df


def merge_orders_transactions(orders_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:

    print("Merging orders and transactions...")
    if 'Transaction ID' in transactions_df.columns and \
       'Transaction ID' in orders_df.columns and \
       'User ID' in transactions_df.columns:
        
        if 'User ID' in transactions_df.columns:
            df_orders_merged = orders_df.merge(
                transactions_df[['Transaction ID', 'User ID']],
                on='Transaction ID',
                how='left'
            )
        else:
            print("Warning: 'User ID' not found in transactions_df for merge. Skipping User ID merge.")
            df_orders_merged = orders_df.copy()
    else:
        print("Warning: 'Transaction ID' or 'User ID' not found in both orders and transactions for merging. Skipping merge.")
        df_orders_merged = orders_df.copy()
    print(df_orders_merged.info())
    return df_orders_merged
