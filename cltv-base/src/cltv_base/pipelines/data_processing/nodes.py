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
    
    date_candidates = expected_mapping.get("Order Date", [])
    if date_candidates:
        date_col = None
        for candidate in date_candidates:
            # Check if the candidate column exists in the DataFrame
            if candidate in df.columns:
                try:
                    # Attempt to convert a sample of the column to datetime
                    _ = pd.to_datetime(df[candidate].head(5), dayfirst=True)
                    date_col = candidate
                    break  # Found a suitable date column, no need to check others
                except (ValueError, TypeError):
                    # This candidate column does not contain valid dates
                    continue
        
        if date_col:
            column_map[date_col] = "Order Date"
            expected_mapping.pop("Order Date")
        else:
            print(f"Warning: Could not find a suitable column for 'Order Date' in {df_name} DataFrame.")
            
    # 2. Process all other columns using the original _auto_map_column helper
    for standard_name, candidates in expected_mapping.items():
        mapped_col = _auto_map_column(df.columns.tolist(), candidates)
        if mapped_col:
            column_map[mapped_col] = standard_name
        else:
            print(f"Warning: Could not find a suitable column for '{standard_name}' in {df_name} DataFrame.")
    
    df_standardized = df.rename(columns=column_map)
    
    print(f"Post-standardization columns: {df_standardized.columns.tolist()}")
    print(f"Column map used: {column_map}")
    return df_standardized

def convert_data_types(orders_df: pd.DataFrame, transactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    print("Converting data types...")
    # Convert 'Purchase Date' in transactions_df
    if 'Purchase Date' in transactions_df.columns:
        transactions_df['Purchase Date'] = pd.to_datetime(transactions_df['Purchase Date'], dayfirst=False, errors='coerce')
    else:
        print("Warning: 'Purchase Date' not found in transactions_df for type conversion.")

    # Convert 'Return Date' in orders_df
    if 'Return Date' in orders_df.columns:
        orders_df['Return Date'] = pd.to_datetime(orders_df['Return Date'], dayfirst=True, errors='coerce')
    else:
        print("Warning: 'Return Date' not found in orders_df for type conversion.")

    # Convert numeric columns in orders_df
    numeric_columns = ['Unit Price', 'Total Amount', 'Total Payable', 'Quantity']
    for col in numeric_columns:
        if col in orders_df.columns:
            orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce')
    
    # Convert numeric columns in transactions_df (assuming 'Total Amount' is the main one)
    if 'Total Amount' in transactions_df.columns:
        transactions_df['Total Amount'] = pd.to_numeric(transactions_df['Total Amount'], errors='coerce')

    # Ensure 'User ID' is string type in transactions_df for consistent merging later
    if 'User ID' in transactions_df.columns:
        transactions_df['User ID'] = transactions_df['User ID'].astype(str)
    else:
        print("Warning: 'User ID' not found in transactions_df. Ensure it's handled upstream if needed for merges.")

    return orders_df, transactions_df

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
        
    return df_orders_merged
