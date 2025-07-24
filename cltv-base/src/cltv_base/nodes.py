# src/customer_analytics/nodes.py

import pandas as pd
import difflib
import os
from datetime import datetime, timedelta

# --- Helper functions (from mapping.py) ---
def _auto_map_column(column_list, candidate_names):
    """Helper for column mapping."""
    for name in candidate_names:
        match = difflib.get_close_matches(name, column_list, n=1, cutoff=0.6)
        if match:
            return match[0]
    return None

def standardize_columns(df: pd.DataFrame, expected_mapping: dict, df_name: str) -> pd.DataFrame:
    """
    Standardizes column names of a DataFrame based on a predefined mapping.
    Args:
        df: The DataFrame to standardize.
        expected_mapping: A dictionary defining standard names and their possible variations.
        df_name: A string indicating the DataFrame type (e.g., "Orders", "Transactions") for logging.
    Returns:
        The DataFrame with standardized column names.
    """
    column_map = {}
    for standard_name, candidates in expected_mapping.items():
        mapped_col = _auto_map_column(df.columns.tolist(), candidates)
        if mapped_col:
            column_map[mapped_col] = standard_name
        else:
            print(f"Warning: Could not find a suitable column for '{standard_name}' in {df_name} DataFrame.")
    
    # Rename columns that were successfully mapped
    df_standardized = df.rename(columns=column_map)
    
    # Check if any critical columns are missing after mapping (optional, but good for robustness)
    # For this initial step, we'll rely on downstream nodes to handle missing columns gracefully
    # or raise errors if they are critical.
    
    return df_standardized

# --- Data Type Conversion (from input.py) ---
def convert_data_types(orders_df: pd.DataFrame, transactions_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts date and numeric fields to proper formats.
    Args:
        orders_df: DataFrame containing order data.
        transactions_df: DataFrame containing transaction data.
    Returns:
        Tuple of processed orders and transactions DataFrames.
    """
    print("Converting data types...")
    # Convert 'Purchase Date' in transactions_df
    if 'Purchase Date' in transactions_df.columns:
        transactions_df['Purchase Date'] = pd.to_datetime(transactions_df['Purchase Date'], dayfirst=True, errors='coerce')
    else:
        print("Warning: 'Purchase Date' not found in transactions_df for type conversion.")

    # Convert 'Return Date' in orders_df
    if 'Return Date' in orders_df.columns:
        orders_df['Return Date'] = pd.to_datetime(orders_df['Return Date'], dayfirst=True, errors='coerce')
    else:
        print("Warning: 'Return Date' not found in orders_df for type conversion.")

    # Convert numeric columns in orders_df
    numeric_columns = ['Unit Price', 'Total Amount', 'Discount Value', 'Shipping Cost', 'Total Payable', 'Quantity']
    for col in numeric_columns:
        if col in orders_df.columns:
            orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce')
    
    # Convert numeric columns in transactions_df (assuming 'Total Amount' is the main one)
    if 'Total Amount' in transactions_df.columns:
        transactions_df['Total Amount'] = pd.to_numeric(transactions_df['Total Amount'], errors='coerce')

    return orders_df, transactions_df

# --- Merge Logic (from streamlit_ui.py's process_data) ---
def merge_orders_transactions(orders_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges user_id from transactions into orders based on transaction_id.
    Args:
        orders_df: DataFrame containing order data.
        transactions_df: DataFrame containing transaction data.
    Returns:
        Orders DataFrame with 'User ID' merged in.
    """
    print("Merging orders and transactions...")
    if 'Transaction ID' in transactions_df.columns and \
       'Transaction ID' in orders_df.columns and \
       'User ID' in transactions_df.columns:
        
        # Select only necessary columns from transactions_df to avoid accidental merges
        # Ensure 'User ID' is present in transactions_df before selecting
        if 'User ID' in transactions_df.columns:
            df_orders_merged = orders_df.merge(
                transactions_df[['Transaction ID', 'User ID']],
                on='Transaction ID',
                how='left'
            )
        else:
            print("Warning: 'User ID' not found in transactions_df for merge. Skipping User ID merge.")
            df_orders_merged = orders_df.copy() # Return copy if merge cannot happen
    else:
        print("Warning: 'Transaction ID' or 'User ID' not found in both orders and transactions for merging. Skipping merge.")
        df_orders_merged = orders_df.copy() # Return copy if merge cannot happen
        
    return df_orders_merged

# --- Initial Data Load (for Streamlit to call directly) ---
# This function is designed to be called directly by Streamlit,
# not as a Kedro node in the pipeline. It simulates loading raw data.
def load_raw_data_for_streamlit(orders_file_path: str, transactions_file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw data from specified paths.
    This function is intended to be called directly from Streamlit,
    not as a Kedro pipeline node.
    """
    print(f"Loading orders from: {orders_file_path}")
    df_orders = pd.read_csv(orders_file_path)
    print(f"Loading transactions from: {transactions_file_path}")
    df_transactions = pd.read_csv(transactions_file_path)
    return df_orders, df_transactions

# --- Column Mappings (from mapping.py) ---
# These are moved here for direct access by nodes and Streamlit
expected_orders_cols = {
    "Transaction ID": ["Transaction ID", "transaction_id"],
    "Order ID": ["Order ID", "order_id"],
    "Product ID": ["Product ID", "product_id", "SKU", "Item Code"],
    "Quantity": ["Quantity", "Qty", "order_quantity"],
    "Total Amount": ["Total Amount", "total_amount", "amount"],
    "Unit Price": ["unit_price", "price"],
    "Order Date": ["Order Date", "order_date", "Order_date"],
    "Discount Code Used": ["Discount Code Used", "discount_code_used", "promo_code"],
    "Discount Value": ["Discount Value", "discount_value", "discount_amount"],
    "Shipping Cost": ["Shipping Cost", "shipping_cost", "freight"],
    "Total Payable": ["Total Payable", "total_payable", "amount_payable"],
    "Return Status": ["Return Status", "return_stat", "is_returned"],
    "Return Date": ["Return Date", "return_date"]
}

expected_transaction_cols = {
    "Transaction ID": ["Transaction ID", "transaction_id"],
    "Visit ID": ["Visit ID", "visit_id"],
    "User ID": ["User ID", "user_id", "Customer ID"],
    "Order ID": ["Order ID", "order_id"],
    "Purchase Date": ["Purchase Date", "purchase_date", "Transaction Date"],
    "Payment Method": ["Payment Method", "payment_method", "Mode of Payment"],
    "Total Amount": ["Total Payable", "total_payable", "amount_payable", "Total_amount"]
}
