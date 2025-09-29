import pandas as pd
import difflib
from typing import Tuple

# --- Helper function for column mapping --- 
def _auto_map_column(column_list, candidate_names):
    
    for name in candidate_names:
        match = difflib.get_close_matches(name, column_list, n=1, cutoff=0.75)
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
        # 'Session Total Cost',
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

    behavioral_df.to_csv('bh_check.csv')

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

def aggregate_behavioral_customer_level(behavioral_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the behavioral dataset at the Customer ID level.
    Handles missing columns gracefully (skips them if not present).
    Produces cleaner, business-friendly column names.
    """

    print("Aggregating behavioral data at customer level...")

    # Define aggregation logic
    agg_map = {
        "Visit ID": "nunique",
        "Session ID": "nunique",
        "Visit Timestamp": ["min", "max"],
        "Channel": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "Geo Location": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "Device ID": "nunique",
        "Device Type": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "Cookie ID": "nunique",
        "OS": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "Entry Page": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "Exit Page": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "Sponsored Listing Viewed": "sum",
        "Banner Viewed": "sum",
        "Homepage Promo Seen": "sum",
        "Product Search View": "sum",
        # "Session Total Cost": ["sum", "mean"],
        "Session Duration": ["sum", "mean"],
        "Page Views": ["sum", "mean"],
        "Bounce Flag": ["sum", "mean"],
        "Ad Campaign ID": lambda x: list(set(x.dropna())),
    }

    # Filter aggregation map to only include columns present in the dataframe
    available_agg_map = {
        col: agg for col, agg in agg_map.items() if col in behavioral_df.columns
    }

    if "Customer ID" not in behavioral_df.columns:
        raise ValueError("Customer ID column is required for aggregation.")

    # Perform aggregation
    agg_df = behavioral_df.groupby("Customer ID").agg(available_agg_map).reset_index()

    # Flatten multi-level column names
    agg_df.columns = [
        " ".join(col).strip() if isinstance(col, tuple) else col
        for col in agg_df.columns.values
    ]

    # Create a nicer mapping for column names
    rename_map = {
        "Visit ID nunique": "Total Unique Visits",
        "Session ID nunique": "Total Unique Sessions",
        "Visit Timestamp min": "First Visit Timestamp",
        "Visit Timestamp max": "Last Visit Timestamp",
        "Visit Timestamp count": "Total Visits",
        "Device ID nunique": "Total Unique Devices",
        "Cookie ID nunique": "Total Unique Cookies",
        "Sponsored Listing Viewed sum": "Total Sponsored Listings Viewed",
        "Banner Viewed sum": "Total Banners Viewed",
        "Homepage Promo Seen sum": "Total Homepage Promos Seen",
        "Product Search View sum": "Total Product Searches Viewed",
        "Session Total Cost sum": "Total Session Cost",
        "Session Total Cost mean": "Avg Session Cost",
        "Session Duration sum": "Total Session Duration",
        "Session Duration mean": "Avg Session Duration",
        "Page Views sum": "Total Page Views",
        "Page Views mean": "Avg Page Views",
        "Bounce Flag sum": "Total Bounces",
        "Bounce Flag mean": "Bounce Rate",
        "Channel <lambda>": "Channel",
        "Geo Location <lambda>": "Geo Location",
        "Device Type <lambda>":"Device Type",
        "Exit Page <lambda>": "Exit Page"


    }
    agg_df.to_csv('behv_cehck_2.csv')

    # Apply renaming only for existing columns
    agg_df = agg_df.rename(columns={k: v for k, v in rename_map.items() if k in agg_df.columns})
    print(agg_df.info())
    return agg_df

def aggregate_orders_transactions_customer_level(
    orders_txn_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates merged orders+transactions dataset at the User ID (customer) level.
    Focuses on spend and counts (transactions, orders, unique products).
    """

    print("Aggregating orders+transactions at customer level...")

    if "User ID" not in orders_txn_df.columns:
        raise ValueError("'User ID' column is required for customer-level aggregation.")

    # Define aggregation logic (only what we need)
    agg_map = {
        "Transaction ID": "nunique",          # total unique transactions
        "Order ID": "nunique",                # total unique orders
        "Product ID": "nunique" if "Product ID" in orders_txn_df.columns else None,
        "Total Amount": "sum" if "Total Amount" in orders_txn_df.columns else None,
        "Total Product Price": "sum" if "Total Product Price" in orders_txn_df.columns else None,
        "Total Payable": "sum" if "Total Payable" in orders_txn_df.columns else None,
        # "Discount Value": "sum" if "Discount Value" in orders_txn_df.columns else None,
        # "Shipping Cost": "sum" if "Shipping Cost" in orders_txn_df.columns else None,
        # "Purchase Date": ["min", "max"] if "Purchase Date" in orders_txn_df.columns else None,
    }

    # Filter out Nones (columns not present in dataframe)
    agg_map = {col: agg for col, agg in agg_map.items() if agg is not None and col in orders_txn_df.columns}

    # Perform aggregation
    agg_df = orders_txn_df.groupby("User ID").agg(agg_map).reset_index()

    # Flatten multi-level columns
    agg_df.columns = [
        " ".join(col).strip() if isinstance(col, tuple) else col
        for col in agg_df.columns.values
    ]

    # Business-friendly rename map
    rename_map = {
        "Transaction ID nunique": "Total Transactions",
        "Order ID nunique": "Total Orders",
        "Product ID nunique": "Total Unique Products",
        "Total Product Price": "Total Order Value",
        "Total Amount sum": "Total Product Sum",
        "Total Payable sum": "Total Payable Value",
        "Discount Value sum": "Total Discounts Availed",
        "Shipping Cost sum": "Total Shipping Paid",
        "Purchase Date min": "First Purchase Date",
        "Purchase Date max": "Last Purchase Date",
    }
    agg_df = agg_df.rename(columns={k: v for k, v in rename_map.items() if k in agg_df.columns})
    print(agg_df.info())
    return agg_df


def merge_customer_ord_txn_behavioral_data(
    orders_txn_customer_df: pd.DataFrame,
    behavioral_agg_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges aggregated customer-level orders+transactions data 
    with aggregated behavioral data.
    Handles missing keys gracefully.
    """

    print("Merging aggregated customer-level orders+transactions with behavioral data...")

    # Ensure ID columns are present
    if "User ID" not in orders_txn_customer_df.columns:
        print("Warning: 'User ID' not found in orders+transactions customer-level dataset. Skipping merge.")
        return orders_txn_customer_df

    if "Customer ID" not in behavioral_agg_df.columns:
        print("Warning: 'Customer ID' not found in aggregated behavioral dataset. Skipping merge.")
        return behavioral_agg_df

    # Perform merge (User ID, Customer ID)
    merged_df = orders_txn_customer_df.merge(
        behavioral_agg_df,
        left_on="User ID",
        right_on="Customer ID",
        how="left"
    )
    merged_df.to_csv('merged_finl_check.csv')
    print(merged_df.info())
    return merged_df