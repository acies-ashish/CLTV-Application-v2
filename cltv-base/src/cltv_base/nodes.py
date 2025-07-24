# src/cltv_base/nodes.py

import pandas as pd
import difflib
import os
from datetime import datetime, timedelta
from typing import Tuple, Dict, List # Import necessary types
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# --- Helper functions (from mapping.py and original streamlit_ui.py) ---
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
    
    return df_standardized

# --- Data Type Conversion (from input.py) ---
def convert_data_types(orders_df: pd.DataFrame, transactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

# --- Initial Data Load (for Streamlit to call directly) ---
# This function is designed to be called directly by Streamlit,
# not as a Kedro node in the pipeline. It simulates loading raw data.
def load_raw_data_for_streamlit(orders_file_path: str, transactions_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

# --- Operations from original operations.py ---

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

    customer_level['aov'] = round(customer_level['monetary'] / customer_level['frequency'], 2)
    
    customer_level['avg_days_between_orders'] = (
        (customer_level['last_purchase'] - customer_level['first_purchase']).dt.days / 
        (customer_level['frequency'] - 1)
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
    Performs RFM segmentation on customer-level data.
    This function acts as a Kedro node.
    """
    print("Performing RFM segmentation...")
    df = customer_level_df.copy()
    df['R_score'] = pd.qcut(df['recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['F_score'] = pd.qcut(df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['M_score'] = pd.qcut(df['monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    df['RFM_score'] = df['R_score'] + df['F_score'] + df['M_score']
    df['RFM'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)

    q1 = df['RFM_score'].quantile(0.33)
    q2 = df['RFM_score'].quantile(0.66)

    def assign_segment(score):
        if score <= q1:
            return 'Low'
        elif score <= q2:
            return 'Medium'
        else:
            return 'High'

    df['segment'] = df['RFM_score'].apply(assign_segment)
    return df

def calculate_historical_cltv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates historical CLTV based on AOV and frequency.
    This function acts as a Kedro node.
    """
    print("Calculating historical CLTV...")
    df['CLTV'] = df['aov'] * df['frequency']
    return df

def get_customers_at_risk(customer_level_df: pd.DataFrame, threshold_days: int) -> pd.DataFrame:
    """
    Identifies customers at risk based on recency.
    This function acts as a Kedro node.
    """
    print(f"Identifying customers at risk (recency > {threshold_days} days)...")
    return customer_level_df[customer_level_df['recency'] > threshold_days]

def label_churned_customers(customer_df: pd.DataFrame, inactive_days_threshold: int) -> pd.DataFrame:
    """
    Labels customers as churned based on recency.
    This function acts as a Kedro node.
    """
    print(f"Labeling churned customers (recency > {inactive_days_threshold} days)...")
    df = customer_df.copy()
    df['is_churned'] = (df['recency'] > inactive_days_threshold).astype(int)
    return df

def get_churn_features_labels(customer_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]: # Changed return type to pd.DataFrame for y
    """
    Extracts features and labels for churn prediction.
    This function acts as a Kedro node.
    """
    print("Extracting churn features and labels...")
    feature_cols = [
        'frequency', 'monetary', 'aov',
        'avg_days_between_orders', 'CLTV_30d', 'CLTV_60d', 'CLTV_90d'
    ]
    # Ensure all feature columns exist before selection
    existing_feature_cols = [col for col in feature_cols if col in customer_df.columns]
    if len(existing_feature_cols) < len(feature_cols):
        missing_cols = set(feature_cols) - set(existing_feature_cols)
        print(f"Warning: Missing churn feature columns: {missing_cols}. Using available features.")
    
    X = customer_df[existing_feature_cols]
    y = customer_df['is_churned']
    return X, y.to_frame() # Convert y Series to DataFrame here

def prepare_survival_data(customer_df: pd.DataFrame, churn_threshold: int) -> pd.DataFrame:
    """
    Adds 'duration' and 'event' for survival analysis.
    This function acts as a Kedro node.
    """
    print("Preparing survival data...")
    df = customer_df.copy()
    df['duration'] = df['lifespan_1d']
    df['event'] = (df['recency'] > churn_threshold).astype(int)
    return df

# --- Modeling Functions (from cltv_model.py, churn_model.py, cox_model.py) ---

def predict_cltv_bgf_ggf(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits BG/NBD and Gamma-Gamma models to predict 3-month CLTV.
    This function acts as a Kedro node.
    """
    print("Predicting CLTV using BG/NBD + Gamma-Gamma models...")
    df = transactions_df.copy()
    
    if 'Purchase Date' not in df.columns or 'User ID' not in df.columns or 'Total Amount' not in df.columns:
        raise KeyError("Missing required columns ('Purchase Date', 'User ID', 'Total Amount') in the transaction dataset for CLTV prediction.")

    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])

    summary_df = summary_data_from_transaction_data(
        df,
        customer_id_col='User ID',
        datetime_col='Purchase Date',
        monetary_value_col='Total Amount',
        observation_period_end=df['Purchase Date'].max()
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
        time=3, 
        freq='D', 
        discount_rate=0.01
    )

    summary_df = summary_df.reset_index()
    return summary_df[['User ID', 'predicted_cltv_3m']]

def train_churn_prediction_model(X: pd.DataFrame, y: pd.DataFrame, n_estimators: int, random_state: int) -> Tuple[RandomForestClassifier, Dict, List, pd.DataFrame, pd.DataFrame]: # Changed y to pd.DataFrame
    """
    Trains a RandomForestClassifier for churn prediction.
    This function acts as a Kedro node.
    """
    print("Training churn prediction model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random_state)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train.values.ravel()) # .values.ravel() to convert DataFrame to 1D array for fit
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    importances = model.feature_importances_.tolist()
    return model, report, importances, X_test, y_test

def predict_churn_probabilities(model: RandomForestClassifier, X: pd.DataFrame) -> pd.DataFrame: # Changed return type to pd.DataFrame
    """
    Predicts churn probabilities using the trained model.
    This function acts as a Kedro node.
    """
    print("Predicting churn probabilities...")
    # Return as DataFrame with 'User ID' as index for merging later
    return pd.DataFrame(model.predict_proba(X)[:, 1], index=X.index, columns=['predicted_churn_prob'])

def assign_predicted_churn_labels(predicted_churn_prob: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame: # Changed input/output to pd.DataFrame
    """
    Assigns binary churn labels based on a probability threshold.
    This function acts as a Kedro node.
    """
    print("Assigning predicted churn labels...")
    # Ensure it's a DataFrame and rename the column
    return (predicted_churn_prob['predicted_churn_prob'] >= threshold).astype(int).to_frame(name='predicted_churn')

def train_cox_survival_model(customer_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[CoxPHFitter, pd.DataFrame]:
    """
    Trains Cox Proportional Hazards model for churn time prediction.
    Returns trained model and predicted expected churn times.
    This function acts as a Kedro node.
    """
    print("Training Cox survival model...")
    df = customer_df.copy()
    cph = CoxPHFitter()

    required_cols = feature_cols + ['duration', 'event']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for Cox model training: {missing}")

    survival_df = df[required_cols]
    cph.fit(survival_df, duration_col='duration', event_col='event')

    df['expected_active_days'] = cph.predict_expectation(survival_df).round(0).astype(int)
    return cph, df

# --- UI Data Preparation Nodes (New functions to prepare data for Streamlit) ---

def prepare_kpi_data(orders_df: pd.DataFrame, rfm_segmented_df: pd.DataFrame, transactions_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates key performance indicators for the insights tab.
    """
    print("Preparing KPI data...")
    kpis = {}

    # Total Revenue
    if 'Quantity' in orders_df.columns and 'Unit Price' in orders_df.columns:
        orders_df['Revenue'] = orders_df['Quantity'] * orders_df['Unit Price']
        kpis['total_revenue'] = float(orders_df['Revenue'].sum()) # Convert to float
    else:
        kpis['total_revenue'] = 0.0
        print("Warning: Missing 'Quantity' or 'Unit Price' in orders data for Total Revenue KPI.")

    # CLTV, AOV, Avg Txns/User
    if 'aov' in rfm_segmented_df.columns and 'CLTV' in rfm_segmented_df.columns and 'frequency' in rfm_segmented_df.columns:
        kpis['avg_cltv'] = float(rfm_segmented_df['CLTV'].mean())
        kpis['avg_aov'] = float(rfm_segmented_df['aov'].mean())
        kpis['avg_txns_per_user'] = float(rfm_segmented_df['frequency'].mean())
    else:
        kpis['avg_cltv'] = 0.0
        kpis['avg_aov'] = 0.0
        kpis['avg_txns_per_user'] = 0.0
        print("Warning: Missing RFM columns in segmented data for CLTV, AOV, Avg Txns/User KPIs.")

    # Data Timeframe
    if 'Order Date' in orders_df.columns:
        start_dt = pd.to_datetime(orders_df['Order Date']).min()
        end_dt = pd.to_datetime(orders_df['Order Date']).max()
        kpis['start_date'] = format_date_with_ordinal(start_dt)
        kpis['end_date'] = format_date_with_ordinal(end_dt)
    else:
        kpis['start_date'] = "N/A"
        kpis['end_date'] = "N/A"
        print("Warning: Missing 'Order Date' in orders data for Data Timeframe KPI.")

    kpis['total_customers'] = int(len(rfm_segmented_df)) # Convert to int
    if 'segment' in rfm_segmented_df.columns:
        kpis['high_value_customers'] = int((rfm_segmented_df['segment'] == 'High').sum()) # Convert to int
        kpis['mid_value_customers'] = int((rfm_segmented_df['segment'] == "Medium").sum()) # Convert to int
        kpis['low_value_customers'] = int((rfm_segmented_df['segment'] == "Low").sum()) # Convert to int
    else:
        kpis['high_value_customers'] = 0
        kpis['mid_value_customers'] = 0
        kpis['low_value_customers'] = 0
    
    return kpis

def prepare_segment_summary_data(rfm_segmented_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates segment-wise summary metrics for visualization.
    """
    print("Preparing segment summary data...")
    if 'segment' not in rfm_segmented_df.columns:
        print("Warning: 'segment' column not found in rfm_segmented_df. Cannot prepare segment summary.")
        return pd.DataFrame()

    segment_summary = rfm_segmented_df.groupby("segment").agg({
        "aov": "mean",
        "CLTV": "mean",
        "frequency": "mean",
        "avg_days_between_orders": "mean",
        "recency": "mean",
        "monetary": "mean"
    }).round(2)
    return segment_summary

def prepare_segment_counts_data(rfm_segmented_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the distribution of customers across RFM segments for the pie chart.
    """
    print("Preparing segment counts data...")
    if 'segment' not in rfm_segmented_df.columns:
        print("Warning: 'segment' column not found in rfm_segmented_df. Cannot prepare segment counts.")
        return pd.DataFrame(columns=['Segment', 'Count'])
    
    segment_counts = rfm_segmented_df['segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    return segment_counts


def prepare_top_products_by_segment_data(orders_df: pd.DataFrame, transactions_df: pd.DataFrame, rfm_segmented_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculates top products by revenue for each customer segment.
    Returns a dictionary where keys are segment names and values are DataFrames of top products.
    """
    print("Preparing top products by segment data...")
    top_products_by_segment = {}
    segments = ['High', 'Medium', 'Low']

    # Standardize orders_df columns for product calculation
    orders_for_products = orders_df.copy()
    orders_for_products.columns = [c.strip().replace(" ", "_").lower() for c in orders_for_products.columns]
    if 'unit_price' in orders_for_products.columns:
        orders_for_products.rename(columns={'unit_price': 'unitprice'}, inplace=True)
    
    required_product_cols = {'transaction_id', 'product_id', 'quantity', 'unitprice'}
    if not required_product_cols.issubset(set(orders_for_products.columns)):
        print(f"Warning: Missing required columns for top products: {required_product_cols - set(orders_for_products.columns)}")
        return {} # Return empty dict if critical columns are missing

    for segment in segments:
        if 'segment' in rfm_segmented_df.columns and 'User ID' in rfm_segmented_df.columns:
            segment_users = rfm_segmented_df[rfm_segmented_df['segment'] == segment]['User ID']
            segment_transaction_ids = transactions_df[transactions_df['User ID'].isin(segment_users)]['Transaction ID']

            filtered_orders = orders_for_products[orders_for_products['transaction_id'].isin(segment_transaction_ids)].copy()
            if not filtered_orders.empty:
                filtered_orders['revenue'] = filtered_orders['quantity'] * filtered_orders['unitprice']

                top_products = (
                    filtered_orders.groupby('product_id')
                    .agg(Total_Quantity=('quantity', 'sum'), Total_Revenue=('revenue', 'sum'))
                    .sort_values(by='Total_Revenue', ascending=False)
                    .head(5)
                    .reset_index()
                )
                top_products_by_segment[segment] = top_products
            else:
                top_products_by_segment[segment] = pd.DataFrame(columns=['product_id', 'Total_Quantity', 'Total_Revenue'])
        else:
            print(f"Warning: RFM segmented data or User ID not available for segment {segment}.")
            top_products_by_segment[segment] = pd.DataFrame(columns=['product_id', 'Total_Quantity', 'Total_Revenue'])

    return top_products_by_segment

def prepare_predicted_cltv_display_data(rfm_segmented_df: pd.DataFrame, predicted_cltv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares data for the predicted CLTV display table.
    """
    print("Preparing predicted CLTV display data...")
    if 'User ID' not in rfm_segmented_df.columns or 'User ID' not in predicted_cltv_df.columns:
        print("Warning: Missing 'User ID' for merging CLTV prediction data.")
        return rfm_segmented_df[['User ID', 'segment', 'CLTV']].copy() # Return basic if merge fails

    # Merge predicted CLTV into the RFM segmented data
    rfm_segmented_df = rfm_segmented_df.merge(predicted_cltv_df, on='User ID', how='left')
    rfm_segmented_df['predicted_cltv_3m'] = rfm_segmented_df['predicted_cltv_3m'].fillna(0)
    
    return rfm_segmented_df[['User ID', 'segment', 'CLTV', 'predicted_cltv_3m']].sort_values(by='predicted_cltv_3m', ascending=False).reset_index(drop=True)

def prepare_cltv_comparison_data(predicted_cltv_display_data: pd.DataFrame) -> pd.DataFrame: # Changed input to predicted_cltv_display_data
    """
    Prepares data for average historical vs predicted CLTV per segment bar chart.
    """
    print("Preparing CLTV comparison data...")
    if not all(col in predicted_cltv_display_data.columns for col in ['segment', 'CLTV', 'predicted_cltv_3m']):
        print("Warning: Missing required CLTV comparison columns. Cannot prepare comparison data.")
        return pd.DataFrame()

    segment_comparison = predicted_cltv_display_data.groupby('segment')[['CLTV', 'predicted_cltv_3m']].mean().reset_index()
    segment_melted = segment_comparison.melt(
        id_vars='segment',
        value_vars=['CLTV', 'predicted_cltv_3m'],
        var_name='CLTV Type',
        value_name='Average CLTV'
    )
    segment_order = ['Low', 'Medium', 'High']
    segment_melted['segment'] = pd.Categorical(segment_melted['segment'], categories=segment_order, ordered=True)
    return segment_melted.sort_values(by='segment')


def calculate_realization_curve_data(orders_df: pd.DataFrame, rfm_segmented_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculates CLTV realization curve data for different customer groups.
    Returns a dictionary where keys are group names and values are DataFrames for the curve.
    """
    print("Calculating realization curve data...")
    realization_data = {}
    
    df = orders_df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if 'unit_price' in df.columns:
        df.rename(columns={'unit_price': 'unitprice'}, inplace=True)
    if 'user_id' not in df.columns and 'user id' in df.columns: # Handle potential variations
        df.rename(columns={'user id': 'user_id'}, inplace=True)

    required_cols = {'order_date', 'quantity', 'unitprice', 'user_id'}
    if not required_cols.issubset(set(df.columns)):
        print(f"Warning: Required columns for realization curve not found: {required_cols - set(df.columns)}")
        return {}

    df['order_date'] = pd.to_datetime(df['order_date'])
    df['revenue'] = df['quantity'] * df['unitprice']

    segment_options = {
        "Overall": rfm_segmented_df['User ID'].unique(), # Use rfm_segmented_df for User IDs
        "High CLTV Users": rfm_segmented_df[rfm_segmented_df['segment'] == 'High']['User ID'].unique(),
        "Mid CLTV Users": rfm_segmented_df[rfm_segmented_df['segment'] == 'Medium']['User ID'].unique(),
        "Low CLTV Users": rfm_segmented_df[rfm_segmented_df['segment'] == 'Low']['User ID'].unique()
    }

    intervals = [15, 30, 45, 60, 90]

    for option_name, selected_users in segment_options.items():
        if len(selected_users) == 0:
            realization_data[option_name] = pd.DataFrame(columns=["Period (Days)", "Avg CLTV per User"])
            continue

        filtered_df = df[df['user_id'].isin(selected_users)]
        user_count = filtered_df['user_id'].nunique()

        if user_count == 0:
            realization_data[option_name] = pd.DataFrame(columns=["Period (Days)", "Avg CLTV per User"])
            continue

        # Calculate start_date based on filtered_df, not overall df
        start_date = filtered_df['order_date'].min()
        if pd.isna(start_date): # Handle case where filtered_df might be empty after date filter
            realization_data[option_name] = pd.DataFrame(columns=["Period (Days)", "Avg CLTV per User"])
            continue

        cltv_values = []

        for days in intervals:
            cutoff = start_date + pd.Timedelta(days=days)
            revenue = filtered_df[filtered_df['order_date'] <= cutoff]['revenue'].sum()
            avg_cltv = revenue / user_count
            cltv_values.append(round(avg_cltv, 2))

        realization_data[option_name] = pd.DataFrame({
            "Period (Days)": intervals,
            "Avg CLTV per User": cltv_values
        })
    return realization_data

def prepare_churn_summary_data(rfm_segmented_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares data for churn summary metrics and expected active days by segment.
    """
    print("Preparing churn summary data...")
    if 'predicted_churn' not in rfm_segmented_df.columns or 'predicted_churn_prob' not in rfm_segmented_df.columns or 'expected_active_days' not in rfm_segmented_df.columns:
        print("Warning: Missing churn prediction columns for churn summary.")
        return pd.DataFrame(), pd.DataFrame()

    churn_by_segment = (
        rfm_segmented_df
        .groupby("segment")['predicted_churn_prob']
        .mean()
        .reset_index()
        .rename(columns={'predicted_churn_prob': 'Avg Churn Probability'})
    )

    active_days = (
        rfm_segmented_df
        .groupby("segment")['expected_active_days']
        .mean()
        .reset_index()
        .rename(columns={'expected_active_days': 'Avg Expected Active Days'})
    )
    return churn_by_segment, active_days

def prepare_churn_detailed_view_data(rfm_segmented_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares detailed churn analysis data for display.
    """
    print("Preparing detailed churn view data...")
    required_cols = ['User ID', 'segment', 'predicted_cltv_3m', 'predicted_churn_prob', 'predicted_churn', 'expected_active_days']
    if not all(col in rfm_segmented_df.columns for col in required_cols):
        print("Warning: Missing required columns for detailed churn view.")
        # Return an empty DataFrame with expected columns to avoid downstream errors
        return pd.DataFrame(columns=required_cols)

    return rfm_segmented_df[required_cols].sort_values(by='predicted_churn_prob', ascending=False).copy()

# Helper function to add ordinal suffix (from original streamlit_ui.py)
def format_date_with_ordinal(date):
    if pd.isna(date): # Handle NaT or None
        return "N/A"
    day = int(date.strftime('%d'))
    suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return f"{day}{suffix} {date.strftime('%B %Y')}"

# Helper function to check for duplicate columns (from original streamlit_ui.py)
# This function is not directly used in the Kedro pipeline nodes, but kept here for completeness
# if a similar check were needed in a node.
def has_duplicate_columns(df1, df2):
    return df1.columns.duplicated().any() or df2.columns.duplicated().any()

