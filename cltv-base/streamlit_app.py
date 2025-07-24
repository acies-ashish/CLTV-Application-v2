# streamlit_app.py (Place this in the root of your Kedro project)

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import shutil # Needed for copying sample files to the fixed input location

# Import Kedro specific components
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Define paths based on Kedro's data structure at the top level
KEDRO_PROJECT_ROOT = Path(__file__).parent
DATA_00_EXTERNAL = KEDRO_PROJECT_ROOT / "data" / "00_external"
DATA_01_RAW = KEDRO_PROJECT_ROOT / "data" / "01_raw"
DATA_02_INTERMEDIATE = KEDRO_PROJECT_ROOT / "data" / "02_intermediate"

# Ensure directories exist
DATA_00_EXTERNAL.mkdir(parents=True, exist_ok=True)
DATA_01_RAW.mkdir(parents=True, exist_ok=True)
DATA_02_INTERMEDIATE.mkdir(parents=True, exist_ok=True)

# Define the FIXED target paths for raw data that Kedro will read from
# Streamlit will ensure data is placed here.
FIXED_ORDERS_RAW_PATH = DATA_00_EXTERNAL / "current_orders_data.csv"
FIXED_TRANSACTIONS_RAW_PATH = DATA_00_EXTERNAL / "current_transactions_data.csv"

# Sample data paths (assuming they are in data/01_raw as per Kedro convention)
SAMPLE_ORDER_PATH = DATA_01_RAW / "Orders_v2.csv"
SAMPLE_TRANS_PATH = DATA_01_RAW / "Transactional_v2.csv"


# Import nodes for direct access to constants like expected_orders_cols
try:
    # Note: The project name created by `kedro new` was 'customer-analytics'.
    # If you renamed your project folder to 'cltv_base', you would need to
    # update this import path accordingly (e.g., from src.cltv_base.nodes).
    # For now, assuming the generated project structure remains 'customer_analytics'.
    from src.cltv_base.nodes import (
        expected_orders_cols, # Directly import mappings
        expected_transaction_cols # Directly import mappings
    )
except ImportError as e:
    st.error(f"Error importing Kedro nodes. Make sure you are running Streamlit from the Kedro project root and your Kedro environment is set up. Error: {e}")
    st.stop() # Stop the app if imports fail

# Bootstrap Kedro project once
bootstrap_project(KEDRO_PROJECT_ROOT)

@st.cache_data(show_spinner=False)
def run_kedro_preprocessing_pipeline():
    """
    Executes the Kedro preprocessing pipeline.
    It expects raw data to be already placed at FIXED_ORDERS_RAW_PATH and FIXED_TRANSACTIONS_RAW_PATH.
    """
    st.info("Initiating Kedro pipeline run...")
    try:
        with KedroSession.create(project_path=KEDRO_PROJECT_ROOT) as session:
            # Explicitly load the context to ensure catalog is available
            context = session.load_context()
            
            # The pipeline runs normally, reading from the fixed paths defined in catalog.yml
            session.run(pipeline_name="preprocessing_pipeline")
            
            # Load the processed data from the catalog after the pipeline runs
            # Access the catalog through the context
            df_orders_merged = context.catalog.load("orders_merged_with_user_id")
            df_transactions_typed = context.catalog.load("transactions_typed")
            
        st.success("✅ Kedro preprocessing pipeline completed successfully!")
        return df_orders_merged, df_transactions_typed
    except Exception as e:
        st.error(f"❌ Error running Kedro pipeline: {e}")
        return None, None

def run_streamlit_app():
    st.set_page_config(page_title="CLTV Dashboard", layout="wide")
    st.title("Customer Lifetime Value Dashboard - Preprocessing Demo")

    tab1, tab2 = st.tabs(["Upload / Load Data", "Processed Data View"])

    with tab1:
        st.subheader("Data Source Selection")
        orders_file = st.file_uploader("Upload Orders CSV", type=["csv"], key="orders_upload")
        transactions_file = st.file_uploader("Upload Transactions CSV", type=["csv"], key="transactions_upload")

        use_sample_data = st.button("🚀 Use Sample Data Instead", key="use_sample_button")

        if orders_file and transactions_file:
            st.session_state['data_source'] = 'uploaded'
            st.session_state['orders_file_obj'] = orders_file
            st.session_state['transactions_file_obj'] = transactions_file
            st.success("Files uploaded. Click 'Process Data' to continue.")
        elif use_sample_data:
            st.session_state['data_source'] = 'sample'
            st.success("Using sample data. Click 'Process Data' to continue.")

        if st.button("⚙️ Process Data (Run Kedro Pipeline)", key="process_data_button"):
            if 'data_source' not in st.session_state:
                st.warning("Please upload files or select sample data first.")
                return

            with st.spinner("Preparing data for Kedro..."):
                try:
                    if st.session_state['data_source'] == 'uploaded':
                        # Save uploaded files to the fixed external data directory for Kedro to read
                        with open(FIXED_ORDERS_RAW_PATH, "wb") as f:
                            f.write(st.session_state['orders_file_obj'].getbuffer())
                        with open(FIXED_TRANSACTIONS_RAW_PATH, "wb") as f:
                            f.write(st.session_state['transactions_file_obj'].getbuffer())
                        
                        st.info(f"Uploaded files saved to {FIXED_ORDERS_RAW_PATH} and {FIXED_TRANSACTIONS_RAW_PATH}")

                    elif st.session_state['data_source'] == 'sample':
                        # Copy sample files to the fixed external data directory for Kedro to read
                        shutil.copy(SAMPLE_ORDER_PATH, FIXED_ORDERS_RAW_PATH)
                        shutil.copy(SAMPLE_TRANS_PATH, FIXED_TRANSACTIONS_RAW_PATH)
                        st.info(f"Sample files copied to {FIXED_ORDERS_RAW_PATH} and {FIXED_TRANSACTIONS_RAW_PATH}")
                    
                    st.session_state['preprocessing_triggered'] = True
                    st.rerun() # Using st.rerun()
                except Exception as e:
                    st.error(f"❌ Error preparing data: {e}")
                    st.session_state['preprocessing_triggered'] = False

    # This block runs after the 'Process Data' button is clicked and reruns the app
    if st.session_state.get('preprocessing_triggered'):
        # Call the Kedro pipeline without passing dynamic paths
        df_orders_merged, df_transactions_typed = run_kedro_preprocessing_pipeline()
        st.session_state['df_orders_merged'] = df_orders_merged
        st.session_state['df_transactions_typed'] = df_transactions_typed
        st.session_state['preprocessing_done'] = True
        st.session_state['preprocessing_triggered'] = False # Reset trigger

    with tab2:
        st.subheader("Preview of Preprocessed Data")
        if st.session_state.get('preprocessing_done') and st.session_state['df_orders_merged'] is not None:
            st.write("#### `orders_merged_with_user_id.parquet` (Orders with User ID)")
            st.dataframe(st.session_state['df_orders_merged'].head())
            st.write(f"Shape: {st.session_state['df_orders_merged'].shape}")

            st.write("#### `transactions_typed.parquet` (Transactions with correct types)")
            st.dataframe(st.session_state['df_transactions_typed'].head())
            st.write(f"Shape: {st.session_state['df_transactions_typed'].shape}")
            
            st.info(f"You can find these files in your Kedro project's `data/02_intermediate` directory.")
        else:
            st.warning("Please process data in the 'Upload / Load Data' tab first.")

if __name__ == "__main__":
    run_streamlit_app()
