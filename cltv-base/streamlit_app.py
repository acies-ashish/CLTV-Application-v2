# streamlit_app.py (Place this in the root of your Kedro project)

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import shutil # Needed for copying sample files to the fixed input location
import plotly.express as px # Keep plotly for chart rendering
import json # For loading JSON datasets
from typing import Dict

# Import Kedro specific components
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Define paths based on Kedro's data structure at the top level
KEDRO_PROJECT_ROOT = Path(__file__).parent
DATA_00_EXTERNAL = KEDRO_PROJECT_ROOT / "data" / "00_external"
DATA_01_RAW = KEDRO_PROJECT_ROOT / "data" / "01_raw"
DATA_02_INTERMEDIATE = KEDRO_PROJECT_ROOT / "data" / "02_intermediate"
DATA_03_PRIMARY = KEDRO_PROJECT_ROOT / "data" / "03_primary" # For models/reports
DATA_04_FEATURE = KEDRO_PROJECT_ROOT / "data" / "04_feature" # New for final UI data

# Ensure directories exist
DATA_00_EXTERNAL.mkdir(parents=True, exist_ok=True)
DATA_01_RAW.mkdir(parents=True, exist_ok=True)
DATA_02_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
DATA_03_PRIMARY.mkdir(parents=True, exist_ok=True)
DATA_04_FEATURE.mkdir(parents=True, exist_ok=True)

# Define the FIXED target paths for raw data that Kedro will read from
FIXED_ORDERS_RAW_PATH = DATA_00_EXTERNAL / "current_orders_data.csv"
FIXED_TRANSACTIONS_RAW_PATH = DATA_00_EXTERNAL / "current_transactions_data.csv"

# Sample data paths (assuming they are in data/01_raw as per Kedro convention)
SAMPLE_ORDER_PATH = DATA_01_RAW / "Orders_v2.csv"
SAMPLE_TRANS_PATH = DATA_01_RAW / "Transactional_v2.csv"

# Bootstrap Kedro project once
bootstrap_project(KEDRO_PROJECT_ROOT)

@st.cache_data(show_spinner=False)
def run_kedro_main_pipeline_and_load_ui_data():
    """
    Executes the main Kedro pipeline and loads all pre-calculated UI data.
    """
    try:
        with KedroSession.create(project_path=KEDRO_PROJECT_ROOT) as session:
            context = session.load_context()
            session.run(pipeline_name="preprocessing_pipeline") 
            
            ui_data = {}
            # The rfm_segmented key now loads the combined final customer data
            ui_data['rfm_segmented'] = context.catalog.load("final_rfm_cltv_churn_data") 
            ui_data['kpi_data'] = context.catalog.load("kpi_data_for_ui")
            ui_data['segment_summary'] = context.catalog.load("segment_summary_data_for_ui")
            ui_data['segment_counts'] = context.catalog.load("segment_counts_data_for_ui")
            ui_data['top_products_by_segment'] = context.catalog.load("top_products_by_segment_data_for_ui")
            ui_data['predicted_cltv_display'] = context.catalog.load("predicted_cltv_display_data_for_ui")
            ui_data['cltv_comparison'] = context.catalog.load("cltv_comparison_data_for_ui")
            ui_data['realization_curve'] = context.catalog.load("realization_curve_data_for_ui")
            ui_data['churn_summary'] = context.catalog.load("churn_summary_data_for_ui")
            ui_data['active_days_summary'] = context.catalog.load("active_days_summary_data_for_ui")
            ui_data['churn_detailed_view'] = context.catalog.load("churn_detailed_view_data_for_ui")
            ui_data['customers_at_risk'] = context.catalog.load("customers_at_risk_df")
            # Also load the original orders and transactions data if needed for UI functions that directly use them
            ui_data['df_orders_merged'] = context.catalog.load("orders_merged_with_user_id")
            ui_data['df_transactions_typed'] = context.catalog.load("transactions_typed")


        return ui_data
    except Exception as e:
        st.error(f"❌ Error running Kedro pipeline or loading UI data: {e}")
        return None

# Helper function to add ordinal suffix (copied from nodes.py)
def format_date_with_ordinal(date):
    if pd.isna(date):
        return "N/A"
    day = int(date.strftime('%d'))
    suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return f"{day}{suffix} {date.strftime('%B %Y')}"

# --- Streamlit UI Rendering Functions (No calculations here) ---

def kpi_card(title, value, color="black"):
    st.markdown(f"""
        <div style="background-color:#aee2fd;
                    padding:18px 12px 14px 12px;
                    border-radius:10px;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
                    min-height:100px;
                    color:black;
                    font-family: 'Segoe UI', sans-serif;
                    text-align:center">
            <div style="font-size:16px; font-weight:600; margin-bottom:6px;">{title}</div>
            <div style="font-size:24px; font-weight:bold; color:{color};">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def show_insights_ui(kpi_data: Dict, segment_summary_data: pd.DataFrame, segment_counts_data: pd.DataFrame, top_products_by_segment_data: Dict[str, pd.DataFrame], df_orders_merged: pd.DataFrame):
    st.subheader("📌 Key KPIs")

    # KPIs are now based on the full dataset as processed by Kedro
    total_revenue = kpi_data.get('total_revenue', 0)
    avg_cltv = kpi_data.get('avg_cltv', 0)
    avg_aov = kpi_data.get('avg_aov', 0)
    avg_txns_per_user = kpi_data.get('avg_txns_per_user', 0)
    start_date_kpi = kpi_data.get('start_date', "N/A") # Now directly from kpi_data, which uses full data
    end_date_kpi = kpi_data.get('end_date', "N/A")    # Now directly from kpi_data, which uses full data
    total_customers = kpi_data.get('total_customers', 0)
    
    # Removed specific segment KPIs as segmentation is now more granular
    # loyalty_leaders = kpi_data.get('loyalty_leaders', 0)
    # active_shoppers = kpi_data.get('active_shoppers', 0)
    # new_discoverers = kpi_data.get('new_discoverers', 0)


    row1 = st.columns(3, gap="small")
    with row1[0]: kpi_card("📈 Total Revenue", f"₹{total_revenue:,.0f}", color="black")
    with row1[1]: kpi_card("💰 CLTV", f"₹{avg_cltv:,.0f}")
    with row1[2]: kpi_card("🛒 Avg Order Value", f"₹{avg_aov:.0f}")
    row2 = st.columns(3, gap="small")
    with row2[0]: st.text('')
    with row2[1]: st.text('')
    with row2[2]: st.text('')

    row3 = st.columns(3, gap="small")
    with row3[0]: kpi_card("📦 Avg Transactions/User", f"{avg_txns_per_user:.0f}")
    with row3[1]: kpi_card("📆 Data Timeframe", f"{start_date_kpi} to {end_date_kpi}", color="black")
    with row3[2]: kpi_card("👥 Total Customers", total_customers, color="black")

    st.divider()
    st.subheader("📈 Visual Insights")

    # Define a color palette for the new 11 segments
    # Using a broader palette or a sequential one if segments have a natural order
    segment_colors = {
        'Champions': '#1f77b4',
        'Loyal Customers': '#2ca02c',
        'Potential Loyalists': '#ff7f0e',
        'Recent Customers': '#d62728',
        'Promising': '#9467bd',
        'Customers Needing Attention': '#8c564b',
        'About to Sleep': '#e377c2',
        'At Risk': '#7f7f7f',
        "Can't Lose Them": '#bcbd22',
        'Hibernating': '#17becf',
        'Lost': '#a52a2a', # Darker red for 'Lost'
        'Unclassified': '#cccccc' # Grey for unclassified
    }

    if not segment_counts_data.empty:
        viz_col1, viz_col2 = st.columns([1, 1.2])
        with viz_col1:
            st.markdown("#### 🎯 Customer Segment Distribution")
            fig1 = px.pie(
                segment_counts_data,
                values='Count',
                names='Segment',
                hole=0.45,
                color='Segment',
                color_discrete_map=segment_colors
            )
            fig1.update_traces(textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig1, use_container_width=True)
            
            # Removed specific segment metrics here as there are too many now
            # metrics_cols = st.columns(3)
            # metrics_cols[0].metric("Loyalty Leaders*", loyalty_leaders)
            # metrics_cols[1].metric("Active Shoppers", active_shoppers)
            # metrics_cols[2].metric("New Discoverers", new_discoverers)
            # st.caption("📌 *Loyalty Leaders refers to users whose **RFM Score is in the top 33%.**")
        
        with viz_col2:
            st.markdown("#### 📊 Segment-wise Summary Metrics")

            # Dynamically get segment order from segment_summary_data if available, otherwise use a default
            # Or, for consistency, define a fixed order for display
            segment_order_display = [
                'Champions', 'Loyal Customers', 'Potential Loyalists', 'Recent Customers',
                'Promising', 'Customers Needing Attention', 'About to Sleep', 'At Risk',
                "Can't Lose Them", 'Hibernating', 'Lost', 'Unclassified'
            ]
            
            # Create a list of columns for each segment card
            # Adjust the number of columns based on how many segments you want per row
            num_segments = len(segment_order_display)
            cols_per_row = 3 # You can adjust this
            num_rows = (num_segments + cols_per_row - 1) // cols_per_row

            if not segment_summary_data.empty:
                for i in range(num_rows):
                    cards_row = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        segment_idx = i * cols_per_row + j
                        if segment_idx < num_segments:
                            segment = segment_order_display[segment_idx]
                            with cards_row[j]:
                                if segment in segment_summary_data.index:
                                    metrics = segment_summary_data.loc[segment]
                                    # Use a consistent color mapping for cards
                                    card_color = segment_colors.get(segment, '#aee2fd') # Default light blue if not found
                                    st.markdown(f"""
                                        <div style="
                                            background-color: {card_color};
                                            padding: 20px 15px;
                                            border-radius: 12px;
                                            color: white; /* Text color for cards */
                                            min-height: 250px;
                                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                                            font-family: 'Segoe UI', sans-serif;
                                        ">
                                            <h4 style="text-align: center; margin-bottom: 15px; font-size: 20px; font-weight: 700;">
                                                {segment}
                                            </h4>
                                            <ul style="list-style: none; padding: 0; font-size: 16px; font-weight: 500; line-height: 1.8;">
                                                <li><b>Avg Order Value:</b> ₹{metrics['aov']:,.2f}</li>
                                                <li><b>Avg CLTV:</b> ₹{metrics['CLTV']:,.2f}</li>
                                                <li><b>Avg Txns/User:</b> {metrics['frequency']:,.2f}</li>
                                                <li><b>Days Between Orders:</b> {metrics['avg_days_between_orders']:,.2f}</li>
                                                <li><b>Avg Recency:</b> {metrics['recency']:,.0f} days</li>
                                                <li><b>Monetary Value:</b> ₹{metrics['monetary']:,.2f}</li>
                                            </ul>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.info(f"No data for {segment} segment.")
                        else:
                            st.empty() # Fill remaining columns in the row with empty space
            else:
                st.warning("Segment summary data not available for segment-wise metrics.")
    else:
        st.warning("Customer segment distribution data not available for insights.")


    # 🛍️ Top Products by Segment
    st.divider()
    st.markdown("#### 🛍️ Top Products Bought by Segment Customers")
    if top_products_by_segment_data:
        # Update selectbox options to new segments
        new_segment_options = [
            'Champions', 'Loyal Customers', 'Potential Loyalists', 'Recent Customers',
            'Promising', 'Customers Needing Attention', 'About to Sleep', 'At Risk',
            "Can't Lose Them", 'Hibernating', 'Lost', 'Unclassified'
        ]
        selected_segment = st.selectbox(
            "Choose a Customer Segment",
            options=new_segment_options,
            index=0 if 'Champions' in new_segment_options else 0, # Set default to Champions if available
            key="top_products_segment_select"
        )
        current_segment_products = top_products_by_segment_data.get(selected_segment, pd.DataFrame())

        if not current_segment_products.empty:
            st.markdown(f"#### 📦 Top 5 Products by Revenue for '{selected_segment}' (All Time)")
            fig_products = px.bar(
                current_segment_products,
                x='product_id',
                y='Total_Revenue',
                text='Total_Revenue',
                labels={'product_id': 'Product ID', 'Total_Revenue': 'Revenue'},
                color='product_id',
                color_discrete_sequence = [
                        '#08306b',   # Very Dark Blue
                        '#2171b5',   # Mid Blue
                        '#4292c6',   # Light Blue
                        '#6baed6',   # Softer Blue
                        "#9dcce6"    # Pale Blue
                    ]
            )
            fig_products.update_traces(texttemplate='₹%{text:.2f}', textposition='outside')
            fig_products.update_layout(yaxis_title="Total Revenue", xaxis_title="Product ID")
            st.plotly_chart(fig_products, use_container_width=True)
        else:
            st.info(f"✅ No products found for the '{selected_segment}' segment.")
    else:
        st.warning("Top products by segment data not available.")

def show_prediction_tab_ui(predicted_cltv_display_data: pd.DataFrame, cltv_comparison_data: pd.DataFrame):
    st.subheader("🔮 Predicted CLTV (Next 3 Months)")
    st.caption("Forecasted Customer Lifetime Value using BG/NBD + Gamma-Gamma model.")

    # Table Filter
    if not predicted_cltv_display_data.empty:
        # Update selectbox options to new segments
        new_segment_options = [
            "All", 'Champions', 'Loyal Customers', 'Potential Loyalists', 'Recent Customers',
            'Promising', 'Customers Needing Attention', 'About to Sleep', 'At Risk',
            "Can't Lose Them", 'Hibernating', 'Lost', 'Unclassified'
        ]
        table_segment = st.selectbox(
            "📋 Table Filter by Segment", new_segment_options,
            index=0, key="predicted_cltv_table_segment_filter"
        )

        if table_segment != "All":
            filtered_df = predicted_cltv_display_data[predicted_cltv_display_data['segment'] == table_segment].copy()
        else:
            filtered_df = predicted_cltv_display_data.copy()

        st.dataframe(
            filtered_df.style.format({'CLTV': '₹{:,.2f}', 'predicted_cltv_3m': '₹{:,.2f}'}),
            use_container_width=True
        )
    else:
        st.warning("Predicted CLTV data not available.")

    st.markdown("---")

    # 📊 New Bar Chart: Segment-Wise Avg CLTV Comparison
    st.markdown("#### 📊 Average Historical vs Predicted CLTV per Segment")
    if not cltv_comparison_data.empty:
        # Update chart segment options to new segments
        chart_segment_options = [
            "All", 'Champions', 'Loyal Customers', 'Potential Loyalists', 'Recent Customers',
            'Promising', 'Customers Needing Attention', 'About to Sleep', 'At Risk',
            "Can't Lose Them", 'Hibernating', 'Lost', 'Unclassified'
        ]
        selected_chart_segment = st.selectbox(
            "Filter Chart by Segment",
            options=chart_segment_options,
            index=0, # Default to "All"
            key="cltv_comparison_chart_segment_filter"
        )

        filtered_chart_data = cltv_comparison_data.copy()
        if selected_chart_segment != "All":
            filtered_chart_data = filtered_chart_data[filtered_chart_data['segment'] == selected_chart_segment]

        fig_bar = px.bar(
            filtered_chart_data, # Use filtered data for the chart
            x='segment',
            y='Average CLTV',
            color='CLTV Type',
            barmode='group',
            labels={'segment': 'Customer Segment', 'Average CLTV': 'Avg CLTV (₹)'},
            color_discrete_map={'CLTV': "#32a2f1", 'predicted_cltv_3m': "#3fd33f"},
            title='Average Historical vs Predicted CLTV per Segment'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("CLTV comparison data not available.")


def show_detailed_view_ui(rfm_segmented: pd.DataFrame, customers_at_risk_df: pd.DataFrame):
    st.subheader("📋 Full RFM Segmented Data with CLTV")
    if not rfm_segmented.empty:
        st.dataframe(rfm_segmented)
    else:
        st.warning("RFM Segmented data not available.")

    st.subheader("⚠️ Customers at Risk (Recency > 70 days)")
    st.caption("These are customers whose last purchase was over 70 days ago and may be at risk of churning.")
    if not customers_at_risk_df.empty:
        st.dataframe(customers_at_risk_df)
    else:
        st.info("No customers identified as at risk, or data not available.")

def show_realization_curve_ui(realization_curve_data: Dict[str, pd.DataFrame]):
    st.subheader("📈 Realization Curve of CLTV Over Time")
    if realization_curve_data:
        # Update segment options for the realization curve
        segment_options_list = [
            'All Segments', 'Overall Average', 'Champions', 'Loyal Customers', 'Potential Loyalists',
            'Recent Customers', 'Promising', 'Customers Needing Attention', 'About to Sleep',
            'At Risk', "Can't Lose Them", 'Hibernating', 'Lost', 'Unclassified'
        ]
        segment_option = st.selectbox("Select Customer Group for CLTV Curve",
                                      options=segment_options_list,
                                      index=0, # Set "All Segments" as default
                                      key="realization_curve_segment_select")
        
        chart_df = realization_curve_data.get(segment_option, pd.DataFrame())

        if not chart_df.empty:
            if segment_option == "All Segments":
                color_col = 'Segment'
                # Use the broader segment_colors for "All Segments" view
                color_map = {
                    'Champions': '#1f77b4',
                    'Loyal Customers': '#2ca02c',
                    'Potential Loyalists': '#ff7f0e',
                    'Recent Customers': '#d62728',
                    'Promising': '#9467bd',
                    'Customers Needing Attention': '#8c564b',
                    'About to Sleep': '#e377c2',
                    'At Risk': '#7f7f7f',
                    "Can't Lose Them": '#bcbd22',
                    'Hibernating': '#17becf',
                    'Lost': '#a52a2a',
                    'Unclassified': '#cccccc'
                }
            else:
                color_col = None
                color_map = None

            fig = px.line(
                chart_df,
                x="Period (Days)",
                y="Avg CLTV per User",
                text="Avg CLTV per User",
                markers=True,
                color=color_col,
                color_discrete_map=color_map
            )
            
            fig.update_traces(
                texttemplate='₹%{text:.2f}',
                textposition='top center',
                textfont=dict(size=14, color='black'),
                marker=dict(size=8)
            )
            
            fig.update_layout(
                title={
                    'text': f"CLTV Realization Curve - {segment_option}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': dict(size=20, color='black')
                },
                xaxis=dict(
                    title=dict(text="Days", font=dict(size=16, color='black')),
                    tickfont=dict(size=14, color='black')
                ),
                yaxis=dict(
                    title=dict(text="Avg CLTV", font=dict(size=16, color='black')),
                    tickfont=dict(size=14, color='black')
                ),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No realization curve data available for '{segment_option}'.")
    else:
        st.warning("Realization curve data not available.")

def show_churn_tab_ui(rfm_segmented: pd.DataFrame, churn_summary_data: pd.DataFrame, active_days_summary_data: pd.DataFrame, churn_detailed_view_data: pd.DataFrame):
    st.subheader("📉 Churn Prediction Overview")

    if 'predicted_churn' in rfm_segmented.columns:
        churned = rfm_segmented[rfm_segmented['predicted_churn'] == 1]
        st.metric("Predicted Churned Customers", len(churned))
        # Handle division by zero if rfm_segmented is empty
        churn_rate = (len(churned) / len(rfm_segmented) * 100) if len(rfm_segmented) > 0 else 0.0
        st.metric("Churn Rate (%)", f"{churn_rate:.2f}")
    else:
        st.warning("Churn prediction data not available for overview metrics.")

    st.divider()
    st.markdown("### 📊 Churn Summary by Segment")

    col1, col2 = st.columns(2)

    # Use the same color map as defined for the pie chart and segment cards
    segment_colors_churn = {
        'Champions': '#1f77b4',
        'Loyal Customers': '#2ca02c',
        'Potential Loyalists': '#ff7f0e',
        'Recent Customers': '#d62728',
        'Promising': '#9467bd',
        'Customers Needing Attention': '#8c564b',
        'About to Sleep': '#e377c2',
        'At Risk': '#7f7f7f',
        "Can't Lose Them": '#bcbd22',
        'Hibernating': '#17becf',
        'Lost': '#a52a2a',
        'Unclassified': '#cccccc'
    }

    with col1:
        st.markdown("#### 🔴 Avg Churn Probability")
        if not churn_summary_data.empty:
            fig_churn = px.bar(
                churn_summary_data.sort_values(by='Avg Churn Probability'),
                x='Avg Churn Probability',
                y='segment',
                orientation='h',
                color='segment',
                color_discrete_map=segment_colors_churn, # Use the new color map
                text='Avg Churn Probability'
            )
            fig_churn.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig_churn.update_layout(height=450, margin=dict(t=30)) # Adjusted height for more segments
            st.plotly_chart(fig_churn, use_container_width=True)
        else:
            st.info("Average churn probability data not available.")

    with col2:
        st.markdown("#### ⏳ Avg Expected Active Days")
        if not active_days_summary_data.empty:
            fig_days = px.bar(
                active_days_summary_data.sort_values(by='Avg Expected Active Days'),
                x='Avg Expected Active Days',
                y='segment',
                orientation='h',
                color='segment',
                color_discrete_map=segment_colors_churn, # Use the new color map
                text='Avg Expected Active Days'
            )
            fig_days.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            fig_days.update_layout(height=450, margin=dict(t=30)) # Adjusted height for more segments
            st.plotly_chart(fig_days, use_container_width=True)
        else:
            st.info("Average expected active days data not available.")
    
    st.divider()
    st.markdown("### 🔍 All Customers at a Glance")

    if st.toggle("🕵️ Detailed View of Churn Analysis"):
        if not churn_detailed_view_data.empty:
            st.dataframe(
                churn_detailed_view_data.style.format({'predicted_churn_prob': '{:.2%}', 'predicted_cltv_3m': '₹{:,.2f}'}),
                use_container_width=True
            )
        else:
            st.info("Detailed churn analysis data not available.")


def run_streamlit_app():
    st.set_page_config(page_title="CLTV Dashboard", layout="wide")
    st.title("Customer Lifetime Value Dashboard - Kedro Powered")

    if 'ui_data' not in st.session_state:
        st.session_state['ui_data'] = None
    if 'preprocessing_done' not in st.session_state:
        st.session_state['preprocessing_done'] = False
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Upload / Load Data", "Insights", "Detailed View", "Predictions", "Realization Curve", "Churn" 
    ])

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
            st.session_state['orders_file_obj'] = None
            st.session_state['transactions_file_obj'] = None
            st.success("Using sample data. Click 'Process Data' to continue.")

        if st.button("⚙️ Process Data (Run Kedro Pipeline)", key="process_data_button"):
            st.cache_data.clear() 
            if 'data_source' not in st.session_state:
                st.warning("Please upload files or select sample data first.")
                return

            with st.spinner("Preparing data for Kedro..."):
                try:
                    if st.session_state['data_source'] == 'uploaded':
                        with open(FIXED_ORDERS_RAW_PATH, "wb") as f:
                            f.write(st.session_state['orders_file_obj'].getbuffer())
                        with open(FIXED_TRANSACTIONS_RAW_PATH, "wb") as f:
                            f.write(st.session_state['transactions_file_obj'].getbuffer())
                        
                        st.info(f"Uploaded files saved to {FIXED_ORDERS_RAW_PATH} and {FIXED_TRANSACTIONS_RAW_PATH}")

                    elif st.session_state['data_source'] == 'sample':
                        shutil.copy(SAMPLE_ORDER_PATH, FIXED_ORDERS_RAW_PATH)
                        shutil.copy(SAMPLE_TRANS_PATH, FIXED_TRANSACTIONS_RAW_PATH)
                        st.info(f"Sample files copied to {FIXED_ORDERS_RAW_PATH} and {FIXED_TRANSACTIONS_RAW_PATH}")
                    
                    st.info("Initiating Kedro pipeline run...")

                    st.session_state['preprocessing_triggered'] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error preparing data: {e}")
                    st.session_state['preprocessing_triggered'] = False

    if st.session_state.get('preprocessing_triggered'):
        st.session_state['ui_data'] = run_kedro_main_pipeline_and_load_ui_data()
        st.session_state['preprocessing_done'] = True
        st.session_state['preprocessing_triggered'] = False
        
        if st.session_state['ui_data'] is not None:
            st.success("✅ Kedro pipeline completed successfully and UI data loaded!")

    if st.session_state.get('preprocessing_done') and st.session_state['ui_data'] is not None and not st.session_state['ui_data']['rfm_segmented'].empty:
        ui_data = st.session_state['ui_data']
        with tab2:
            show_insights_ui(ui_data['kpi_data'], ui_data['segment_summary'], ui_data['segment_counts'], ui_data['top_products_by_segment'], ui_data['df_orders_merged'])
        with tab3:
            show_detailed_view_ui(ui_data['rfm_segmented'], ui_data['customers_at_risk'])
        with tab4:
            show_prediction_tab_ui(ui_data['predicted_cltv_display'], ui_data['cltv_comparison'])
        with tab5:
            show_realization_curve_ui(ui_data['realization_curve'])
        with tab6:
            show_churn_tab_ui(ui_data['rfm_segmented'], ui_data['churn_summary'], ui_data['active_days_summary'], ui_data['churn_detailed_view'])
    else:
        for tab in [tab2, tab3, tab4, tab5, tab6]:
            with tab:
                st.warning("⚠ Please upload or load data first in the 'Upload / Load Data' tab and click 'Process Data'.")

if __name__ == "__main__":
    run_streamlit_app()
