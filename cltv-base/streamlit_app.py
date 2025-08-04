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
            # Run the "full_pipeline" as registered in pipeline_registry.py
            session.run(pipeline_name="full_pipeline") 
            
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
        st.error(f"Error running Kedro pipeline or loading UI data: {e}")
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
                    text-align:center">
            <div style="font-size:16px; font-weight:600; margin-bottom:6px;">{title}</div>
            <div style="font-size:24px; font-weight:bold; color:{color};">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def show_findings_ui(kpi_data: Dict, segment_summary_data: pd.DataFrame, segment_counts_data: pd.DataFrame, top_products_by_segment_data: Dict[str, pd.DataFrame], df_orders_merged: pd.DataFrame):
    st.subheader("Key Performance Indicators")

    # Display Data Timeframe prominently but outside KPI cards
    start_date_kpi = kpi_data.get('start_date', "N/A")
    end_date_kpi = kpi_data.get('end_date', "N/A")
    st.info(f"Data Timeframe: {start_date_kpi} to {end_date_kpi}")
    st.markdown("---") # Add a separator

    # KPIs are now based on the full dataset as processed by Kedro
    total_revenue = kpi_data.get('total_revenue', 0)
    avg_cltv = kpi_data.get('avg_cltv', 0)
    avg_aov = kpi_data.get('avg_aov', 0)
    avg_txns_per_user = kpi_data.get('avg_txns_per_user', 0)
    total_customers = kpi_data.get('total_customers', 0)
    churn_rate = kpi_data.get('churn_rate', 0.0)


    row1_kpis = st.columns(3, gap="small")
    with row1_kpis[0]: kpi_card("Total Revenue", f"₹{total_revenue:,.0f}", color="black")
    with row1_kpis[1]: kpi_card("Average CLTV", f"₹{avg_cltv:,.0f}")
    with row1_kpis[2]: kpi_card("Average Order Value", f"₹{avg_aov:.0f}")
    
    # Add a small vertical space between rows
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    row2_kpis = st.columns(3, gap="small")
    with row2_kpis[0]: kpi_card("Avg Transactions/User", f"{avg_txns_per_user:.0f}")
    with row2_kpis[1]: kpi_card("Total Customers", total_customers, color="black")
    with row2_kpis[2]: kpi_card("Churn Rate", f"{churn_rate:.2f}%", color="red")


    st.divider()
    st.subheader("Segment Visuals") # Changed from 'Visual Insights'

    # Define a minimalist color palette for the new 11 segments
    segment_colors = {
        'Champions': '#60A5FA', # Muted Blue
        'Loyal Customers': '#818CF8', # Muted Indigo
        'Potential Loyalists': '#A78BFA', # Muted Violet
        'Recent Customers': '#C4B5FD', # Lighter Violet
        'Promising': '#D8B4FE', # Lavender
        'Customers Needing Attention': '#F0ABFC', # Light Purple
        'About to Sleep': '#F87171', # Light Red
        'At Risk': '#EF4444', # Red
        "Can't Lose Them": '#DC2626', # Darker Red
        'Hibernating': '#FBBF24', # Amber
        'Lost': '#F59E0B', # Darker Amber
        'Unclassified': '#D1D5DB' # Light Grey
    }

    # Define segment order and descriptions at the start of the function
    segment_order_display = [
        'Champions', 'Loyal Customers', 'Potential Loyalists', 'Recent Customers',
        'Promising', 'Customers Needing Attention', 'About to Sleep', 'At Risk',
        "Can't Lose Them", 'Hibernating', 'Lost', 'Unclassified'
    ]
    
    segment_descriptions = {
        'Champions': "The cream of the crop - your top customers who are the most loyal and generate the most of the revenue. They buy recently, frequently, and spend a lot.",
        'Loyal Customers': "Customers who buy frequently and recently, but may not spend as much as Champions. They are consistent and valuable.",
        'Potential Loyalists': "Recent customers who have made a few purchases and have good potential to become loyal customers if nurtured. They need attention to increase frequency and monetary value.",
        'Recent Customers': "Customers who have made a purchase very recently. They are still fresh and might make repeat purchases soon.",
        'Promising': "Customers who have bought recently and spent a good amount, but their frequency might be lower. They show potential for higher engagement.",
        'Customers Needing Attention': "Customers who haven't purchased for a while and might be at risk of churning. They need re-engagement strategies.",
        'About to Sleep': "Customers who were active but haven't purchased recently. They are on the verge of becoming 'Hibernating' or 'Lost'.",
        'At Risk': "Customers who haven't purchased for a significant period and are likely to churn. Urgent re-engagement is needed.",
        "Can't Lose Them": "High-value customers who were once frequent and high-spending but haven't purchased recently. Losing them would be a significant blow.",
        'Hibernating': "Customers who haven't purchased for a very long time. They are difficult to reactivate but not entirely lost.",
        'Lost': "Customers who have not purchased for the longest time and are highly unlikely to return. Focus on new customer acquisition.",
        'Unclassified': "Customers who do not fit clearly into any of the defined RFM segments. Further analysis may be needed for these."
    }


    if not segment_counts_data.empty:
        st.markdown("#### Customer Segment Distribution")
        
        # Use st.columns for chart and description side-by-side
        col_chart, col_description = st.columns([0.6, 0.4]) # Adjust ratio as needed

        with col_chart:
            fig1 = px.pie(
                segment_counts_data,
                values='Count',
                names='Segment',
                hole=0.45,
                color='Segment',
                color_discrete_map=segment_colors
            )
            fig1.update_traces(textinfo='percent+label', textposition='inside')
            fig1.update_layout(height=500) 
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_description:
            st.markdown("#### Understanding Your Customer Segments")
            
            # Create a selectbox for segment descriptions
            selected_segment_for_desc = st.selectbox(
                "Select a segment to view its description:",
                options=segment_order_display,
                key="segment_description_selector"
            )
            
            # Display the description for the selected segment
            if selected_segment_for_desc:
                description = segment_descriptions.get(selected_segment_for_desc, "Description not available.")
                st.markdown(f"**{selected_segment_for_desc}:** {description}")
            else:
                st.info("Please select a segment from the dropdown to view its description.")


        st.markdown("#### Segment-wise Summary Metrics")
        
        # Row 2: First 6 segment cards
        cards_row_1 = st.columns(6) # 6 columns for 6 cards
        for i in range(6):
            segment = segment_order_display[i] # Get segment name
            with cards_row_1[i]:
                card_color = segment_colors.get(segment, '#aee2fd')
                text_color = "white" if segment in ['At Risk', "Can't Lose Them", 'Lost'] else "black" 
                
                if segment in segment_summary_data.index:
                    metrics = segment_summary_data.loc[segment]
                    st.markdown(f"""
                        <div style="
                            background-color: {card_color};
                            padding: 20px 15px;
                            border-radius: 12px;
                            color: {text_color};
                            min-height: 250px;
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
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
                    # Display a "No data" card to maintain layout
                    st.markdown(f"""
                        <div style="
                            background-color: {card_color};
                            padding: 20px 15px;
                            border-radius: 12px;
                            color: {text_color};
                            min-height: 250px;
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                            align-items: center;
                            text-align: center;
                        ">
                            <h4 style="margin-bottom: 15px; font-size: 20px; font-weight: 700;">
                                {segment}
                            </h4>
                            <p style="font-size: 16px; font-weight: 500;">No data available for this segment.</p>
                        </div>
                    """, unsafe_allow_html=True)
            # No need for an else: st.empty() here, as we iterate through segment_order_display and render a card for each.

        # Row 3: Remaining segment cards (5 of them)
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True) # Space between rows of cards
        cards_row_2 = st.columns(5) # 5 columns for 5 cards
        for i in range(5):
            segment_idx = 6 + i # Start from the 7th segment in the list
            segment = segment_order_display[segment_idx] # Get segment name
            with cards_row_2[i]:
                card_color = segment_colors.get(segment, '#aee2fd')
                text_color = "white" if segment in ['At Risk', "Can't Lose Them", 'Lost'] else "black"
                
                if segment in segment_summary_data.index:
                    metrics = segment_summary_data.loc[segment]
                    st.markdown(f"""
                        <div style="
                            background-color: {card_color};
                            padding: 20px 15px;
                            border-radius: 12px;
                            color: {text_color};
                            min-height: 250px;
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
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
                    # Display a "No data" card to maintain layout
                    st.markdown(f"""
                        <div style="
                            background-color: {card_color};
                            padding: 20px 15px;
                            border-radius: 12px;
                            color: {text_color};
                            min-height: 250px;
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                            align-items: center;
                            text-align: center;
                        ">
                            <h4 style="margin-bottom: 15px; font-size: 20px; font-weight: 700;">
                                {segment}
                            </h4>
                            <p style="font-size: 16px; font-weight: 500;">No data available for this segment.</p>
                        </div>
                    """, unsafe_allow_html=True)
            
    else:
        st.warning("Customer segment distribution data not available for findings.")


    # Top Products by Segment
    st.divider()
    st.markdown("#### Top Products Bought by Segment Customers")
    if top_products_by_segment_data:
        # Add radio button for selection
        metric_choice = st.radio(
            "View Top Products by:",
            ("Total Quantity", "Total Revenue"),
            key="top_products_metric_choice",
            horizontal=True
        )

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
            # Determine y-axis column and title based on choice
            if metric_choice == "Total Quantity":
                y_col = 'Total_Quantity'
                y_axis_title = 'Total Quantity'
                text_template = '%{text:.0f}'
                chart_title = f"Top 5 Products by Quantity for '{selected_segment}' (All Time)"
            else: # Total Revenue
                y_col = 'Total_Revenue'
                y_axis_title = 'Total Revenue (₹)'
                text_template = '₹%{text:,.2f}'
                chart_title = f"Top 5 Products by Revenue for '{selected_segment}' (All Time)"

            st.markdown(f"#### {chart_title}")
            fig_products = px.bar(
                current_segment_products,
                x='product_id',
                y=y_col, # Dynamic y-axis
                text=y_col, # Dynamic text
                labels={'product_id': 'Product ID', y_col: y_axis_title}, # Dynamic label
                color='product_id',
                color_discrete_sequence = [
                        '#08306b',   # Very Dark Blue
                        '#2171b5',   # Mid Blue
                        '#4292c6',   # Light Blue
                        '#6baed6',   # Softer Blue
                        "#9dcce6"    # Pale Blue
                    ]
            )
            fig_products.update_traces(texttemplate=text_template, textposition='outside') # Dynamic text format
            fig_products.update_layout(yaxis_title=y_axis_title, xaxis_title="Product ID") # Dynamic axis title
            st.plotly_chart(fig_products, use_container_width=True)
        else:
            st.info(f"No products found for the '{selected_segment}' segment.")
    else:
        st.warning("Top products by segment data not available.")

def show_prediction_tab_ui(predicted_cltv_display_data: pd.DataFrame, cltv_comparison_data: pd.DataFrame):
    # Removed the outer expander for the Predictions tab content
    st.subheader("Predicted CLTV (Next 3 Months) Overview")
    st.caption("Forecasted Customer Lifetime Value using BG/NBD + Gamma-Gamma model.")

    # Nested expander for the Predicted CLTV table
    with st.expander("Predicted CLTV Table", expanded=False):
        if not predicted_cltv_display_data.empty:
            new_segment_options = [
                "All", 'Champions', 'Loyal Customers', 'Potential Loyalists', 'Recent Customers',
                'Promising', 'Customers Needing Attention', 'About to Sleep', 'At Risk',
                "Can't Lose Them", 'Hibernating', 'Lost', 'Unclassified'
            ]
            table_segment = st.selectbox(
                "Table Filter by Segment", new_segment_options,
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

    # Nested expander for the CLTV Comparison Chart
    with st.expander("CLTV Comparison Chart", expanded=False):
        if not cltv_comparison_data.empty:
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
    # Removed the outer expander for the Detailed View tab content
    st.subheader("Full RFM Segmented Data & At-Risk Customers Overview")

    # Nested expander for Full RFM Segmented Data
    with st.expander("Full RFM Segmented Data with CLTV", expanded=False):
        if not rfm_segmented.empty:
            st.dataframe(rfm_segmented)
        else:
            st.warning("RFM Segmented data not available.")

    # Nested expander for Customers at Risk
    with st.expander("Customers at Risk (Recency > 70 days)", expanded=False):
        st.caption("These are customers whose last purchase was over 70 days ago and may be at risk of churning.")
        if not customers_at_risk_df.empty:
            st.dataframe(customers_at_risk_df)
        else:
            st.info("No customers identified as at risk, or data not available.")

def show_realization_curve_ui(realization_curve_data: Dict[str, pd.DataFrame]):
    st.subheader("Realization Curve of CLTV Over Time")
    if realization_curve_data:
        # Define all available segments for the multiselect, including "Overall Average" and "All Segments"
        all_options = [
            'Overall Average', 'Champions', 'Loyal Customers', 'Potential Loyalists',
            'Recent Customers', 'Promising', 'Customers Needing Attention', 'About to Sleep',
            'At Risk', "Can't Lose Them", 'Hibernating', 'Lost', 'Unclassified'
        ]
        
        # Define default selected options
        default_selected = [
            'Overall Average', 'Champions', 'Loyal Customers', 'Potential Loyalists'
        ]

        # Use st.multiselect for selecting multiple groups
        selected_options = st.multiselect(
            "Select Customer Group(s) for CLTV Curve",
            options=all_options,
            default=[opt for opt in default_selected if opt in all_options], # Ensure defaults exist
            key="realization_curve_segment_multiselect"
        )
        
        if selected_options:
            # Concatenate dataframes for all selected options
            charts_to_display = []
            for option in selected_options:
                df = realization_curve_data.get(option)
                if df is not None and not df.empty:
                    # Ensure 'Segment' column exists for coloring, even for 'Overall Average'
                    if 'Segment' not in df.columns:
                        df_copy = df.copy()
                        df_copy['Segment'] = option # Assign the option name as segment
                        charts_to_display.append(df_copy)
                    else:
                        charts_to_display.append(df)
                else:
                    st.info(f"No data available for '{option}'.")

            if charts_to_display:
                combined_df = pd.concat(charts_to_display, ignore_index=True)

                # Use the broader segment_colors for consistent coloring
                color_map = {
                    'Champions': '#60A5FA',
                    'Loyal Customers': '#818CF8',
                    'Potential Loyalists': '#A78BFA',
                    'Recent Customers': '#C4B5FD',
                    'Promising': '#D8B4FE',
                    'Customers Needing Attention': '#F0ABFC',
                    'About to Sleep': '#F87171',
                    'At Risk': '#EF4444',
                    "Can't Lose Them": '#DC2626',
                    'Hibernating': '#FBBF24',
                    'Lost': '#F59E0B',
                    'Unclassified': '#D1D5DB',
                    'Overall Average': '#000000' # Black for overall average
                }

                fig = px.line(
                    combined_df,
                    x="Period (Days)",
                    y="Avg CLTV per User",
                    text="Avg CLTV per User",
                    markers=True,
                    color='Segment', # Always color by 'Segment' now
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
                        'text': f"CLTV Realization Curve - Selected Segments",
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
                st.info("No data to display for the selected groups.")
        else:
            st.info("Please select at least one customer group to display the realization curve.")
    else:
        st.warning("Realization curve data not available.")

def show_churn_tab_ui(rfm_segmented: pd.DataFrame, churn_summary_data: pd.DataFrame, active_days_summary_data: pd.DataFrame, churn_detailed_view_data: pd.DataFrame):
    st.subheader("Churn Prediction Overview")

    if 'predicted_churn' in rfm_segmented.columns:
        churned = rfm_segmented[rfm_segmented['predicted_churn'] == 1]
        st.metric("Predicted Churned Customers", len(churned))
        # Handle division by zero if rfm_segmented is empty
        churn_rate = (len(churned) / len(rfm_segmented) * 100) if len(rfm_segmented) > 0 else 0.0
        st.metric("Churn Rate (%)", f"{churn_rate:.2f}")
    else:
        st.warning("Churn prediction data not available for overview metrics.")

    st.divider()
    st.markdown("### Churn Summary by Segment")

    col1, col2 = st.columns(2)

    # Use the same color map as defined for the pie chart and segment cards
    segment_colors_churn = {
        'Champions': '#60A5FA',
        'Loyal Customers': '#818CF8',
        'Potential Loyalists': '#A78BFA',
        'Recent Customers': '#C4B5FD',
        'Promising': '#D8B4FE',
        'Customers Needing Attention': '#F0ABFC',
        'About to Sleep': '#F87171',
        'At Risk': '#EF4444',
        "Can't Lose Them": '#DC2626',
        'Hibernating': '#FBBF24',
        'Lost': '#F59E0B',
        'Unclassified': '#D1D5DB'
    }

    with col1:
        st.markdown("#### Avg Churn Probability")
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
        st.markdown("#### Avg Expected Active Days")
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
    st.markdown("### All Customers at a Glance")

    # Removed st.toggle, displaying content directly
    if not churn_detailed_view_data.empty:
        st.dataframe(
            churn_detailed_view_data.style.format({'predicted_churn_prob': '{:.2%}', 'predicted_cltv_3m': '₹{:,.2f}'}),
            use_container_width=True
        )
    else:
        st.info("Detailed churn analysis data not available.")


def run_streamlit_app():
    st.set_page_config(page_title="CLTV Dashboard", layout="wide")
    st.title("Customer Lifetime Value Dashboard")

    # --- Load custom CSS ---
    # This assumes .streamlit/style.css exists in the same directory as streamlit_app.py
    try:
        with open(KEDRO_PROJECT_ROOT / ".streamlit" / "style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Could not find .streamlit/style.css. Default Streamlit font will be used.")
    # --- End custom CSS load ---


    if 'ui_data' not in st.session_state:
        st.session_state['ui_data'] = None
    if 'preprocessing_done' not in st.session_state:
        st.session_state['preprocessing_done'] = False
    
    # Renamed 'Insights' to 'Findings'
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Upload / Load Data", "Findings", "Detailed View", "Predictions", "Realization Curve", "Churn" 
    ])

    with tab1:
        st.subheader("Data Source Selection")
        orders_file = st.file_uploader("Upload Orders CSV", type=["csv"], key="orders_upload")
        transactions_file = st.file_uploader("Upload Transactions CSV", type=["csv"], key="transactions_upload")

        use_sample_data = st.button("Use Sample Data Instead", key="use_sample_button")

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

        if st.button("Process Data (Run Kedro Pipeline)", key="process_data_button"):
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
                    st.error(f"Error preparing data: {e}")
                    st.session_state['preprocessing_triggered'] = False

    if st.session_state.get('preprocessing_triggered'):
        st.session_state['ui_data'] = run_kedro_main_pipeline_and_load_ui_data()
        st.session_state['preprocessing_done'] = True
        st.session_state['preprocessing_triggered'] = False
        
        if st.session_state['ui_data'] is not None:
            st.success("Kedro pipeline completed successfully and UI data loaded!")

    if st.session_state.get('preprocessing_done') and st.session_state['ui_data'] is not None and not st.session_state['ui_data']['rfm_segmented'].empty:
        ui_data = st.session_state['ui_data']
        with tab2:
            show_findings_ui(ui_data['kpi_data'], ui_data['segment_summary'], ui_data['segment_counts'], ui_data['top_products_by_segment'], ui_data['df_orders_merged'])
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
                st.warning("Please upload or load data first in the 'Upload / Load Data' tab and click 'Process Data'.")

if __name__ == "__main__":
    run_streamlit_app()
