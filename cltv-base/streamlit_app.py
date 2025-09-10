import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import shutil 
import plotly.express as px 
import json
from typing import Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import Kedro specific components
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Defined paths based on Kedro's data structure at the top level
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
SAMPLE_TRANS_PATH = DATA_01_RAW / "Transactional.csv"


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
            # ui_data['recency_threshold'] = context.catalog.load("calculated_at_risk_threshold")
            ui_data['df_orders_merged'] = context.catalog.load("orders_merged_with_user_id")
            ui_data['df_transactions_typed'] = context.catalog.load("transactions_typed")
            ui_data['calculated_distribution_threshold'] = context.catalog.load("calculated_distribution_threshold")
            ui_data['calculated_user_value_threshold'] = context.catalog.load("calculated_user_value_threshold")
            ui_data['calculated_ml_threshold'] = context.catalog.load("calculated_ml_threshold")
            # 
            # **- NEW LOAD -**
            # 
            ui_data['churn_performance'] = context.catalog.load("churn_performance_metrics_for_ui")

            def _safe_load(name):
                try:
                    return context.catalog.load(name)
                except Exception:
                    return None

            ui_data['monthly_rfm'] = _safe_load("monthly_rfm")
            ui_data['quarterly_rfm'] = _safe_load("quarterly_rfm")
            ui_data['monthly_pair_migrations'] = _safe_load("monthly_pair_migrations") 
            ui_data['quarterly_pair_migrations'] = _safe_load("quarterly_pair_migrations")

        return ui_data
    except Exception as e:
        st.error(f"Error running Kedro pipeline or loading UI data: {e}")
        return None


# date formater
def format_date_with_ordinal(date):
    if pd.isna(date):
        return "N/A"
    day = int(date.strftime('%d'))
    suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return f"{day}{suffix} {date.strftime('%B %Y')}"

# --- Streamlit UI Rendering Functions  ---

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
    start_date_kpi = kpi_data.get('start_date', "N/A").iloc[0]
    end_date_kpi = kpi_data.get('end_date', "N/A").iloc[0]
    st.info(f"Data Timeframe: {start_date_kpi} to {end_date_kpi}")
    st.markdown("---") # Add a separator

    # KPIs are now based on the full dataset as processed by Kedro
    total_revenue = kpi_data['total_revenue'].iloc[0]
    total_Orders = kpi_data['total_orders'].iloc[0]
    total_customers = kpi_data['total_customers'].iloc[0]
    avg_aov = kpi_data['avg_aov'].iloc[0]
    avg_cltv = kpi_data['avg_cltv'].iloc[0]
    avg_orders_per_user = kpi_data['avg_txns_per_user'].iloc[0]
    
    row1_kpis = st.columns(3, gap="small")
    with row1_kpis[0]: kpi_card("Total Customers", total_customers, color="black")
    with row1_kpis[1]: kpi_card("Total Revenue", f"${total_revenue:,.0f}", color="black")
    with row1_kpis[2]: kpi_card("Total Orders", f"{total_Orders:.0f}", color="black")
    
    
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    row2_kpis = st.columns(3, gap="small")
    with row2_kpis[0]: kpi_card("Average CLTV", f"${avg_cltv:,.0f}")
    with row2_kpis[1]: kpi_card("Avg Transactions/User", f"{avg_orders_per_user:.0f}")
    with row2_kpis[2]: kpi_card("Average Order Value", f"${avg_aov:.0f}")


    st.divider()
    st.subheader("Segment Visuals")

    segment_colors = {
        "Champions": "rgba(218, 165, 32, 1.0)",
        "Potential Champions": "rgba(60, 179, 113, 1.0)",
        "Customers Needing Attention": "rgba(255, 165, 0, 1.0)",
        "Recent Customers": "rgba(70, 130, 180, 1.0)",
        "Loyal Lapsers": "rgba(106, 90, 205, 1.0)",
        "About to Sleep": "rgba(255, 99, 71, 1.0)",
        "Lost": "rgba(128, 128, 128, 1.0)"
    }

    # Define segment order and descriptions at the start of the function
    segment_order_display = ['Champions', 'Potential Champions', 'Recent Customers', 
                 'Customers Needing Attention', 'Loyal Lapsers', 'About to Sleep', 'Lost']
    
    segment_descriptions = {
        'Champions': "The cream of the crop - your top customers who are the most loyal and generate the most of the revenue. They buy recently, frequently, and spend a lot.",
        'Potential Champions': "Recent customers who have made a few purchases and have good potential to become loyal customers if nurtured. They need attention to increase frequency and monetary value.",
        'Recent Customers': "Customers who have made a purchase very recently. They are still fresh and might make repeat purchases soon.",
        'Customers Needing Attention': "Customers who haven't purchased for a while and might be at risk of churning. They need re-engagement strategies.",
        'About to Sleep': "Customers who were active but haven't purchased recently. They are on the verge of becoming 'Lost'.",
        'Loyal Lapsers': "Customers who haven't purchased for a significant period and are likely to churn. Customers who used make more revenue are now at the verge of being of lost.",
        'Lost': "Customers who have not purchased for the longest time and are highly unlikely to return.",
        'Unclassified': "Customers who do not fit clearly into any of the defined RFM segments. Further analysis may be needed for these."
    }


    
    if not segment_counts_data.empty:
        st.markdown("#### Customer Segment Distribution")


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
                key="segment_description_selector")
            
# Display the description for the selected segment
            if selected_segment_for_desc:
                description = segment_descriptions.get(selected_segment_for_desc, "Description not available.")
                st.markdown(f"**{selected_segment_for_desc}:** {description}")
            else:
                st.info("Please select a segment from the dropdown to view its description.")
# Check if data is available
    if not segment_summary_data.empty and segment_summary_data.shape[0] > 0:
        st.markdown("#### Segment-wise Summary Metrics")
        
        # Create an iterator for the columns
        cards_row_1 = st.columns(4)
        cards_row_2 = st.columns(4)
        all_columns = cards_row_1 + cards_row_2

        # Loop through the available segments and display a card for each
        for i, segment in enumerate(segment_order_display):
            # Use the modulus operator to get the column index (0-7)
            # And select the correct column from all_columns
            col_idx = i % 8
            col = all_columns[i]

            with col:
                card_color = segment_colors.get(segment, '#aee2fd')
                text_color = "black"

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
                            <li><b>Avg Order Value:</b> ${metrics['aov']:,.2f}</li>
                            <li><b>Avg CLTV:</b> ${metrics['CLTV']:,.2f}</li>
                            <li><b>Avg Txns/User:</b> {metrics['frequency']:,.2f}</li>
                            <li><b>Days Between Orders:</b> {metrics['avg_days_between_orders']:,.2f}</li>
                            <li><b>Avg Recency:</b> {metrics['recency']:,.0f} days</li>
                            <li><b>Monetary Value:</b> ${metrics['monetary']:,.2f}</li>
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

    # Add 'Overall' as the first option for the selectbox
        new_segment_options = [
            'Overall', 'Champions', 'Potential Champions', 'Recent Customers', 
            'Customers Needing Attention', 'Loyal Lapsers', 'About to Sleep', 'Lost', 'Unclassified'
        ]
        
        selected_segment = st.selectbox(
            "Choose a Customer Segment",
            options=new_segment_options,
            index=0, # Set the default to 'Overall' (the first item)
            key="top_products_segment_select"
        )
        
        # First, get the correct base DataFrame
        all_products_data = pd.concat(top_products_by_segment_data.values()) \
                     .groupby('product_id') \
                     .sum() \
                     .reset_index()
        if selected_segment == 'Overall':
            df_to_process = all_products_data
        else:
            df_to_process = top_products_by_segment_data.get(selected_segment, pd.DataFrame())

        # Then, check if the DataFrame is not empty before proceeding
        if not df_to_process.empty:
            # Determine the column to sort by and chart title based on metric_choice
            if metric_choice == "Total Quantity":
                y_col = 'Total_Quantity'
                y_axis_title = 'Total Quantity'
                text_template = '%{text:.0f}'
                if selected_segment == 'Overall':
                    chart_title = "Top 5 Products by Quantity (Overall)"
                else:
                    chart_title = f"Top 5 Products by Quantity for '{selected_segment}' (All Time)"
            else: # Total Revenue
                y_col = 'Total_Revenue'
                y_axis_title = 'Total Revenue ($)'
                text_template = '$%{text:,.2f}'
                if selected_segment == 'Overall':
                    chart_title = "Top 5 Products by Revenue (Overall)"
                else:
                    chart_title = f"Top 5 Products by Revenue for '{selected_segment}' (All Time)"

            # Now perform the sorting and get the top 5
            current_segment_products = df_to_process.sort_values(
            by=y_col, ascending=False
        ).head(5)
            if not current_segment_products.empty:
    # Determine y-axis column and title based on choice
                if metric_choice == "Total Quantity":
                    y_col = 'Total_Quantity'
                    y_axis_title = 'Total Quantity'
                    text_template = '%{text:.0f}'
                    
                    # Update the chart title for the 'Overall' case
                    if selected_segment == 'Overall':
                        chart_title = "Top 5 Products by Quantity (Overall)"
                    else:
                        chart_title = f"Top 5 Products by Quantity for '{selected_segment}' (All Time)"
                    
                else: # Total Revenue
                    y_col = 'Total_Revenue'
                    y_axis_title = 'Total Revenue ($)'
                    text_template = '$%{text:,.2f}'
                    
                    # Update the chart title for the 'Overall' case
                    if selected_segment == 'Overall':
                        chart_title = "Top 5 Products by Revenue (Overall)"
                    else:
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
            new_segment_options =   ['Champions', 'Potential Champions', 'Recent Customers', 
                 'Customers Needing Attention', 'Loyal Lapsers', 'About to Sleep', 'Lost', 'Unclassified']
            
            table_segment = st.selectbox(
                "Table Filter by Segment", new_segment_options,
                index=0, key="predicted_cltv_table_segment_filter"
            )

            if table_segment != "All":
                filtered_df = predicted_cltv_display_data[predicted_cltv_display_data['segment'] == table_segment].copy()
            else:
                filtered_df = predicted_cltv_display_data.copy()

            st.dataframe(
                filtered_df.style.format({'CLTV': '${:,.2f}', 'predicted_cltv_3m': '${:,.2f}'}),
                use_container_width=True
            )
        else:
            st.warning("Predicted CLTV data not available.")

    # Nested expander for the CLTV Comparison Chart
    with st.expander("CLTV Comparison Chart", expanded=False):
        if not cltv_comparison_data.empty:
            chart_segment_options = ['Champions', 'Potential Champions', 'Recent Customers', 
                 'Customers Needing Attention', 'Loyal Lapsers', 'About to Sleep', 'Lost', 'Unclassified']
            
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

def show_detailed_view_ui(
    rfm_segmented: pd.DataFrame,
    customers_at_risk_df: pd.DataFrame,
    threshold_value: dict):

    st.subheader("Full RFM Segmented Data & At-Risk Customers Overview")
    with st.expander("Full RFM Segmented Data with CLTV", expanded=False):
        if not rfm_segmented.empty:
            st.dataframe(rfm_segmented)
        else:
            st.warning("RFM Segmented data not available.")
    
    # Label logic for current metric
    # if selected_metric in ['recency']:
    #   operator_text = f"{selected_metric.title()} > {threshold_value:.0f}"
    #   risk_lambda = lambda x: "at risk" if x >= threshold_value else "regular customers"
    #   caption_text = f"Customers whose {selected_metric.title()} exceeds {threshold_value:.0f}: may be at risk of churning."
    # else:
    #   operator_text = f"{selected_metric.title()} < {threshold_value:.0f}"
    #   risk_lambda = lambda x: "at risk" if x < threshold_value else "regular customers"
    #   caption_text = f"Customers whose {selected_metric.title()} is below {threshold_value:.0f}: may be at risk of churning."
    #threshold= threshold_value['calculated_distribution_threshold']
    operator_text = f"RFM Score < {threshold_value:.0f}"
    caption_text = f"Customers whose RFM Score is lesser than {threshold_value:.0f}: may be at risk of churning."

    with st.expander(f"Customers at Risk ({operator_text})", expanded=False):
        st.caption(caption_text)
        if not customers_at_risk_df.empty:
            st.dataframe(customers_at_risk_df)
        else:
            st.info("No customers identified as at risk, or data not available.")

    with st.expander("RFM Score Distribution (Box & Histogram)"):
        def plot_metric_distribution(rfm_df: pd.DataFrame,  threshold_val: float):
            # Correct logic for all metrics:
            
            rfm_df = rfm_df.copy()
            rfm_df["Risk_Label"] = rfm_df["rfm_score"].apply(lambda x: "at risk" if x < threshold_val else "regular customers")
            colors = {
                "at risk": ("crimson", "lightcoral"),
                "regular customers": ("deepskyblue", "lightblue")
            }
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f"Box Plot ", f"Histogram"),
                horizontal_spacing=0.15
            )
            for label, (box_color, _) in colors.items():
                subset = rfm_df[rfm_df["Risk_Label"] == label]
                fig.add_trace(
                    go.Box(
                        y=subset["rfm_score"],
                        name=label,
                        marker_color=box_color,
                        boxpoints="outliers"
                    ),
                    row=1, col=1
                )
            for label, (_, hist_color) in colors.items():
                subset = rfm_df[rfm_df["Risk_Label"] == label]
                fig.add_trace(
                    go.Histogram(
                        x=subset["rfm_score"],
                        name=label,
                        marker_color=hist_color,
                        opacity=0.7
                    ),
                    row=1, col=2
                )
            fig.update_layout(
                height=500,
                width=1000,
                title_text=f"RFM Score Distribution by Risk Segment",
                barmode="overlay"
            )
            return fig

        fig_metric = plot_metric_distribution(rfm_segmented, threshold_value)
        st.plotly_chart(fig_metric, use_container_width=True)


def show_realization_curve_ui(realization_curve_data: Dict[str, pd.DataFrame]):
    st.subheader("Realization Curve of CLTV Over Time")
    if realization_curve_data:
        # Define all available segments for the multiselect, including "Overall Average" and "All Segments"
        all_options = ['Champions', 'Potential Champions', 'Recent Customers', 
                 'Customers Needing Attention', 'Loyal Lapsers', 'About to Sleep', 'Lost', 'Unclassified', 'Overall Average']
        
        # Define default selected options
        default_selected = [
            'Overall Average'
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
                         "Champions": "rgba(218, 165, 32, 1.0)",
                         "Potential Champions": "rgba(60, 179, 113, 1.0)",
                         "Customers Needing Attention": "rgba(255, 165, 0, 1.0)",
                         "Recent Customers": "rgba(70, 130, 180, 1.0)",
                         "Loyal Lapsers": "rgba(106, 90, 205, 1.0)",
                         "About to Sleep": "rgba(255, 99, 71, 1.0)",
                         "Lost": "rgba(128, 128, 128, 1.0)"
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
    # Use st.columns to create a clean side-by-side layout for the KPI cards
        col1, col2 = st.columns(2)

        # Calculate the metrics
        churned = rfm_segmented[rfm_segmented['predicted_churn'] == 1]
        
        # Handle the case where the dataframe might be empty to prevent division by zero
        churn_rate = (len(churned) / len(rfm_segmented) * 100) if len(rfm_segmented) > 0 else 0.0

        # Custom CSS for the KPI cards
        # We use st.markdown with unsafe_allow_html=True to inject the styling.
        card_style = """
        <style>
        .kpi-card {
            background-color: #B6D7F9; /* A soft, light blue for the background */
            color: black;            /* Black text for high contrast */
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Adds a subtle shadow for depth */
            font-family: 'Inter', sans-serif;
        }
        .kpi-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .kpi-value {
            font-size: 2.5rem;
            font-weight: 800;
        }
        </style>
        """
        st.markdown(card_style, unsafe_allow_html=True)

        # Card for Predicted Churned Customers
        with col1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Predicted Churned Customers</div>
                    <div class="kpi-value">{len(churned)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Card for Churn Rate (%)
        with col2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Churn Rate (%)</div>
                    <div class="kpi-value">{churn_rate:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        # Display a warning if the required data is not available
        st.warning("Churn prediction data not available for overview metrics.")

    st.divider()
    st.markdown("### Churn Summary by Segment")

    # Use the same color map as defined for the pie chart and segment cards
    segment_colors_churn = {
                         "Champions": "rgba(218, 165, 32, 1.0)",
                         "Potential Champions": "rgba(60, 179, 113, 1.0)",
                         "Customers Needing Attention": "rgba(255, 165, 0, 1.0)",
                         "Recent Customers": "rgba(70, 130, 180, 1.0)",
                         "Loyal Lapsers": "rgba(106, 90, 205, 1.0)",
                         "About to Sleep": "rgba(255, 99, 71, 1.0)",
                         "Lost": "rgba(128, 128, 128, 1.0)"
    }

    
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
        fig_churn.update_traces(texttemplate='%{x:.1%}', textposition='outside')
        fig_churn.update_layout(
            height=450, 
            margin=dict(t=30),
            xaxis=dict(tickformat=".0%") # Format axis labels as percentages
            ) 
        st.plotly_chart(fig_churn, use_container_width=True)
        

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
        fig_days.update_traces(texttemplate='%{x:.0f}', textposition='outside')
        fig_days.update_layout(height=450, margin=dict(t=30))
        st.plotly_chart(fig_days, use_container_width=True)

    
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


#tab7 funcions
def _segments_in_order(df: pd.DataFrame):
    # Build order dynamically from dataset
    preferred =   ['Champions', 'Potential Champions', 'Recent Customers', 
                 'Customers Needing Attention', 'Loyal Lapsers', 'About to Sleep', 'Lost', 'Unclassified']
        
    present = [s for s in preferred if s in df['Segment'].unique().tolist()]
    extras = sorted(list(set(df['Segment'].unique().tolist()) - set(present)))
    return present + extras

def _list_pairs(migration_by_pair: dict, period_freq: str):
    pairs = []
    # Check if migration_by_pair is a DataFrame and if it's empty
    if isinstance(migration_by_pair, pd.DataFrame) and migration_by_pair.empty:
        return pairs
    # Check if it's an empty dictionary
    elif isinstance(migration_by_pair, dict) and not migration_by_pair:
        return pairs
    
    # Iterate through the dictionary keys, which are the (cp, np_) tuples
    for pair_tuple in migration_by_pair.keys():
        pairs.append(pair_tuple)
    
    # Sort the pairs to ensure chronological order in the dropdown
    pairs.sort()
    return pairs

def _filter_moved_for_pair(rfm_df, current_period, next_period, from_seg, to_seg, period_col, period_freq):
    """Filter customers who moved from one segment to another between periods."""
    #cust_col = _detect_customer_col(rfm_df)

    d = rfm_df[["User ID", period_col, "Segment"]].copy()
    d["next_period"] = d.groupby("User ID")[period_col].shift(-1)
    d["next_segment"] = d.groupby("User ID")["Segment"].shift(-1)

    moved = d[
        (d[period_col] == current_period)
        & (d["next_period"] == next_period)
        & (d["Segment"] == from_seg)
        & (d["next_segment"] == to_seg)
    ][["User ID", period_col, "Segment", "next_period", "next_segment"]]

    return moved


def pair_key(migration_by_pair: dict, cp: str, np_: str, period_freq: str):
    # Return the (key) that matches cp->np_ in the dict (supports str/pd.Period mixing)
    if not migration_by_pair:
        return None
    cpP = pd.Period(cp, period_freq)
    npP = pd.Period(np_, period_freq)
    key = None
    if (cpP, npP) in migration_by_pair:
        key = (cpP, npP)
    else:
        for (k1, k2) in migration_by_pair.keys():
            kk1 = k1 if isinstance(k1, pd.Period) else pd.Period(str(k1), period_freq)
            kk2 = k2 if isinstance(k2, pd.Period) else pd.Period(str(k2), period_freq)
            if kk1 == cpP and kk2 == npP:
                key = (k1, k2)
                break
    return key

def show_customer_migration_tab_ui(monthly_rfm: pd.DataFrame,
                                 quarterly_rfm: pd.DataFrame,
                                 monthly_pairs: dict,
                                 quarterly_pairs: dict):
    st.subheader("Customer Migration")

    # Guard: artifacts available?
    if monthly_rfm is None or quarterly_rfm is None or monthly_pairs is None or quarterly_pairs is None:
        st.warning(
            "Migration artifacts not found. Ensure your Kedro pipeline saves "
            "monthly_rfm, quarterly_rfm, monthly_pair_migrations, and quarterly_pair_migrations."
        )
        return

    # ----------------- INLINE CONTROLS (replacing sidebar) -----------------
    st.markdown("### Migration Controls")
    freq = st.radio(
        "Migration Frequency",
        ["Monthly (M)", "Quarterly (Q)"],
        index=0,
        key="migration_freq",
        horizontal=True
    )

    # Pick data by frequency
    if freq.startswith("Monthly"):
        rfm_df = monthly_rfm.copy()
        pairs = _list_pairs(monthly_pairs, "M") # This now returns a list of tuples
        pair_dict = monthly_pairs
        period_col = "month"; period_freq = "M"
    else:
        rfm_df = quarterly_rfm.copy()
        pairs = _list_pairs(quarterly_pairs, "Q") # This now returns a list of tuples
        pair_dict = quarterly_pairs
        period_col = "quarter"; period_freq = "Q"

    if not pairs:
        st.warning("No period pairs found in the data.")
        return
    
    if rfm_df is None or rfm_df.empty or not pairs:
        st.info("No migration pairs available. Run the pipeline on enough periods first.")
        return

    seg_list = _segments_in_order(rfm_df)

    # Place the remaining selectors in a single row
    c1, c2, c3 = st.columns([1.6, 1, 1], gap="small")
    with c1:
        # Correctly format the tuples for display in the selectbox
        # We need to handle `pd.Period` objects gracefully
        display_labels = []
        for cp, np_ in pairs:
            # Handle potential string or Period objects
            cp_str = str(cp) if not isinstance(cp, pd.Period) else cp.to_timestamp().strftime('%Y-%m')
            np_str = str(np_) if not isinstance(np_, pd.Period) else np_.to_timestamp().strftime('%Y-%m')
            display_labels.append(f"{cp_str} → {np_str}")

        pair_label = st.selectbox(
            "Period Pair",
            display_labels, # The correctly formatted labels
            key="mig_pair"
        )
    with c2:
        from_seg = st.selectbox("From Segment", seg_list, key="mig_from_seg")
    with c3:
        to_seg = st.selectbox("To Segment", seg_list, key="mig_to_seg")

    cp, np_ = pair_label.split(" → ")


    # ----------------- MAIN CONTENT -----------------
    # Drilldown table
    #st.markdown("### Drilldown (Who moved?)")
    with st.expander(f"**{from_seg} → {to_seg}** for **{cp} → {np_}**", expanded=False):
        moved = _filter_moved_for_pair(rfm_df, cp, np_, from_seg, to_seg, period_col, period_freq)
        if not moved.empty:
            st.dataframe(moved)
    
    
    #st.write(f"**{from_seg} → {to_seg}** for **{cp} → {np_}**")
    #st.dataframe(moved, use_container_width=True, height=280)

    csv = moved.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (this From→To)",
        data=csv,
        file_name=f"moved_{from_seg}to{to_seg}__{cp}to{np_}.csv",
        mime="text/csv",
    )

    # ---- Bar chart --------
    st.markdown("### Customer Distribution by Segment")
    key = pair_key(pair_dict, cp, np_, period_freq)
    # Dropdown to select the previous segment
    from_seg = st.selectbox(
        "Select Previous Segment:",
        seg_list,
        key="bar_chart_from_seg"
    )

    # Get the counts for the selected previous segment
    counts = pair_dict[key]['counts'].reindex(index=seg_list, columns=seg_list).fillna(0).astype(int)
    segment_counts = counts.loc[from_seg]

    # Calculate the percentage distribution
    total = segment_counts.sum()
    if total > 0:
        segment_distribution = (segment_counts / total).mul(100).round(1).astype(str) + '%'
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'Next Segment': segment_counts.index,
            'Customer Count': segment_counts.values,
            'Percentage': (segment_counts / total).values
        })

        # Filter out segments with 0 customers for a cleaner chart
        plot_df = plot_df[plot_df['Customer Count'] > 0]

        # Use Plotly Express to create the bar chart
        fig = px.bar(
            plot_df,
            x='Next Segment',
            y='Customer Count',
            color='Next Segment', # Color the bars by segment
            text='Percentage',
            title=f"Distribution of Customers from **{from_seg}** ({cp}) to Next Segments ({np_})",
            labels={
                "Next Segment": f"Next Segment ({np_})",
                "Customer Count": "Number of Customers"
            }
        )
        
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        
        # Customize the layout for better readability
        fig.update_layout(
            xaxis_title=f"Next Segment ({np_})",
            yaxis_title="Number of Customers",
            font=dict(size=14, color="black"),
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=10),
            height=450,
            width=350 # Legend is redundant since x-axis labels are clear
        )
    st.plotly_chart(fig, use_container_width=True)
    # Heatmap for this pair (row-wise %)
    st.markdown("### Heatmap (Row-wise % for selected pair)")
    key = pair_key(pair_dict, cp, np_, period_freq)
    if key is None:
        st.info("Selected period pair not found in artifacts.")
        return

    mat = pair_dict[key]['percent'].copy()
    mat = mat.reindex(index=seg_list, columns=seg_list).fillna(0.0)

    fig_hm = px.imshow(
        mat.values, x=mat.columns, y=mat.index, aspect="auto",
        color_continuous_scale="Blues", origin="upper"
    )

    fig_hm.update_traces(
        text=np.round(mat.values * 100).astype(int),
        texttemplate="%{text}%",
        hovertemplate="From %{y} → %{x}<br>%{z:.1%}<extra></extra>"
    )

    fig_hm.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=520)
    st.plotly_chart(fig_hm, use_container_width=True)

    
        # --- START OF SANKEY CODE BLOCK ---
    seg_list = ['Champions', 'Potential Champions', 'Recent Customers', 
                 'Customers Needing Attention', 'Loyal Lapsers', 'About to Sleep', 'Lost']
    segment_palette = {
    "Champions": "rgba(218, 165, 32, 1.0)",
    "Potential Champions": "rgba(60, 179, 113, 1.0)",
    "Customers Needing Attention": "rgba(255, 165, 0, 1.0)",
    "Recent Customers": "rgba(70, 130, 180, 1.0)",
    "Loyal Lapsers": "rgba(106, 90, 205, 1.0)",
    "About to Sleep": "rgba(255, 99, 71, 1.0)",
    "Lost": "rgba(128, 128, 128, 1.0)" 
    }

    
    sel = st.multiselect(
        "Show ONLY flows involving these segments (either side):",
        seg_list,
        default=seg_list
    )

    

    # --- Data Preparation for Sankey Chart ---
    counts = pair_dict[key]['counts'].reindex(index=seg_list, columns=seg_list).fillna(0).astype(int)
    S = len(seg_list)
    labels = [f"{s} ({cp})" for s in seg_list] + [f"{s} ({np_})" for s in seg_list]

    source = []; target = []; value = []
    for i, s_from in enumerate(seg_list):
        for j, s_to in enumerate(seg_list):
            v = int(counts.iloc[i, j])
            if v <= 0:
                continue
            
            if s_from not in sel and s_to not in sel:
                continue
            source.append(i)
            target.append(S + j)
            value.append(v)

    # --- Color Assignment Logic ---
    
    node_colors = [segment_palette.get(s, 'black') for s in seg_list] * 2

    # Get a list of colors for each link based on the source node's segment
    # This is the core fix for the ordering problem
    y_positions = [1 - (i / (S - 1)) for i in range(S)]
    node_y = y_positions + y_positions
    # For robustness, add a small epsilon to avoid 0.0 or 1.0 which can cause issues
    node_y = [pos if pos not in [0.0, 1.0] else (0.0001 if pos == 0.0 else 0.9999) for pos in node_y]

    link_colors = []
    for i in range(len(source)):
        source_index = source[i]
        source_segment_name = seg_list[source_index]
        link_color = segment_palette.get(source_segment_name, 'rgba(200, 200, 200, 0.4)')
        # Make the link color semi-transparent
        link_colors.append(link_color.replace('1.0', '0.4'))

    # --- Sankey Chart Creation ---
    if value:
        fig = go.Figure(go.Sankey(
            node=dict(
                label=labels,
                pad=15,
                thickness=10,
                color=node_colors,
                #font=dict(size=15, color="black"),
                line=dict(color="black", width=0.5),
                hovertemplate="Segment: %{label}<br>Count: %{value}<extra></extra>",
                y=node_y
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors,
                hovertemplate="From: %{source.label}<br>To: %{target.label}<br>Count: %{value}<extra></extra>"
            )
        ))
        fig.update_layout(
            title="Customer Migration Flow",
            font=dict(size=15, color="white"),
            margin=dict(l=0, r=0, t=30, b=0),
            height=520
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No links to display with the current selection.")


# **- NEW FUNCTION -**
# 

def show_performance_tab_ui(churn_performance: Dict):
    st.subheader("Churn Model Performance Metrics")
    st.markdown("Metrics for the Random Forest Classifier trained to predict churn.")

    # KPI Cards for overall metrics
    if not churn_performance['metrics'].empty:
        st.markdown("#### Overall Performance")
        metrics_df = churn_performance['metrics'].set_index("Metric")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{metrics_df.loc['Accuracy', 'Value']:.2f}")
        with col2:
            st.metric("Precision (Churn)", f"{metrics_df.loc['Precision (Churn)', 'Value']:.2f}")
        with col3:
            st.metric("Recall (Churn)", f"{metrics_df.loc['Recall (Churn)', 'Value']:.2f}")
        with col4:
            st.metric("F1-Score (Churn)", f"{metrics_df.loc['F1-Score (Churn)', 'Value']:.2f}")

    st.markdown("#### Confusion Matrix")
    st.caption("How many predictions were correct vs. incorrect?")

    if not churn_performance['confusion_matrix'].empty:
        fig_cm = px.imshow(
            churn_performance['confusion_matrix'],
            text_auto=True,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            color_continuous_scale=px.colors.sequential.Teal,
            aspect="auto"
        )
        fig_cm.update_layout(
            xaxis=dict(title="Predicted"),
            yaxis=dict(title="Actual"),
            width=500,
            height=500
        )
        st.plotly_chart(fig_cm)
    else:
        st.warning("Churn model performance data not available.")


def run_streamlit_app():
    st.set_page_config(page_title="CLTV Dashboard", layout="wide")
    st.title("Customer Lifetime Value Dashboard")
    if 'ui_data' not in st.session_state:
        st.session_state['ui_data'] = None
    if 'preprocessing_done' not in st.session_state:
        st.session_state['preprocessing_done'] = False
    
    # **- NEW TAB -**
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Upload / Load Data", "Findings",  "Predictions", "Realization Curve", "Churn", "Customer Migration", "Model Performance"
    ])

    with tab1:
        st.subheader("Data Source Selection")
        orders_file = st.file_uploader("Upload Orders CSV", type=["csv"], key="orders_upload")
        transactions_file = st.file_uploader("Upload Transactions CSV", type=["csv"], key="transactions_upload")

        use_sample_data = st.button("Use Sample Data", key="use_sample_button")

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

        if st.button("Process Data", key="process_data_button"):
            st.cache_data.clear() 
            if 'data_source' not in st.session_state:
                st.warning("Please upload files or select sample data first.")
                return

            with st.spinner("Preparing data"):
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
                    
                    st.info("Processing Data")

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
            st.success("UI loaded!")

    if st.session_state.get('preprocessing_done') and st.session_state['ui_data'] is not None and not st.session_state['ui_data']['rfm_segmented'].empty:
        ui_data = st.session_state['ui_data']
        with tab2:
            show_findings_ui(ui_data['kpi_data'], ui_data['segment_summary'], ui_data['segment_counts'], ui_data['top_products_by_segment'], ui_data['df_orders_merged'])
        with tab3:
            show_prediction_tab_ui(ui_data['predicted_cltv_display'], ui_data['cltv_comparison'])
        with tab4:
            show_realization_curve_ui(ui_data['realization_curve'])
        with tab5:

            show_detailed_view_ui(
                ui_data['rfm_segmented'],
                ui_data['customers_at_risk'],
                ui_data['calculated_distribution_threshold']
            )
            show_churn_tab_ui(ui_data['rfm_segmented'], ui_data['churn_summary'], ui_data['active_days_summary'], ui_data['churn_detailed_view'])
        
        with tab6:
            show_customer_migration_tab_ui(
                ui_data['monthly_rfm'], ui_data['quarterly_rfm'],
                ui_data['monthly_pair_migrations'], ui_data['quarterly_pair_migrations'])
        
        # **- NEW TAB -**
        with tab7:
            show_performance_tab_ui(ui_data['churn_performance'])

    else:
        for tab in [tab2, tab3, tab4, tab5, tab6, tab7]:
            with tab:
                st.warning("Please upload or load data first in the 'Upload / Load Data' tab and click 'Process Data'.")

if __name__ == "__main__":
    run_streamlit_app()