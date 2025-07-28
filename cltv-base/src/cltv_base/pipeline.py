# src/cltv_base/pipeline.py

from kedro.pipeline import Pipeline, node
# Corrected imports: explicitly import the 'pipeline' module from each sub-directory
from cltv_base.pipelines.data_processing import pipeline as data_processing_pipeline_module
from cltv_base.pipelines.customer_features import pipeline as customer_features_pipeline_module
from cltv_base.pipelines.cltv_modeling import pipeline as cltv_modeling_pipeline_module
from cltv_base.pipelines.churn_modeling import pipeline as churn_modeling_pipeline_module
from cltv_base.pipelines.ui_data_preparation import pipeline as ui_data_preparation_pipeline_module

from .nodes import combine_final_customer_data # Import the general utility node

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the project's pipeline, assembling all sub-pipelines.
    """
    # Create instances of each sub-pipeline by calling their create_pipeline function
    data_processing_pipeline = data_processing_pipeline_module.create_pipeline()
    customer_features_pipeline = customer_features_pipeline_module.create_pipeline()
    cltv_modeling_pipeline = cltv_modeling_pipeline_module.create_pipeline()
    churn_modeling_pipeline = churn_modeling_pipeline_module.create_pipeline()
    ui_data_preparation_pipeline = ui_data_preparation_pipeline_module.create_pipeline()

    # Define a node to combine the final customer data from different pipelines
    combine_customer_data_node = node(
        func=combine_final_customer_data,
        inputs=[
            "historical_cltv_customers",          # From customer_features_pipeline
            "predicted_churn_probabilities",      # From churn_modeling_pipeline
            "predicted_churn_labels",             # From churn_modeling_pipeline
            "cox_predicted_active_days"           # From churn_modeling_pipeline
        ],
        outputs="final_rfm_cltv_churn_data",
        name="combine_final_customer_data_for_ui",
    )

    # Assemble the full pipeline
    return (
        data_processing_pipeline
        + customer_features_pipeline
        + cltv_modeling_pipeline
        + churn_modeling_pipeline
        + Pipeline([combine_customer_data_node]) # Wrap the single node in a Pipeline
        + ui_data_preparation_pipeline
    )
