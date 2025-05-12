from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_model_input_table, preprocess_companies, preprocess_shuttles
# src/eco2mix_forecasting/pipelines/data_download/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import fetch_data_node, convert_data_node

#pipeline de transformations de la donnée
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs="preprocessed_companies",
                name="preprocess_companies_node",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="preprocessed_shuttles",
                name="preprocess_shuttles_node",
            ),
            node(
                func=create_model_input_table,
                inputs=["preprocessed_shuttles", "preprocessed_companies", "reviews"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
                        node(
                func=fetch_data_node,
                inputs="params:destination_folder",
                outputs=None,
                name="fetch_data_node"
            ),
            node(
                func=convert_data_node,
                inputs=["params:xls_path", "params:csv_path"],
                outputs=None,
                name="convert_data_node"
            ),
        ]
    )


# src/eco2mix_forecasting/pipelines/data_preprocessing/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import clean_data_node, merge_data_node

def create_pipeline(**kwargs):
    """ Crée le pipeline de prétraitement des données """
    return Pipeline(
        [
            node(
                func=clean_data_node,
                inputs="raw_data",
                outputs="cleaned_data",
                name="clean_data_node"
            ),
            node(
                func=merge_data_node,
                inputs=["cleaned_data", "additional_data"],
                outputs="merged_data",
                name="merge_data_node"
            ),
        ]
    )
