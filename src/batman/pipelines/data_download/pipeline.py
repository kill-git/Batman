from kedro.pipeline import Pipeline, node
from .nodes import fetch_data_node, convert_data_node

def create_pipeline(**kwargs) -> Pipeline:
     """ Create a pipeline for downloading and converting data."""
     return Pipeline([
        node(fetch_data_node, inputs="params:destination_folder",
             outputs="fetch_done",
             name="fetch_data"),
        node(convert_data_node, inputs=["params:xls_path", "params:csv_path", "fetch_done"],
             outputs=None,
             name="convert_data")
    ])