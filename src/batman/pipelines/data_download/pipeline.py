from kedro.pipeline import Pipeline, node
from .nodes import fetch_data_node, convert_data_node

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=fetch_data_node,
            inputs="params:temp_folder",
            outputs="eco2mix_raw_zip",
            name="fetch_data"
        ),
        node(
            func=convert_data_node,
            inputs="eco2mix_raw_zip",
            outputs="eco2mix_converted_zip",
            name="convert_data"
        )
    ])
