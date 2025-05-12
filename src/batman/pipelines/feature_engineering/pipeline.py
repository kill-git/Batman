from kedro.pipeline import Pipeline, node
from .nodes import create_features_node, final_feature_cleaning_node

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=create_features_node,
            inputs="merged_data",
            outputs="features_raw",
            name="create_features"
        ),
        node(
            func=final_feature_cleaning_node,
            inputs="features_raw",
            outputs="final_data",
            name="clean_engineered_data"
        ),
    ])