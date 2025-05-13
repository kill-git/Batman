from kedro.pipeline import Pipeline, node
from .nodes import concat_annual_node, concat_tempo_node, preprocess_annual_node, preprocess_tempo_node, merge_data_node, clean_merged_data_node

def create_pipeline(**kwargs):
    return Pipeline([
        node(concat_annual_node, inputs="params:path_annual", outputs="annual_data", name="concat_annual"),
        node(concat_tempo_node, inputs="params:path_tempo", outputs="tempo_data", name="concat_tempo"),
        node(preprocess_annual_node, inputs="annual_data", outputs="annual_data_preprocessed", name="preprocess_annual"),
        node(preprocess_tempo_node, inputs="tempo_data", outputs="tempo_data_preprocessed", name="preprocess_tempo"),
        node(merge_data_node, inputs=["annual_data_preprocessed", "tempo_data_preprocessed"], outputs="merged_data_raw", name="merge"),
        node(clean_merged_data_node, inputs="merged_data_raw", outputs="merged_data", name="clean_merged")
    ])