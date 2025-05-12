from kedro.pipeline import Pipeline, node
from .nodes import train_and_evaluate_node

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_and_evaluate_node,
            inputs=["final_data", "params:train"],
            outputs="trained_model",
            name="train_model"
        )
    ])