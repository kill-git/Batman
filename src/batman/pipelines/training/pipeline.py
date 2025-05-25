from kedro.pipeline import Pipeline, node
from .nodes import (
    split_features_target_node,
    split_train_test_node,
    normalize_features_node,
    train_node,
    evaluate_model_node
)

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=split_features_target_node,
            inputs=["final_data", "params:target"],
            outputs=["X", "y"],
            name="split_features_target"
        ),
        node(
            func=split_train_test_node,
            inputs=["X", "y", "params:test_size"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_train_test"
        ),
        node(
            func=normalize_features_node,
            inputs=["X_train", "X_test", "params:scaler"],
            outputs=["X_train_scaled", "X_test_scaled", "scaler"],
            name="normalize_features"
        ),
        node(
            func=train_node,
            inputs={"X": "X_train_scaled", "y": "y_train", "params": "params:train"},
            outputs=["model", "df_results", "model_type"],
            name="train_model"
        ),
        node(
            func=evaluate_model_node,
            inputs={"model": "model", 
                    "X_test": "X_test_scaled", 
                    "y_test": "y_test", 
                    "df_results": "df_results", 
                    "params": "params:train"
                    },
            outputs={
                    "prediction_plot": "prediction_plot",
                    "residual_plot": "residual_plot",
                    "optuna_optimization_history": "optuna_optimization_history",
                    "optuna_param_importances": "optuna_param_importances",
                    "optuna_parallel_coordinates": "optuna_parallel_coordinates"
                },
            name="generate_report"
        ),
    ])