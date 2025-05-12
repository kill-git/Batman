from batman.core.model_training import full_train_evaluate_pipeline, train_xgboost_model
from batman.core.optuna_optimization import xgboost_search_space
import mlflow
import mlflow.sklearn

def train_and_evaluate_node(data, params):
    X = data.drop(columns=[params["target"]])
    y = data[params["target"]]
    # 👇 CONFIG MLflow manuelle ici
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("batman_experiment")
    
    with mlflow.start_run(run_name=params.get("run_name", "DefaultRun")):
        model, cv_results = full_train_evaluate_pipeline(
            model_fn=train_xgboost_model,
            X=X,
            y=y,
            params=params.get("model_params", {}),
            search_space_fn=xgboost_search_space,
            use_optuna=params.get("use_optuna", False),
            n_trials=params.get("n_trials", 20),
            n_splits=params.get("n_splits", 5),
            run_name=params.get("run_name"),
            validation_type=params.get("validation_type", "time_series_cv"),
            walkforward_initial_train_size=params.get("walkforward_initial_train_size", 3000),
            walkforward_test_size=params.get("walkforward_test_size", 24),
            threshold_rmse=params.get("threshold_rmse", 300),
            generate_report=True,
            datetimes=X.index
        )

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_params(params.get("model_params", {}))
        mlflow.log_metrics({"rmse": cv_results[-1]["RMSE"]})

    return model