from batman.core.data_preprocessing import split_features_target, split_train_test, normalize_features
from batman.core.model_training import full_train_evaluate_pipeline, get_model_function
from batman.core.optuna_optimization import get_search_space, generate_full_report
from batman.core.evaluation import evaluate_model

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from optuna import load_study

import logging

logger = logging.getLogger(__name__)

def split_features_target_node(data, target):
    return split_features_target(data, target)

def split_train_test_node(X, y, test_size):
    return split_train_test(X, y, test_size=test_size)

def normalize_features_node(X_train, X_test, method):
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test, method)
    # Log the scaler model
    scaler_name = f"Scaler_{method}"
    mlflow.sklearn.log_model(
            sk_model=scaler,
            artifact_path="scaler",
            registered_model_name= scaler_name,
            signature=infer_signature(X_test, X_test_scaled),
            input_example=X_test.iloc[:1]
        )
    client = MlflowClient()
    version = client.get_latest_versions(scaler_name, stages=["None"])[-1].version

    # Ajoute un alias (si souhaité)
    client.set_registered_model_alias(scaler_name, "prod", version)

    # Ajoute un tag pour méthode utilisée
    client.set_model_version_tag(scaler_name, version, "method", method)
    logger.info("🔒 Scaler sauvegardé dans Kedro et loggé dans MLflow.")
    return X_train_scaled, X_test_scaled, scaler

def train_node(X, y, params):
    model_type = params.get("model_type", "xgboost")
    model_fn = get_model_function(model_type)
    search_space_fn = get_search_space(model_type)
    
    logger.info(f"🔧 Début de l'entraînement du modèle {model_type}.")
    model, df_results = full_train_evaluate_pipeline(
        model_fn=model_fn,
        X=X,
        y=y,
        params=params.get("model_params", {}),
        search_space_fn=search_space_fn,
        use_optuna=params.get("use_optuna", False),
        storage_url=params.get("storage_url", None),
        n_trials=params.get("n_trials", 20),
        n_splits=params.get("n_splits", 5),
        run_name=f"{model_type}_run",
        validation_type=params.get("validation_type", "time_series_cv"),
        walkforward_initial_train_size=params.get("walkforward_initial_train_size", 3000),
        walkforward_test_size=params.get("walkforward_test_size", 24),
    )
    logger.info(f"✅ Entraînement du modèle {model_type} terminé.")
    rmse = df_results.iloc[-1]["RMSE"]
    logger.info(f"📉 RMSE final pour {model_type}: {rmse:.4f}")
    return model, df_results, model_type

def evaluate_model_node(model, X_test, y_test, df_results, params):
    """
    Évalue le modèle sur les données de test et génère un rapport visuel.
    """
    model_type = params.get("model_type", "xgboost")
    threshold = params.get("threshold")
    storage_url = params.get("storage_url")

    y_pred = model.predict(X_test)
    signature = infer_signature(X_test, y_pred)
    result = evaluate_model(model, X_test, y_test)
    rmse = result['RMSE']
    logger.info(f"📉 RMSE sur les données de test : {rmse:.4f}")
  
    study = None
    if storage_url:
        try:
            study = load_study(study_name=f"{model_type}_run", storage=storage_url)
            logger.info(f"📂 Study '{model_type}_run' chargée depuis {storage_url}")
        except Exception as e:
            logger.warning(f"⚠️ Impossible de charger la study '{model_type}_run' : {e}")
    # Rapport visuel
    figures = generate_full_report(model, X_test, y_test, study=study, num_points=300)
    
    # Logging conditionnel
    if rmse <= threshold:
        
        
        mlflow.log_params(model.get_params())
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_R2", result['R2'])
        mlflow.log_metric("test_MAE", result['MAE'])
        mlflow.log_metric("test_MAPE", result['MAPE'])
        mlflow.set_tags({
            "model_type": model_type,
            "feature_set": ", ".join(X_test.columns.tolist()),
            "target": params.get("target", "Consommation"),
            "task": "energy_forecasting",
        })
        model_name = f"EnergyForecastModel_{model_type}"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=X_test.iloc[:1]
        )
        client = MlflowClient()
        version = client.get_latest_versions(model_name, stages=["None"])[-1].version

        # Ajout alias "prod" si RMSE valide
        client.set_registered_model_alias(model_name, "prod", version)

        # Tags supplémentaires utiles pour Superman ou MLops
        client.set_model_version_tag(model_name, version, "task", "energy_forecasting")
        client.set_model_version_tag(model_name, version, "framework", "sklearn")
        client.set_model_version_tag(model_name, version, "target", params.get("target", "Consommation"))
        logger.info(f"✅ Modèle {model_type} sauvegardé dans MLflow.")
    else:
        logger.warning(f"❌ RMSE {rmse:.4f} > seuil {threshold}, modèle non loggué dans MLflow.")
        
    return figures