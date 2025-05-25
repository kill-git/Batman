from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb

from batman.core.evaluation import evaluate_model_timeseries_cv, print_evaluation_metrics, walk_forward_validation
from batman.core.optuna_optimization import optimize_model_with_optuna, generate_full_report
from batman.core.run_logger import log_run

import pandas as pd

import logging

logger = logging.getLogger(__name__)


def build_linear_model(**params):
    """
    Retourne un LinearRegression avec les paramètres passés.
    """
    return LinearRegression(**params)

def build_ridge_model(**params):
    """
    Retourne un Ridge avec les paramètres passés.
    """
    return Ridge(**params)

def build_lasso_model(**params):
    """
    Retourne un Lasso avec les paramètres passés.
    """
    return Lasso(**params)

def build_random_forest_model(**params):
    """
    Retourne un RandomForestRegressor avec les paramètres passés.
    """
    return RandomForestRegressor(**params)

def build_knn_model(**params):  
    """
    Retourne un KNeighborsRegressor avec les paramètres passés.
    """
    return KNeighborsRegressor(**params)

def build_xgboost_model(**params):
    return xgb.XGBRegressor(**params)


def get_model_function(model_type):
    """
    Retourne la fonction d'entraînement du modèle selon le type spécifié.
    """
    model_functions = {
        "linear": build_linear_model,
        "ridge": build_ridge_model,
        "lasso": build_lasso_model,
        "random_forest": build_random_forest_model,
        "knn": build_knn_model,
        "xgboost": build_xgboost_model
    }
    
    if model_type not in model_functions:
        raise ValueError(f"Type de modèle inconnu : {model_type}. Options disponibles : {list(model_functions.keys())}")
    
    return model_functions[model_type]

def full_train_evaluate_pipeline(
    model_fn,
    X,
    y,
    params=None,
    search_space_fn=None,
    use_optuna=False,
    storage_url=None,
    n_trials=20,
    n_splits=5,
    run_name="default_run",
    validation_type="time_series_cv",  # "time_series_cv" ou "walk_forward"
    walkforward_initial_train_size=3000,
    walkforward_test_size=24,
    max_train_size=None
):
    """
    Pipeline complet d'entraînement, validation, logging, et rapport visuel.

    Paramètres :
    - model_fn : fonction retournant un modèle non entraîné.
    - X, y : données.
    - params : dictionnaire d'hyperparamètres (optionnel).
    - search_space_fn : fonction pour espace de recherche (Optuna).
    - threshold_rmse : seuil pour sauvegarder.
    - use_optuna : utiliser Optuna pour optimisation.
    - n_trials : nombre d'essais Optuna.
    - n_splits : nombre de splits pour TimeSeries CV.
    - run_name : nom du run.
    - storage_url : stockage optuna (sqlite par ex).
    - validation_type : "time_series_cv" ou "walk_forward".
    - walkforward_initial_train_size : taille initiale pour walk-forward.
    - walkforward_test_size : taille du test dans walk-forward.

    Retourne :
    - modèle entraîné final
    - résultats de validation croisée
    """

    best_params = None
    optuna_study = None

    if use_optuna:
        if search_space_fn is None:
            raise ValueError("Un search_space_fn doit être fourni pour utiliser Optuna.")
        best_params, optuna_study = optimize_model_with_optuna(
                                        model_fn, 
                                        X, y, 
                                        search_space_fn,
                                        n_trials=n_trials, 
                                        n_splits=n_splits, 
                                        storage_url=storage_url, 
                                        study_name=run_name
                                    )
        model = model_fn(**best_params)
    else:
        model = model_fn(**(params or {}))

    # Validation croisée selon le type demandé
    if validation_type == "time_series_cv":
        cv_results = evaluate_model_timeseries_cv(model, X, y, n_splits=n_splits)
    elif validation_type == "walk_forward":
        cv_results = walk_forward_validation(
            model_fn=lambda: model_fn(**(best_params if use_optuna else (params or {}))),
            X=X,
            y=y,
            initial_train_size=walkforward_initial_train_size,
            test_size=walkforward_test_size,
            max_train_size=max_train_size
        )
    else:
        raise ValueError(f"Type de validation inconnu : {validation_type}")

    # Résultats
    df_cv = pd.DataFrame(cv_results)
    mean_rmse_final = df_cv['RMSE'].mean()

    logger.info("\n--- Résultats Validation Croisée Finale ---")

    if optuna_study:
        logger.info("\n--- Résultats pendant Optimisation Optuna ---")
        logger.info(f"Meilleur RMSE trouvé pendant Optuna pour {run_name}: {optuna_study.best_value:.4f}")
        logger.info(f"Meilleurs paramètres trouvés pendant Optuna pour {run_name}: {optuna_study.best_params}")
    else:
        logger.info("\n--- Pas d'optimisation Optuna ---")
        logger.info(f"RMSE Final pour {run_name}: {mean_rmse_final:.4f}")
        logger.info(f"Paramètres utilisés pour {run_name}: {params if params else 'Aucun'}")

    model.fit(X, y)

    return model, df_cv