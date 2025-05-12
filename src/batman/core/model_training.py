from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb

from batman.core.evaluation import evaluate_model_timeseries_cv, print_evaluation_metrics, walk_forward_validation
from batman.core.optuna_optimization import optimize_model_with_optuna, generate_full_report
from batman.core.run_logger import log_run

import pandas as pd
import joblib

def train_linear_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_ridge_model(X, y, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

def train_lasso_model(X, y, alpha=0.1):
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return model

def train_random_forest(X, y, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return model

def train_knn_model(X, y, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X, y)
    return model

def train_xgboost_model(X, y, n_estimators=100, learning_rate=0.1, max_depth=3):
    model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    model.fit(X, y)
    return model

def save_model(model, filename):
    joblib.dump(model, filename)

def get_all_models(X, y):
    """
    Entraîne tous les modèles définis et retourne un dictionnaire nom -> modèle entraîné.
    """
    models = {
        "LinearRegression": train_linear_model(X, y),
        "Ridge": train_ridge_model(X, y),
        "Lasso": train_lasso_model(X, y),
        "RandomForest": train_random_forest(X, y),
        "KNN": train_knn_model(X, y),
    }
    return models

def full_train_evaluate_pipeline(
    model_fn,
    X,
    y,
    params=None,
    search_space_fn=None,
    threshold_rmse=300,
    use_optuna=False,
    n_trials=20,
    n_splits=5,
    run_name=None,
    storage_url=None,
    validation_type="time_series_cv",  # "time_series_cv" ou "walk_forward"
    walkforward_initial_train_size=3000,
    walkforward_test_size=24,
    max_train_size=None,
    generate_report=True,
    datetimes=None,
    X_test=None,
    y_test=None
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
    - generate_report : générer un rapport complet.
    - datetimes : index datetime pour le rapport.
    - X_test, y_test : données test pour le rapport.

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
            model_fn, X, y, search_space_fn,
            n_trials=n_trials, n_splits=n_splits, storage_url=storage_url, study_name=run_name
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

    print("\n--- Résultats Validation Croisée Finale ---")
    print_evaluation_metrics({'Mean RMSE Final': mean_rmse_final})

    if optuna_study:
        print("\n--- Résultats pendant Optimisation Optuna ---")
        print(f"Meilleur RMSE trouvé pendant Optuna : {optuna_study.best_value:.4f}")

    # Logging
    run_info = {
        "run_name": run_name or "Unnamed_Run",
        "model_name": model.__class__.__name__,
        "use_optuna": use_optuna,
        "params": best_params if use_optuna else params,
        "mean_rmse_final": mean_rmse_final,
        "optuna_best_rmse": optuna_study.best_value if optuna_study else None,
        "cv_results": cv_results
    }
    log_run(run_info)

    if mean_rmse_final <= threshold_rmse:
        print(f"✅ RMSE {mean_rmse_final:.2f} sous le seuil {threshold_rmse}, modèle à sauvegarder.")
    else:
        print(f"❌ RMSE {mean_rmse_final:.2f} au-dessus du seuil {threshold_rmse}, modèle NON sauvegardé.")

    model.fit(X, y)
    
    # Génération du rapport
    if generate_report:
        last_split_idx = -walkforward_test_size if validation_type == "walk_forward" else -1
        X_eval = X_test if X_test is not None and y_test is not None else X.iloc[last_split_idx:]
        y_eval = y_test if X_test is not None and y_test is not None else y.iloc[last_split_idx:]
        generate_full_report(model, X_eval, y_eval, study=optuna_study, datetimes=datetimes)

    return model, cv_results