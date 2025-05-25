import optuna
import optuna.visualization as vis

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from batman.core.utils import plot_model_predictions, plot_residual_errors

import logging
logger = logging.getLogger(__name__)

def generate_full_report(model, X_test, y_test, study=None, num_points=300):
    """
    Génère un rapport visuel complet après entraînement du modèle.

    - model : modèle entraîné
    - X_test, y_test : données test
    - study : objet optuna.study (optionnel)
    - datetimes : index datetime pour les prédictions (optionnel)
    - num_points : nombre de points à afficher pour les prédictions
    """

    logger.info("\n📈 --- Analyse des prédictions ---")
    y_pred = model.predict(X_test)

    # Figures matplotlib
    fig_predictions = plot_model_predictions(
        y_true=y_test,
        y_pred=y_pred,
        title="Comparaison Réel vs Prédictions"
    )

    fig_residuals = plot_residual_errors(
        y_true=y_test,
        y_pred=y_pred,
        title="Distribution des Erreurs Résiduelles"
    )

    figures = {
        "prediction_plot": fig_predictions,
        "residual_plot": fig_residuals
    }

    # Ajout des figures Optuna si disponibles
    if study:
        import optuna.visualization as vis
        try:
            figures["optuna_optimization_history"] = vis.plot_optimization_history(study)
            figures["optuna_param_importances"] = vis.plot_param_importances(study)
            figures["optuna_parallel_coordinates"] = vis.plot_parallel_coordinate(study)
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de la génération des figures Optuna : {e}")

    return figures

def optimize_model_with_optuna(model_fn, X, y, search_space_fn, n_trials=20, n_splits=5, storage_url=None, study_name="optuna_study"):
    """
    Optimisation hyperparamètres via Optuna avec espace de recherche flexible.

    model_fn doit accepter **params
    search_space_fn doit accepter (trial) et retourner un dict de paramètres
    """
    logger.info(f"🚀 Démarrage optimisation Optuna pour {study_name}")
    def objective(trial):
        
        params = search_space_fn(trial)
        model = model_fn(**params)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmses = []
        logger.info(f"🔎 Essai {trial.number}: {params}")
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            rmses.append(rmse)

            trial.report(np.mean(rmses), step=len(rmses))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(rmses)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        storage=storage_url,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"🎯 Meilleurs paramètres trouvés pour {study_name}: {study.best_params}")

    return study.best_params, study

def xgboost_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
    }
    
def random_forest_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
    }
    
def knn_search_space(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 2, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_categorical("p", [1, 2]),
        "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "leaf_size": trial.suggest_int("leaf_size", 10, 50),
    }

def ridge_search_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.0001, 100.0, log=True),
        "solver": trial.suggest_categorical("solver", ["auto", "saga", "lsqr", "svd"]),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
    }

def lasso_search_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.0001, 1.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 10000),
        "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
        "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
    }

def lightgbm_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
    }

def catboost_search_space(trial):
    return {
        "iterations": trial.suggest_int("iterations", 300, 1500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255)
    }

def linear_search_space(trial):
    """
    Espace de recherche pour un modèle linéaire.
    """
    return {
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "copy_X": trial.suggest_categorical("copy_X", [True, False])
    }

def get_search_space(model_type):
    """
    Retourne la fonction de recherche d'espace pour le modèle spécifié.
    """
    if model_type == "xgboost":
        return xgboost_search_space
    elif model_type == "random_forest":
        return random_forest_search_space
    elif model_type == "knn":
        return knn_search_space
    elif model_type == "ridge":
        return ridge_search_space
    elif model_type == "lasso":
        return lasso_search_space
    elif model_type == "lightgbm":
        return lightgbm_search_space
    elif model_type == "catboost":
        return catboost_search_space
    elif model_type == "linear":
        return linear_search_space
    else:
        raise ValueError(f"Modèle inconnu : {model_type}")