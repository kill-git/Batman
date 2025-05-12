import optuna
import optuna.visualization as vis

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from src.utils import plot_model_predictions, plot_residual_errors

def generate_full_report(model, X_test, y_test, study=None, datetimes=None, num_points=300):
    """
    Génère un rapport visuel complet après entraînement du modèle.

    - model : modèle entraîné
    - X_test, y_test : données test
    - study : objet optuna.study (optionnel)
    - datetimes : index datetime pour les prédictions (optionnel)
    - num_points : nombre de points à afficher pour les prédictions
    """

    print("\n📈 --- Analyse des prédictions ---")
    y_pred = model.predict(X_test)
    
    plot_model_predictions(
        y_true=y_test,
        y_pred=y_pred,
        datetimes=datetimes,
        num_points=num_points,
        title="Comparaison Réel vs Prédictions"
    )

    plot_residual_errors(
        y_true=y_test,
        y_pred=y_pred,
        title="Distribution des Erreurs Résiduelles"
    )

    if study:
        print("\n🧠 --- Analyse de l'optimisation Optuna ---")

        try:
            vis.plot_optimization_history(study).show()
            vis.plot_param_importances(study).show()
            vis.plot_parallel_coordinate(study).show()
        except Exception as e:
            print(f"Erreur lors des plots Optuna : {e}")

def optimize_model_with_optuna(model_fn, X, y, search_space_fn, n_trials=20, n_splits=5, storage_url=None, study_name="optuna_study"):
    """
    Optimisation hyperparamètres via Optuna avec espace de recherche flexible.

    model_fn doit accepter **params
    search_space_fn doit accepter (trial) et retourner un dict de paramètres
    """

    def objective(trial):
        params = search_space_fn(trial)
        model = model_fn(**params)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        rmses = []

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

    print(f"🎯 Meilleurs paramètres trouvés: {study.best_params}")

    try:
        fig = vis.plot_optimization_history(study)
        fig.show()
    except Exception as e:
        print(f"Visualisation échouée : {e}")

    return study.best_params, study

def xgboost_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
    }
    
def random_forest_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
    }
    
def knn_search_space(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 2, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_categorical("p", [1, 2])
    }

def ridge_search_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.0001, 100.0, log=True),
    }

def lasso_search_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.0001, 1.0, log=True),
    }

def lightgbm_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
    }

def catboost_search_space(trial):
    return {
        "iterations": trial.suggest_int("iterations", 300, 1500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }