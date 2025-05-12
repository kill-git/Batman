from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur le jeu de test.

    Calcule les métriques :
    - Coefficient de détermination (R²)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)

    Paramètres :
    - model : modèle entraîné.
    - X_test : jeu de test des features.
    - y_test : valeurs réelles de la cible pour le test.

    Retourne :
    - Dictionnaire contenant les métriques évaluées.
    """
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def print_evaluation_metrics(metrics, title="Évaluation du modèle"):
    """
    Affiche les métriques d'évaluation du modèle.

    Paramètres :
    - metrics : dictionnaire contenant les métriques à afficher.
    """
    print(f"{title} :")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

def evaluate_model_timeseries_cv(model, X, y, n_splits=5):
    """
    Évalue la performance d’un modèle via une validation croisée temporelle (TimeSeriesSplit).

    Paramètres :
    - model : modèle non entraîné (sera cloné à chaque split).
    - X : variables explicatives (pandas DataFrame ou array).
    - y : variable cible.
    - n_splits : nombre de découpes temporelles (par défaut : 5).

    Retourne :
    - Liste de dictionnaires de métriques (un par split).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        fold_metrics = evaluate_model(model_clone, X_test, y_test)
        results.append(fold_metrics)

    return results

def print_cv_results(cv_results, title="Validation croisée (TimeSeriesSplit)"):
    """
    Affiche les résultats d'une validation croisée sous forme compacte.

    Paramètres :
    - cv_results : liste de dictionnaires de scores (retournée par evaluate_model_timeseries_cv)
    - title : titre de l'affichage
    """
    print(f"\n{title}")
    print("-" * len(title))

    # Récupération des noms de métriques
    metric_names = cv_results[0].keys()

    # Construction du header
    header = ["Metric"] + [f"Fold {i+1}" for i in range(len(cv_results))] + ["Moyenne"]
    print("{:<10s}".format(header[0]), end=" | ")
    for h in header[1:]:
        print("{:>12s}".format(h), end=" ")
    print("\n" + "-" * (14 * len(header)))

    # Affichage ligne par ligne des métriques
    for metric in metric_names:
        values = [cv[metric] for cv in cv_results]
        mean_value = sum(values) / len(values)

        print("{:<10s}".format(metric), end=" | ")
        for val in values:
            print("{:>12.6f}".format(val), end=" ")
        print("{:>12.6f}".format(mean_value))
        
def evaluate_model_purged_cv(model, X, y, n_splits=5, embargo=0):
    """
    Évalue la performance d’un modèle via une validation croisée temporelle avec purge.

    Paramètres :
    - model : modèle non entraîné (sera cloné à chaque split).
    - X : variables explicatives (pandas DataFrame).
    - y : variable cible (pandas Series).
    - n_splits : nombre de découpes temporelles.
    - embargo : nombre d'observations à exclure après le jeu de test pour éviter les fuites.

    Retourne :
    - Liste de dictionnaires de métriques (un par split).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for train_idx, test_idx in tscv.split(X):
        # Appliquer l'embargo
        if embargo > 0:
            max_train_idx = train_idx[-1]
            min_test_idx = test_idx[0]
            if min_test_idx - max_train_idx <= embargo:
                continue  # Skip this split due to embargo

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        predictions = model_clone.predict(X_test)

        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        results.append({'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

    return results

def walk_forward_validation(model_fn, X, y, initial_train_size, test_size=1, max_train_size=None):
    """
    Walk-forward validation adaptée aux séries temporelles.
    Paramètres :
    - model_fn : fonction qui retourne un modèle vierge
    - X : variables explicatives (pandas DataFrame).
    - y : variable cible (pandas Series).
    - initial_train_size : taille de l'échantillon d'entraînement initial
    - test_size : taille de la prédiction à chaque pas (par défaut : 1)
    - max_train_size : taille max de la fenêtre de training (None = croissant)

    Retourne :
    - liste de dicts de métriques par pas.
    """
    n_samples = len(X)
    metrics_list = []

    train_start = 0
    train_end = initial_train_size

    while train_end + test_size <= n_samples:
        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]

        X_test = X.iloc[train_end:train_end + test_size]
        y_test = y.iloc[train_end:train_end + test_size]

        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        metrics_list.append({
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })

        train_end += test_size
        if max_train_size is not None and (train_end - train_start) > max_train_size:
            train_start = train_end - max_train_size

    return metrics_list