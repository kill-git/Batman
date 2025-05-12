import pandas as pd
import numpy as np
from typing import List

def create_date_features(df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
    """
    Génère des variables temporelles dérivées à partir d'une colonne de date.

    Args:
        df (pd.DataFrame): DataFrame contenant la colonne de date.
        date_column (str): Nom de la colonne contenant les timestamps.

    Returns:
        pd.DataFrame: DataFrame enrichi avec les colonnes temporelles suivantes :
            - year, month, day
            - weekday (0=lundi, 6=dimanche)
            - weekofyear, quarter, dayofyear
            - is_weekend (booléen), is_end_of_month (booléen)
    """
    
    assert date_column in df.columns, f"La colonne '{date_column}' doit être présente dans le DataFrame."
    
    df[date_column] = pd.to_datetime(df[date_column])  # Conversion explicite en datetime
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['weekday'] = df[date_column].dt.weekday  # Jour de la semaine (0 = lundi)
    df['weekofyear'] = df[date_column].dt.isocalendar().week
    df['quarter'] = df[date_column].dt.quarter
    df['dayofyear'] = df[date_column].dt.dayofyear
    df['is_weekend'] = df['weekday'] >= 5  # Samedi (5) et dimanche (6)
    df['is_end_of_month'] = df[date_column].dt.is_month_end
    return df

def create_lag_features(df: pd.DataFrame, target_column: str, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
    """
    Crée des variables de décalage (lag) pour une série temporelle.

    Args:
        df (pd.DataFrame): DataFrame contenant les données chronologiques.
        target_column (str): Nom de la colonne cible (ex: 'Consommation').
        lags (List[int]): Liste des décalages souhaités (en nombre de lignes).

    Returns:
        pd.DataFrame: DataFrame avec des colonnes 'target_column_lag_X' pour chaque X dans lags.
    """
    assert target_column in df.columns, f"La colonne '{target_column}' doit être présente dans le DataFrame."
    assert 'Datetime' in df.columns, "La colonne 'Datetime' doit être présente dans le DataFrame."
    
    df = df.sort_values(by='Datetime')  # Tri par date pour cohérence des décalages
    for lag in lags:
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    return df

def create_rolling_features(df: pd.DataFrame, target_column: str, window: int = 3) -> pd.DataFrame:
    """
    Calcule une moyenne mobile simple sur la colonne cible.

    Args:
        df (pd.DataFrame): DataFrame contenant la série temporelle.
        target_column (str): Nom de la colonne cible.
        window (int): Taille de la fenêtre pour la moyenne glissante (en heures par exemple).

    Returns:
        pd.DataFrame: DataFrame avec une nouvelle colonne de moyenne mobile.
    """
    assert window > 0, "La taille de la fenêtre doit être supérieure à 0."
    assert target_column in df.columns, f"La colonne '{target_column}' doit être présente dans le DataFrame."
    assert 'Datetime' in df.columns, "La colonne 'Datetime' doit être présente dans le DataFrame."
    df = df.sort_values(by='Datetime')  # Tri nécessaire pour la fenêtre glissante
    df[f'{target_column}_rolling_mean_{window}'] = df[target_column].shift(1).rolling(window=window, min_periods=1).mean()
    return df

def create_cyclical_features(
    df: pd.DataFrame,
    column: str,
    max_val: int,
    transform_func: callable = None,
    prefix: str = None
) -> pd.DataFrame:
    """
    Crée des features cycliques (sin et cos) pour une colonne donnée, avec transformation optionnelle.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        column (str): Nom de la colonne cible (ex: 'month', 'weekday', 'Heures', etc.).
        max_val (int): Valeur maximale du cycle (12 pour mois, 24 pour heures, etc.).
        transform_func (callable, optional): Fonction à appliquer aux valeurs avant transformation.
        prefix (str, optional): Préfixe personnalisé pour les colonnes créées (sin/cos). Par défaut = `column`.

    Returns:
        pd.DataFrame: DataFrame enrichi des colonnes `{prefix}_sin` et `{prefix}_cos`.
    """
    # Préfixe utilisé pour nommer les nouvelles colonnes
    base = prefix if prefix is not None else column

    # Appliquer la transformation si nécessaire, sinon utiliser la colonne brute
    if transform_func:
        temp_col = f'{base}_transformed'
        df[temp_col] = df[column].apply(transform_func)
        values = df[temp_col]
    else:
        values = df[column]

    # Créer les colonnes cycliques (sans écraser les colonnes d'origine)
    df[f'{base}_sin'] = np.sin(2 * np.pi * values / max_val)
    df[f'{base}_cos'] = np.cos(2 * np.pi * values / max_val)

    return df

def create_hour_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait l'heure et l'encode cycliquement à partir d'une colonne horaire.

    Args:
        df (pd.DataFrame): DataFrame contenant une colonne horaire au format HH:MM.
        hour_column (str): Nom de la colonne contenant les heures.

    Returns:
        pd.DataFrame: DataFrame avec 'hour', 'hour_sin', 'hour_cos'.
    """
    assert 'Heures' in df.columns, "La colonne 'Heures' doit être présente dans le DataFrame."
    return create_cyclical_features(df,
                                    column='Heures',
                                    max_val=24, 
                                    transform_func=lambda x: pd.to_datetime(x, format='%H:%M').hour,
                                    prefix='hour')