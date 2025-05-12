import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set(style="whitegrid")  # Style général pour seaborn

def plot_time_series(
    df: pd.DataFrame,
    value_column: str,
    title: str = "Série Temporelle",
    xlabel: str = "Date",
    ylabel: str = "Valeur",
    figsize: tuple = (12, 6),
    marker: str = 'o',
    linestyle: str = '-',
    color: str = None
):
    """
    Trace une série temporelle à partir d'un DataFrame indexé par datetime.

    Paramètres :
    - df : DataFrame avec un index de type datetime.
    - value_column : Colonne contenant les valeurs à tracer.
    - title, xlabel, ylabel : Titres et labels.
    - figsize : Taille de la figure.
    - marker, linestyle, color : Personnalisation du tracé.
    """
    plt.figure(figsize=figsize)
    sns.lineplot(data=df, x=df.index, y=value_column, marker=marker, linestyle=linestyle, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    title: str = "Histogramme",
    xlabel: str = None,
    ylabel: str = "Fréquence",
    figsize: tuple = (10, 6),
    color: str = "blue"
):
    """
    Affiche un histogramme pour une colonne du DataFrame.

    Paramètres :
    - df : DataFrame contenant les données.
    - column : Colonne à tracer.
    - bins : Nombre de barres.
    - title, xlabel, ylabel : Titres et labels.
    - figsize : Taille de la figure.
    - color : Couleur des barres.
    """
    plt.figure(figsize=figsize)
    sns.histplot(df[column].dropna(), bins=bins, kde=False, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: list = None,
    title: str = "Matrice de Corrélation",
    annot: bool = True,
    cmap: str = "coolwarm",
    fmt: str = ".2f",
    figsize: tuple = (12, 10)
):
    """
    Affiche une heatmap de la matrice de corrélation d’un DataFrame.

    Paramètres :
    - df : DataFrame contenant les données numériques.
    - columns : Liste de colonnes à inclure dans la corrélation (optionnel).
    - title : Titre du graphique.
    - annot : Afficher les valeurs dans les cases.
    - cmap : Palette de couleurs.
    - fmt : Format des annotations.
    - figsize : Taille de la figure.
    """
    if columns:
        data = df[columns]
    else:
        data = df.select_dtypes(include="number")  # Par défaut, toutes les colonnes numériques

    corr = data.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def plot_target_correlations(
    df: pd.DataFrame,
    target_column: str,
    columns: list = None,
    method: str = "pearson",
    sort_ascending: bool = False,
    figsize: tuple = (10, 6),
    color_palette: str = "coolwarm",
    title: str = "Corrélation avec la variable cible",
    include_datetime_index: bool = False,
    datetime_column_name: str = "date_numeric"
):
    """
    Affiche un graphique en barres représentant la corrélation entre une variable cible
    et un ensemble de dimensions numériques.

    Paramètres :
    - df : DataFrame source contenant les données.
    - target_column : Nom de la colonne cible à corréler avec les autres dimensions.
    - columns : Liste optionnelle des colonnes à utiliser pour la corrélation.
                Si None, toutes les colonnes numériques sont prises en compte.
    - method : Méthode de corrélation à utiliser ('pearson', 'spearman', 'kendall').
    - sort_ascending : Si True, trie les corrélations de la plus faible à la plus forte.
    - figsize : Tuple pour la taille de la figure matplotlib.
    - color_palette : Palette de couleurs Seaborn à appliquer aux barres.
    - title : Titre du graphique.
    - include_datetime_index : Si True, inclut l’index datetime comme variable numérique.
    - datetime_column_name : Nom de la colonne temporaire créée pour représenter l’index datetime.

    Affiche :
    - Un graphique en barres horizontales triées, avec un gradient de couleur basé sur la palette donnée.
    """

    # Copie du DataFrame pour éviter toute modification de l’original
    df_corr = df.copy()

    # Conversion de l’index datetime en float si demandé
    if include_datetime_index and isinstance(df_corr.index, pd.DatetimeIndex):
        df_corr[datetime_column_name] = df_corr.index.astype("int64") / 1e9  # secondes depuis epoch

    # Sélection des colonnes numériques par défaut
    numeric_df = df_corr.select_dtypes(include="number")

    # Si l'utilisateur a spécifié des colonnes
    if columns:
        # On ajoute la colonne cible si elle n'est pas dans la liste
        if target_column not in columns:
            columns = columns + [target_column]

        # On filtre les colonnes qui existent réellement dans le DataFrame
        valid_columns = [col for col in columns if col in df_corr.columns]

        # On tente de convertir toutes les colonnes sélectionnées en numérique
        selected = df_corr[valid_columns].apply(pd.to_numeric, errors='coerce')

        # On supprime les colonnes entièrement non convertibles (NaN)
        selected = selected.dropna(axis=1, how="all")

        # Vérification que la colonne cible est bien présente après conversion
        if target_column not in selected.columns:
            raise ValueError(f"La colonne cible '{target_column}' n’a pas pu être convertie en numérique.")

        # On remplace le DataFrame numérique de base par cette sélection nettoyée
        numeric_df = selected

    # Calcul de la matrice de corrélation avec la méthode choisie
    correlations = numeric_df.corr(method=method)[target_column].drop(target_column)

    # Tri des résultats selon la préférence utilisateur
    correlations = correlations.sort_values(ascending=sort_ascending)

    # Affichage du graphique
    plt.figure(figsize=figsize)
    sns.barplot(
        x=correlations.values,
        y=correlations.index,
        hue=correlations.index,         # Pour assigner les couleurs
        palette=color_palette,
        dodge=False,
        legend=False                    # Pas de légende inutile
    )
    plt.title(title)
    plt.xlabel(f"Corrélation ({method})")
    plt.ylabel("Variables")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.show()

def plot_consumption_by_time_granularity(
    df: pd.DataFrame,
    value_col: str = "Consommation",
    granularity: str = "month",  # Options : 'month', 'quarter', 'hour', 'weekday'
    cmap: str = "coolwarm",
    agg_func: str = "mean",
    figsize: tuple = (10, 6),
    title: str = None
):
    """
    Affiche un graphique montrant les niveaux de consommation selon une granularité temporelle.

    Paramètres :
    - df : DataFrame avec un index Datetime nommé 'Datetime'.
    - value_col : Nom de la colonne contenant la consommation électrique.
    - granularity : Niveau temporel pour grouper ('month', 'quarter', 'hour', 'weekday').
    - cmap : Palette de couleurs pour la heatmap.
    - agg_func : Fonction d’agrégation ('mean', 'sum', etc.).
    - figsize : Taille de la figure.
    - title : Titre du graphique.
    """
    df_plot = df.copy()

    # Vérifie que l'index est bien de type datetime
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        raise ValueError("L'index du DataFrame doit être un DatetimeIndex nommé 'Datetime'.")

    dt = df_plot.index

    # Détermination de la granularité
    if granularity == "month":
        df_plot["__time_group__"] = dt.month
        x_labels = ['Janv', 'Fév', 'Mars', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sept', 'Oct', 'Nov', 'Déc']
    elif granularity == "quarter":
        df_plot["__time_group__"] = dt.quarter
        x_labels = ['T1', 'T2', 'T3', 'T4']
    elif granularity == "weekday":
        df_plot["__time_group__"] = dt.weekday
        x_labels = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    elif granularity == "hour":
        df_plot["__time_group__"] = dt.hour
        x_labels = [str(h) for h in range(24)]
    else:
        raise ValueError(f"Granularité non supportée : {granularity}")

    # Agrégation
    grouped = df_plot.groupby("__time_group__")[value_col].agg(agg_func)
    # Création du graphique
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(
        grouped.values.reshape(1, -1),
        cmap=cmap,
        annot=True,
        fmt=".0f",
        xticklabels=x_labels,
        yticklabels=[agg_func.capitalize()]
    )
    for text in heatmap.texts:
        text.set_rotation(90)
    plt.title(title or f"{agg_func.capitalize()} de la consommation par {granularity}")
    plt.xlabel(granularity.capitalize())
    plt.ylabel("")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_model_predictions(
    y_true,
    y_pred,
    datetimes=None,
    title: str = "Prédictions vs Réel",
    xlabel: str = "Date",
    ylabel: str = "Valeur",
    figsize: tuple = (14, 6),
    num_points: int = 200,
    color_true: str = "blue",
    color_pred: str = "orange",
    linestyle_true: str = "-",
    linestyle_pred: str = "--"
):
    """
    Trace les prédictions d’un modèle de régression par rapport aux valeurs réelles sur une échelle temporelle.
    Les points affichés sont espacés uniformément sur toute la série.

    Paramètres :
    - y_true : array-like
        Valeurs réelles de la variable cible.
    - y_pred : array-like
        Valeurs prédites par le modèle.
    - datetimes : array-like ou None
        Série temporelle associée aux valeurs. Si None, utilise un index range par défaut.
    - title : str
        Titre du graphique.
    - xlabel, ylabel : str
        Titres des axes X et Y.
    - figsize : tuple
        Taille de la figure matplotlib.
    - num_points : int
        Nombre de points à afficher (échantillonnés de manière espacée).
    - color_true, color_pred : str
        Couleurs personnalisées pour les lignes réelle et prédite.
    - linestyle_true, linestyle_pred : str
        Styles de ligne pour les courbes réelle et prédite.
    """
    # Convertir en Series
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    
    if datetimes is not None:
        x = pd.Series(datetimes).reset_index(drop=True)
    else:
        x = pd.Series(range(len(y_true)))

    # Échantillonnage uniforme
    if num_points < len(y_true):
        indices = np.linspace(0, len(y_true) - 1, num=num_points, dtype=int)
        y_true = y_true.iloc[indices]
        y_pred = y_pred.iloc[indices]
        x = x.iloc[indices]

    # Création d’un DataFrame pour seaborn
    df_plot = pd.DataFrame({
        "Date": x,
        "Réel": y_true,
        "Prédiction": y_pred
    })

    plt.figure(figsize=figsize)
    sns.lineplot(data=df_plot, x="Date", y="Réel", label="Réel", color=color_true, linestyle=linestyle_true)
    sns.lineplot(data=df_plot, x="Date", y="Prédiction", label="Prédiction", color=color_pred, linestyle=linestyle_pred)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
    

def plot_residual_errors(y_true, y_pred, title="Erreurs Résiduelles"):
    """
    Trace les erreurs résiduelles d'un modèle de régression.
    
    Paramètres :
    - y_true : array-like
        Valeurs réelles de la variable cible.
    - y_pred : array-like
        Valeurs prédites par le modèle.
    - title : str
        Titre du graphique.
    
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(12, 4))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Erreur (y_test - y_pred)")
    plt.grid(True)
    plt.show()