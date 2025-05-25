import pandas as pd
import re
import glob
from datetime import datetime
import os
import requests
from zipfile import ZipFile
from io import BytesIO
from urllib.parse import urlparse, parse_qs

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import logging

logger = logging.getLogger(__name__)

def fetch_eCO2mix_data(destination_folder: str = "../data/external/") -> None:
    """
    Télécharge et extrait tous les fichiers ZIP de la div .eco2mix-download-data
    sur la page des indicateurs éCO2mix de RTE.
    
    Args:
        destination_folder (str): Dossier où extraire les fichiers.
    """
    os.makedirs(destination_folder, exist_ok=True)

    page_url = "https://www.rte-france.com/eco2mix/telecharger-les-indicateurs"

    # Setup Selenium headless
    chrome_opts = Options()
    chrome_opts.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_opts)

    try:
        logger.info(f"🌐 Chargement de {page_url}")
        driver.get(page_url)
        driver.implicitly_wait(10)

        # Sélectionne tous les liens <a> de la div cible
        anchors = driver.find_elements(By.CSS_SELECTOR, "div.eco2mix-download-data a")

        # Filtre : on garde les .zip et les calendriers Tempo
        download_links = []
        for a in anchors:
            href = a.get_attribute("href")
            if not href:
                continue
            low = href.lower()
            if low.endswith(".zip") or "downloadcalendriertempo" in low:
                download_links.append(href)

        logger.info(f"🔗 {len(download_links)} liens détectés (ZIP + Tempo).")

        for url in download_links:
            parsed = urlparse(url)
            name = os.path.basename(parsed.path)
            
            # --- Traitement ZIP ---
            logger.info(f"⬇️  Téléchargement ZIP : {name}")
            resp = requests.get(url)
            resp.raise_for_status()
            with ZipFile(BytesIO(resp.content)) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    fname = os.path.basename(member.filename)
                    if not fname:
                        continue
                    out_path = os.path.join(destination_folder, fname)
                    with zf.open(member) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())
            logger.info(f"✅ Contenu de {name} extrait dans {destination_folder}")

    finally:
        driver.quit()

def convert_xls_eCO2mix_to_csv(input_path: str, output_path: str) -> None:
    """
    Convertit un fichier .xls (format texte tabulé) en .csv propre.

    Cette fonction effectue les opérations suivantes :
    1. Lit un fichier .xls en supposant qu'il est encodé en cp1252.
    2. Remplace les tabulations ('\\t') par des virgules (',').
    3. Supprime une virgule finale superflue sur chaque ligne sauf l’en-tête.
    4. Supprime la ligne de disclaimer RTE si elle est présente.
    5. S'assure que chaque ligne du fichier de sortie se termine par un seul saut de ligne.
    6. Enregistre le résultat final encodé en UTF-8.

    Paramètres :
    ----------
    input_path : str
        Chemin vers le fichier .xls d'entrée (fichier texte avec tabulations).
    output_path : str
        Chemin vers le fichier .csv nettoyé à produire.

    Exemple :
    --------
    convert_xls_to_clean_csv(
        input_path='../data/raw/eCO2mix_RTE_En-cours-TR.xls',
        output_path='../data/processed/eCO2mix_clean.csv'
    )
    """
    with open(input_path, 'r', encoding='cp1252') as f:
        lines = f.readlines()

    cleaned_lines = []
    for i, line in enumerate(lines):
        # Supprimer ligne de disclaimer si elle correspond
        if "RTE ne pourra" in line:
            continue
        if "L'ensemble des informations disponibles" in line:
            continue

        # Remplacer les tabulations par des virgules
        line = line.replace('\t', ',')

        # Supprimer une éventuelle virgule en trop à la fin (sauf pour l'en-tête)
        if i > 0:
            line = re.sub(r',\s*$', '', line)

        # S'assurer que chaque ligne finit par un seul \n
        line = line.rstrip() + '\n'

        cleaned_lines.append(line)

    # Sauvegarde dans un nouveau fichier CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    logger.info(f"✅ Fichier converti et nettoyé : {output_path}")

def convert_all_xls_eCO2mix_data(xls_path: str, csv_path: str) -> None:
    """
    Convertit tous les fichiers .xls dans le dossier d'entrée en fichiers .csv nettoyés.
    Paramètre : 
    xls_path : str
        Chemin vers le dossier contenant les fichiers .xls à convertir.
    csv_path : str
        Chemin vers le dossier où les fichiers .csv nettoyés seront enregistrés.
    """
    xls_files = glob.glob(os.path.join(xls_path, "*.xls"))
    raw_csv_paths = []
    for xls_path in xls_files:
        # Convertit les fichiers XLS en CSV
        base_name = os.path.basename(xls_path).replace(".xls", ".csv")
        out_path = os.path.join(csv_path, base_name)
        convert_xls_eCO2mix_to_csv(xls_path, out_path)
        raw_csv_paths.append(out_path)

def load_data(filepath: str, encoding: str = 'utf-8', sep: str = ",") -> pd.DataFrame:
    """
    Charge les données depuis un fichier TXT avec des champs séparés par une tabulation,
    et convertit la colonne de date.

    Paramètres :
    - filepath : chemin du fichier TXT.
    - date_column : nom de la colonne contenant les dates.

    Retourne :
    - DataFrame contenant les données avec la colonne date convertie en datetime.
    """
    try:
        data = pd.read_csv(filepath, sep=sep, encoding=encoding, low_memory=False, dtype=str)
        return data
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données : {e}")
        raise e

def preprocess_annual_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraite les données annuelles :
    - Convertit la colonne 'Date' & 'Heure' en 'Datetime'.
    - Supprime les colonnes inutiles.

    Paramètres :
    - df : DataFrame à prétraiter.

    Retourne :
    - DataFrame prétraité.
    """
    
    assert 'Consommation' in df.columns, "La colonne 'Consommation' est requise."
    assert 'Date' in df.columns, "La colonne 'Date' est requise."
    assert 'Heures' in df.columns, "La colonne 'Heures' est requise."
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%Y-%m-%d")
    # Creation de Datetime
    df['Datetime'] = pd.to_datetime(df['Date'].dt.strftime("%Y-%m-%d") + " " + df['Heures'], 
                                format="%Y-%m-%d %H:%M", errors='coerce')
    df = df.drop_duplicates()
    # Supprimer les colonnes inutiles
    columns_to_drop = df.columns.difference(['Date', 'Heures', 'Datetime', 'Consommation'])
    df = df.drop(columns=columns_to_drop, errors='ignore', axis=1)
    before = len(df)
    df = df.dropna(subset=['Consommation'])
    after = len(df)
    logger.info(f"⚠️  Suppression de {before - after} lignes sans consommation.")
    return df

def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Effectue un encodage one-hot sur une colonne donnée d'un DataFrame.

    Paramètres :
    - df : DataFrame à encoder.
    - column : nom de la colonne à encoder.

    Retourne :
    - DataFrame avec la colonne encodée.
    """
    if column not in df.columns:
        raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")
    
    one_hot = pd.get_dummies(df[column], prefix=column)
    df = df.drop(column, axis=1)
    df = pd.concat([df, one_hot], axis=1)
    return df

def preprocess_tempo_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraite la donnée TEMPO de eCO2mix : 
    - Applique le one-hot encoding sur la colonne TEMPO (i.e 'Type de jour TEMPO').
    
    Paramètres :
    - df : DataFrame TEMPO à prétraiter.
    
    Retourne : 
    - DataFrame prétraité.
    """
    assert 'Type de jour TEMPO' in df.columns, "La colonne 'Type de jour TEMPO' est requise."
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format="%Y-%m-%d")
    return one_hot_encode(df, 'Type de jour TEMPO')

def concat_eCO2mix_annual_data(path_annual: str) -> pd.DataFrame:
    """
    Concatène tous les fichiers annuels de eCO2mix dans un seul DataFrame.

    Paramètres :
    - path_annual : chemin vers le dossier contenant les fichiers annuels.

    Retourne :
    - DataFrame concaténé.
    """
    pattern_annuel  = os.path.join(path_annual, "eCO2mix_RTE_Annuel-Definitif_20*.csv")
    pattern_consol  = os.path.join(path_annual, "eCO2mix_RTE_En-cours-Consolide.csv")
    pattern_realtime = os.path.join(path_annual, "eCO2mix_RTE_En-cours-TR.csv")
    annual_files = sorted(glob.glob(pattern_annuel)) + [pattern_consol, pattern_realtime]
    list_df_annual = []
    # Assertion pour vérifier que tous les fichiers existent
    missing_files = [f for f in annual_files if not os.path.isfile(f)]
    assert not missing_files, f"Fichiers manquants : {missing_files}"
    logger.info("Lecture des fichiers annuels :")
    for file in annual_files:
        logger.info(f"  - {file}")

        df = load_data(file, encoding="utf-8")
        if df is not None:
            # Vous pouvez également ajouter des transformations propres aux fichiers annuels ici si nécessaire
            list_df_annual.append(df)

    # Concaténation de toutes les données annuelles en un seul DataFrame
    df_annual = pd.concat(list_df_annual, axis=0, ignore_index=False)
    # Remise à zéro de l'index si besoin (sinon, vous conservez l'index fourni par le fichier source)
    df_annual.reset_index(drop=True, inplace=True)
    return df_annual

def concat_eCO2mix_tempo_data(path_tempo: str) -> pd.DataFrame:
    """
    Concatène tous les fichiers TEMPO de eCO2mix dans un seul DataFrame.
    Paramètres :
    - path_tempo : chemin vers le dossier contenant les fichiers tempo.

    Retourne :
    - DataFrame concaténé.
    """
    tempo_pattern = os.path.join(path_tempo, "eCO2mix_RTE_tempo*.csv")
    tempo_files = sorted(glob.glob(tempo_pattern))
    list_df_tempo = []
    # Assertion pour vérifier que tous les fichiers existent
    assert tempo_files, f"Aucun fichier trouvé avec le pattern : {tempo_pattern}"
    
    logger.info("Lecture des fichiers tempo :")
    for file in tempo_files:
        logger.info(f"  - {file}")
        try:
            df = load_data(file, encoding="utf-8")
            # Vous pouvez également ajouter des transformations propres aux fichiers tempo ici si nécessaire
            list_df_tempo.append(df)
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier {file}: {e}")
    df_tempo = pd.concat(list_df_tempo, axis=0, ignore_index=True)
    return df_tempo

def merge_eCO2mix_data(df_annual: pd.DataFrame, df_tempo: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les données annuelles et TEMPO sur la colonne 'Date'.

    Paramètres :
    - df_annual : DataFrame des données annuelles.
    - df_tempo : DataFrame des données TEMPO.

    Retourne :
    - DataFrame fusionné.
    """
    assert 'Date' in df_annual.columns, "La colonne 'Date' est requise dans df_annual."
    assert 'Date' in df_tempo.columns, "La colonne 'Date' est requise dans df_tempo."
    
    df_tempo['Date'] = pd.to_datetime(df_tempo['Date'], errors='coerce', format="%Y-%m-%d")
    df_annual['Date'] = pd.to_datetime(df_annual['Date'], errors='coerce', format="%Y-%m-%d")
    
    merged_df = pd.merge(df_annual, df_tempo, on='Date', how='left')
    
    return merged_df

def preprocess_eCO2mix_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraite les données fusionnées de eCO2mix :
    - Supprime les lignes ou il existe aucune valeur dans les colonnes 'Type de jour TEMPO_BLEU', 'Type de jour TEMPO_BLANC', 'Type de jour TEMPO_ROUGE'.
    - indexé par 'Datetime'.

    Paramètres :
    - df : DataFrame fusionné à prétraiter.
    
    Retourne:
    - DataFrame prétraité.
    """
    assert 'Datetime' in df.columns, "La colonne 'Datetime' est requise."
    assert 'Consommation' in df.columns, "La colonne 'Consommation' est requise."
    assert 'Type de jour TEMPO_BLEU' in df.columns, "La colonne 'Type de jour TEMPO_BLEU' est requise."
    assert 'Type de jour TEMPO_BLANC' in df.columns, "La colonne 'Type de jour TEMPO_BLANC' est requise."
    assert 'Type de jour TEMPO_ROUGE' in df.columns, "La colonne 'Type de jour TEMPO_ROUGE' est requise."
    
    # Supprime les lignes où il n'y a aucune valeur dans les colonnes TEMPO
    before = len(df)
    df = df.dropna(subset=['Type de jour TEMPO_BLEU', 'Type de jour TEMPO_BLANC', 'Type de jour TEMPO_ROUGE'], how='all')
    after = len(df)
    logger.info(f"⚠️  Suppression de {before - after} lignes sans type de jour TEMPO.")
    df['Consommation'] = pd.to_numeric(df['Consommation'], errors='coerce')
    # Indexe par Datetime
    # df.set_index('Datetime', inplace=True) # Non necessaire, artefact d'une ancienne implémentation
    
    return df

def preprocess_eCO2mix_data_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraite les données de eCO2mix après ingénierie des caractéristiques :
        - Supprime les lignes qui ne possède pas les données features lag, rolling mean vide.
    
    Retourne :
        - DataFrame prétraité.
    """
    engineered_cols = [col for col in df.columns if any(
        kw in col.lower() for kw in ['lag', 'rolling'])]

    assert engineered_cols, "Aucune colonne d'ingénierie détectée dans le DataFrame."

    # Suppression des lignes contenant des NaN dans les colonnes d'ingénierie
    before = len(df)
    df_cleaned = df.dropna(subset=engineered_cols)
    after = len(df_cleaned)
    logger.info(f"⚠️  Suppression de {before - after} lignes sans données d'ingénierie.")
    return df_cleaned

def split_features_target(data: pd.DataFrame, target: str) -> tuple:
    """
    Sépare les caractéristiques et la cible dans un DataFrame.
    """
    assert target in data.columns, f"La colonne cible '{target}' n'existe pas dans le DataFrame."
    assert 'Datetime' in data.columns, "La colonne 'Datetime' est requise pour l'indexation."
    assert 'Date' in data.columns, "La colonne 'Date' est requise pour la séparation des caractéristiques."
    assert 'Heures' in data.columns, "La colonne 'Heures' est requise pour la séparation des caractéristiques."
    
    # Trie par ordre chronologique
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
    data_sorted = data.sort_values("Datetime").copy()
    
    X = data_sorted.drop(columns=[target, "Date", "Heures", "Datetime"], errors="ignore")
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except Exception:
            logger.warning(f"⚠️ La colonne '{col}' ne peut pas être convertie en numerique. Elle reste de type {X[col].dtype}")
            
    y = pd.DataFrame(data_sorted[target])
    y.columns = [target]
    y = y.astype("float32")
    logger.info(f"🔧 Données triées par 'Datetime'. Séparation X (shape={X.shape}) / y (len={len(y)})")
    return X, y

def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """
    Sépare les données en ensembles d'entraînement et de test.
    """
    n_samples = len(X)
    split_index = int(n_samples * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    y_train = pd.DataFrame(y_train).reset_index(drop=True)
    y_test = pd.DataFrame(y_test).reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test

def normalize_features(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = "standard") -> tuple:
    """
    Normalise les caractéristiques d'entraînement et de test.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Méthode de normalisation inconnue : {method}")
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    logger.info(f"🔧 Normalisation des données avec {method} scaling.")
    return X_train_scaled, X_test_scaled, scaler