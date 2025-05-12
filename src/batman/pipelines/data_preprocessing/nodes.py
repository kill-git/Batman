# nodes.py

from eco2mix_forecasting.core.data.preprocessing import (
    concat_eCO2mix_annual_data,
    concat_eCO2mix_tempo_data,
    merge_eCO2mix_data,
    preprocess_eCO2mix_data
)

def concat_annual_node(path_annual: str):
    """
    Node pour concaténer les données annuelles eCO2mix
    
    Args:
        path_annual: Chemin vers les fichiers de données annuelles
        
    Returns:
        DataFrame contenant les données annuelles concaténées
    """
    return concat_eCO2mix_annual_data(path_annual)

def concat_tempo_node(path_tempo: str):
    """
    Node pour concaténer les données Tempo (RTE)
    
    Args:
        path_tempo: Chemin vers les fichiers de données Tempo
        
    Returns:
        DataFrame contenant les données Tempo concaténées
    """
    return concat_eCO2mix_tempo_data(path_tempo)

def merge_data_node(annual_df, tempo_df):
    """
    Node pour fusionner les données annuelles et Tempo
    
    Args:
        annual_df: DataFrame des données annuelles
        tempo_df: DataFrame des données Tempo
        
    Returns:
        DataFrame fusionné
    """
    return merge_eCO2mix_data(annual_df, tempo_df)

def clean_merged_data_node(df):
    """
    Node pour nettoyer et prétraiter les données fusionnées
    
    Args:
        df: DataFrame fusionné à nettoyer
        
    Returns:
        DataFrame nettoyé et prétraité
    """
    return preprocess_eCO2mix_data(df)