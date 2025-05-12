import pandas as pd


def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies["iata_approved"] = _is_true(companies["iata_approved"])
    companies["company_rating"] = _parse_percentage(companies["company_rating"])
    return companies


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles


def create_model_input_table(
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    rated_shuttles = rated_shuttles.drop("id", axis=1)
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()
    return model_input_table

# src/eco2mix_forecasting/pipelines/data_download/nodes.py
from src.superman_mspr.pipelines.data_science.nodes import fetch_eCO2mix_data, convert_all_xls_eCO2mix_data

def fetch_data_node(destination_folder: str):
    """ Télécharge les fichiers eCO2mix """
    return fetch_eCO2mix_data(destination_folder)

def convert_data_node(xls_path: str, csv_path: str):
    """ Convertit les fichiers XLS en CSV """
    return convert_all_xls_eCO2mix_data(xls_path, csv_path)

# src/eco2mix_forecasting/pipelines/data_preprocessing/nodes.py
import pandas as pd
from src.superman_mspr.pipelines.data_science.nodes import clean_data, merge_data

def clean_data_node(raw_data: pd.DataFrame) -> pd.DataFrame:
    """ Nettoie les données brutes """
    return clean_data(raw_data)

def merge_data_node(data_1: pd.DataFrame, data_2: pd.DataFrame) -> pd.DataFrame:
    """ Fusionne deux ensembles de données """
    return merge_data(data_1, data_2)


