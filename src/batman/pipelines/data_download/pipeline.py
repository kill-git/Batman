from kedro.pipeline import Pipeline, node
from .nodes import fetch_data_node, convert_data_node

def create_pipeline(**kwargs) -> Pipeline:
    """ Crée une pipeline pour télécharger et convertir les données avec stockage MinIO."""
    return Pipeline([
        node(
            func=fetch_data_node,
            inputs="params:destination_folder",
            outputs="fetch_done",
            name="fetch_data_minio"
        ),
        node(
            func=convert_data_node,
            inputs=["params:xls_path", "params:csv_path"],  # Correction ici
            outputs=None,
            name="convert_data_minio"
        )
    ])
