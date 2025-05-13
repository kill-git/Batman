from minio import Minio
import os
import datetime
from batman.core.data_preprocessing import fetch_eCO2mix_data, convert_all_xls_eCO2mix_data

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "eco2mix-data"

# Initialisation du client MinIO
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

def fetch_data_node(destination_folder: str):
    # Création du bucket si nécessaire
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
    
    # Création d'un ID de version basé sur la date
    version_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Téléchargement des données en local (temporairement)
    fetch_eCO2mix_data(destination_folder)
    
    # Upload des fichiers sur MinIO avec versioning
    for root, dirs, files in os.walk(destination_folder):
        for file in files:
            file_path = os.path.join(root, file)
            # Préfixer avec l'ID de version
            minio_path = f"{version_id}/{file}"
            minio_client.fput_object(
                BUCKET_NAME,
                minio_path,
                file_path
            )
            print(f"⬆️  Fichier {file} uploadé sur MinIO en tant que {minio_path}")
            
            # Suppression du fichier local après upload
            os.remove(file_path)
            print(f"🗑️  Fichier local {file_path} supprimé")
    return version_id  # Retourne l'ID de version pour suivi

def convert_data_node(xls_path: str, csv_path: str):
    # Conversion des fichiers .xls en .csv
    convert_all_xls_eCO2mix_data(xls_path, csv_path)
