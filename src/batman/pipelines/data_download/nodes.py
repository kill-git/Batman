from zipfile import ZipFile
from pathlib import Path
import shutil
import tempfile
# from minio import Minio
import os
# import datetime
from batman.core.data_preprocessing import fetch_eCO2mix_data, convert_all_xls_eCO2mix_data

def fetch_data_node(temp_folder: str) -> Path:
    os.makedirs(temp_folder, exist_ok=True)
    fetch_eCO2mix_data(temp_folder)

    zip_path = Path(temp_folder).with_suffix(".zip")
    shutil.make_archive(base_name=str(zip_path.with_suffix("")), format="zip", root_dir=temp_folder)
    
    # Déplacement dans un autre dossier de sortie (ex : build/artifacts/tmp_output.zip)
    final_zip_path = Path("build") / "tmp_output.zip"
    os.makedirs(final_zip_path.parent, exist_ok=True)
    shutil.copy(zip_path, final_zip_path)

    # Nettoyage complet des fichiers temporaires
    shutil.rmtree(temp_folder)
    zip_path.unlink(missing_ok=True)  # Supprime temp.zip si encore là

    return final_zip_path

def convert_data_node(zip_path: Path) -> Path:
    temp_dir = Path("temp_convert")

    # 🔄 Nettoyage si déjà présent
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # 🆕 Création du dossier de travail
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 📦 Extraction du fichier zip
    with ZipFile(zip_path, "r") as zipf:
        zipf.extractall(temp_dir)

    # 📁 Conversion des .xls en .csv
    output_dir = temp_dir / "csvs"
    output_dir.mkdir(parents=True, exist_ok=True)
    convert_all_xls_eCO2mix_data(str(temp_dir), str(output_dir))

    # 📦 Recompression en un nouveau zip
    out_zip_path = temp_dir / "output.zip"
    with ZipFile(out_zip_path, "w") as zipf:
        for file in output_dir.glob("*.csv"):
            zipf.write(file, arcname=file.name)

    # 🗃️ Copier le zip généré dans un emplacement temporaire persistant (hors `temp_dir`)
    final_zip_path = Path(tempfile.gettempdir()) / f"converted_{out_zip_path.name}"
    shutil.copy(out_zip_path, final_zip_path)

    # 🧹 Nettoyage complet du dossier temporaire local
    shutil.rmtree(temp_dir)

    return final_zip_path

# TO DO : à supprimer après la fin de l'implémentation des environnements de dev et prod
"""
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

def convert_data_node(xls_path: str, csv_path: str, _:str):
    # Conversion des fichiers .xls en .csv
    convert_all_xls_eCO2mix_data(xls_path, csv_path)
"""