# src/batman/datasets/zip_dataset.py

from kedro.io.core import (
    AbstractVersionedDataset, 
    Version, 
    get_filepath_str, 
    get_protocol_and_path,
    VersionNotFoundError
)
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional
from tempfile import NamedTemporaryFile
import shutil
from fsspec.implementations.local import LocalFileSystem
import fsspec
import os

class ZipDataset(AbstractVersionedDataset[Path, Path]):
    """
    Dataset versionné pour gérer un .zip sur S3 (MinIO) ou en local.
    """

    def __init__(
        self,
        filepath: str,                    # ex: "s3://batman-data/01_raw/eco2mix.zip"
        version: Version | None = None,
        credentials: Optional[Dict[str, Any]] = None,
        fs_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
        load_args: Optional[Dict[str, Any]] = None,
    ):
        # 1) Extraire protocole et "chemin nu" (clé S3 ou chemin local).
        protocol, path_str = get_protocol_and_path(filepath)
        #    → protocol == "s3", path_str == "batman-data/01_raw/eco2mix.zip"
        self._protocol = protocol

        # 2) Construire le filesystem AVANT d'appeler super(),
        #    car il nous sert de exists_function + glob_function.
        #    On passe credentials et fs_args éventuels (ex: endpoint_url, etc.)
        self._credentials = credentials or {}
        self._fs_args = fs_args or {}
        self._fs = fsspec.filesystem(self._protocol, **self._credentials, **self._fs_args)

        # 3) Appel à AbstractVersionedDataset.__init__() en lui donnant :
        #    - le PurePosixPath (chemin "nu" sans protocole)
        #    - la version
        #    - la fonction exists() de fsspec
        #    - la fonction glob() de fsspec
        super().__init__(
            filepath=PurePosixPath(path_str),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

        # 4) Conserver aussi save_args / load_args pour plus tard
        self._save_args = save_args or {}
        self._load_args = load_args or {}

    def _get_load_path(self) -> PurePosixPath | None:
        """
        Surcharge : si aucune version n'existe encore, renvoyer None
        plutôt que de laisser remonter VersionNotFoundError.
        """
        try:
            return super()._get_load_path()
        except VersionNotFoundError:
            return None

    def _load(self) -> Path:
        """
        1) On récupère la PurePosixPath versionnée (ex:
           "batman-data/01_raw/eco2mix.zip/2025-06-01TXX.XX.XX.XXXZ/eco2mix.zip").
        2) Si None → on lève FileNotFoundError.
        3) On convertit en 'raw_key' (as_posix()) pour que self._fs.open()
           sache où trouver l'objet sur S3/MinIO.
        4) On télécharge dans un temporay file et on retourne le Path local.
        """
        versioned_path = self._get_load_path()
        if versioned_path is None:
            raise FileNotFoundError(f"Aucune version disponible pour le ZIP '{self._filepath}'.")

        raw_key = versioned_path.as_posix()
        if not self._fs.exists(raw_key):
            raise FileNotFoundError(f"Le fichier ZIP n'existe pas à '{raw_key}'")

        # Téléchargement dans un tmp
        tmp = NamedTemporaryFile(prefix="kedro_zip_", suffix=".zip", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        with self._fs.open(raw_key, mode="rb", **self._load_args) as src, \
             open(tmp_path, mode="wb") as dst:
            shutil.copyfileobj(src, dst)

        return tmp_path

    def _save(self, data: Path) -> None:
        """
        1) 'data' est un Path local pointant vers un fichier .zip valide.
        2) On récupère la PurePosixPath versionnée (via _get_save_path()).
        3) On convertit cette PurePosixPath en 'raw_key' pour S3.
        4) Si protocol=="file", on crée le dossier parent local.
        5) On effectue la copie locale->S3 (ou local->local).
        """
        save_path = self._get_save_path()
        raw_key = save_path.as_posix()

        if self._protocol == "file":
            # Si j'écris en local, créer le dossier parent
            parent_dir = Path(PurePosixPath(raw_key).parent)
            parent_dir.mkdir(parents=True, exist_ok=True)

        # Copie depuis le Path local 'data' vers MinIO/S3 ou vers le fs local
        with open(data, "rb") as src, \
             self._fs.open(raw_key, "wb", **self._save_args) as dst:
            shutil.copyfileobj(src, dst)

    def _exists(self) -> bool:
        """
        Retourne True si une version existe. 
        On appelle _get_load_path() (qui peut renvoyer None si jamais sauvegardé).
        Si None -> False, sinon, on vérifie via self._fs.exists(raw_key).
        """
        versioned_path = self._get_load_path()
        if versioned_path is None:
            return False

        raw_key = versioned_path.as_posix()
        return self._fs.exists(raw_key)

    def _describe(self) -> Dict[str, Any]:
        return {
            "filepath": str(self._filepath),   # ex: "batman-data/01_raw/eco2mix.zip"
            "protocol": self._protocol,        # "s3" ou "file"
            "versioned": True,
            "fs_args": bool(self._fs_args),
            "credentials": bool(self._credentials),
        }