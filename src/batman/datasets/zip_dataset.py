# src/batman/datasets/zip_dataset.py

from kedro.io.core import AbstractVersionedDataset, Version
from pathlib import Path, PurePosixPath
from typing import Any, Dict
import shutil

class ZipDataset(AbstractVersionedDataset[Path, Path]):
    def __init__(self, filepath: str, version: Version | None = None):
        super().__init__(filepath=PurePosixPath(filepath), version=version)

    def _load(self) -> Path:
        load_path = Path(self._get_load_path())
        if not load_path.is_file():
            raise FileNotFoundError(f"Zip file not found at {load_path}")
        return load_path  # ← renvoie le chemin du .zip

    def _save(self, data: Path) -> None:
        save_path = Path(self._get_save_path())
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # On copie le fichier zip fourni par le node
        shutil.copy(data, save_path)

    def _describe(self) -> Dict[str, Any]:
        return {"filepath": str(self._filepath)}

    def _exists(self) -> bool:
        return Path(self._get_load_path()).is_file()