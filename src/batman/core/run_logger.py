import json
import os
from datetime import datetime

def log_run(run_info, file_path="results.json"):
    """
    Sauvegarde un run d'entraînement dans un fichier json.
    """
    run_info['timestamp'] = datetime.now().isoformat()

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            all_runs = json.load(f)
    else:
        all_runs = []

    all_runs.append(run_info)

    with open(file_path, "w") as f:
        json.dump(all_runs, f, indent=4)

    print(f"🔖 Run sauvegardé dans {file_path}")
