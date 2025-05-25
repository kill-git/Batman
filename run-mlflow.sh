#!/bin/bash

# Détermine le chemin absolu du dossier Batman (le dossier où ce script est lancé)
BATMAN_ROOT=$(pwd)

# Chemin du dossier de stockage mlflow (un niveau au-dessus de Batman)
MLFLOW_STORAGE="$(realpath "$BATMAN_ROOT/../mlflow_storage/mlruns")"

# Crée le dossier mlflow_storage/mlruns s'il n'existe pas
if [ ! -d "$MLFLOW_STORAGE" ]; then
  mkdir -p "$MLFLOW_STORAGE"
  echo "📁 Dossier créé : $MLFLOW_STORAGE"
fi

# Lancer MLflow UI avec ce dossier comme backend
echo "🚀 Lancement de MLflow UI à l'adresse http://127.0.0.1:5000"
mlflow ui --backend-store-uri "file://$MLFLOW_STORAGE" --port 5000
