# run-mlflow.ps1
# Force l'encodage UTF-8 dans la console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Récupère le dossier courant (racine de Batman)
$projectRoot = Get-Location

# Définit le chemin absolu vers le dossier de stockage mlflow (../mlflow_storage/mlruns)
$mlflowStorage = Resolve-Path "$projectRoot\..\mlflow_storage\mlruns"

# Crée le dossier s'il n'existe pas
if (-not (Test-Path $mlflowStorage)) {
    New-Item -ItemType Directory -Path $mlflowStorage -Force | Out-Null
    Write-Host "Dossier créé : $mlflowStorage"
}

# Lance MLflow UI avec ce dossier comme backend
Write-Host "Lancement de MLflow UI à l'adresse http://127.0.0.1:5000"
mlflow ui --backend-store-uri "file:///$mlflowStorage" --port 5001