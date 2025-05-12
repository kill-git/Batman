import shap
import matplotlib.pyplot as plt
import seaborn as sns

def explain_linear_model_shap(model, X_train, X_test, max_display=10, sample_index=0):
    """
    Explique un modèle LinearRegression avec SHAP (LinearExplainer).
    
    Paramètres :
    - model : un modèle scikit-learn LinearRegression déjà entraîné.
    - X_train : données d'entraînement (non standardisées).
    - X_test : données à expliquer (non standardisées).
    - max_display : nombre max de variables à afficher dans les plots.
    - sample_index : index de la ligne pour l’explication locale.

    Retourne :
    - shap_values : objets SHAP calculés pour X_test.
    """
    print("Création de l'explainer SHAP pour modèle linéaire...")
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")

    print("Calcul des valeurs SHAP...")
    shap_values = explainer(X_test)

    print("Affichage des plots SHAP...")

    # Résumé global
    shap.summary_plot(shap_values, X_test, max_display=max_display)

    # Barres de contribution globale
    shap.plots.bar(shap_values, max_display=max_display)

    # Explication d’une ligne spécifique
    print(f"Explication locale pour l'observation {sample_index}")
    shap.plots.waterfall(shap_values[sample_index], max_display=max_display)

    return shap_values