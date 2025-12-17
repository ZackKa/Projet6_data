import joblib  # Pour charger et sauvegarder des modèles scikit-learn
import bentoml  # Pour gérer le déploiement et le stockage des modèles ML


# Configuration


# Chemin vers le fichier du modèle entraîné au format joblib
MODEL_PATH = "energy_model_8cols.joblib"
# Nom sous lequel le modèle sera enregistré dans le store BentoML
BENTO_MODEL_NAME = "seattle_energy_model_8cols"

# Liste des features (colonnes) qui seront exposées à l'API pour faire des prédictions
API_FEATURES = [
    "PropertyGFATotal",
    "NumberofFloors",
    "IsLargeBuilding",
    "FloorsPer1000GFA",
    "PropertyGFAParking",
    "BuildingAge",
    "NumberofBuildings",
    "NumUseTypes"
]


# Chargement du modèle sklearn


# Recharge le pipeline exactement comme il était entraîné
# Cela inclut toutes les transformations, encodages, et le modèle RandomForest
model = joblib.load(MODEL_PATH)


# Sauvegarde dans BentoML


# Enregistre le modèle dans le store BentoML pour le déploiement ultérieur
bentoml.sklearn.save_model(
    name=BENTO_MODEL_NAME,  # Nom du modèle dans BentoML
    model=model,            # Le modèle à enregistrer dans le store BentoML pour le déploiement ultérieur et l'utilisation dans l'API
    metadata={              # Informations supplémentaires pour décrire le modèle
        "target": "SiteEnergyUse_clipped",  # Variable cible prédite
        "unit": "kBtu",                     # Unité de mesure de la consommation énergétique
        "api_features": API_FEATURES,       # Liste des colonnes attendues par l'API
        "description": "RandomForest optimisé pour prédire la consommation énergétique des bâtiments non résidentiels à Seattle (8 colonnes)"
    }
)

print("✅ Modèle 8 colonnes sauvegardé avec succès dans BentoML")
