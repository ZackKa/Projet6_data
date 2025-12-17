import joblib
import bentoml

# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "energy_model_8cols.joblib"
BENTO_MODEL_NAME = "seattle_energy_model_8cols"

# Features exposées à l'API (métier)
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

# -----------------------------
# Chargement du modèle sklearn
# -----------------------------

# Recharge le pipeline exactement comme il était entraîné
model = joblib.load(MODEL_PATH)

# -----------------------------
# Sauvegarde dans BentoML
# -----------------------------

# Enregistre le modèle dans le store BentoML
bentoml.sklearn.save_model(
    name=BENTO_MODEL_NAME,
    model=model,
    metadata={
        "target": "SiteEnergyUse_clipped",
        "unit": "kBtu",
        "api_features": API_FEATURES,
        "description": "RandomForest optimisé pour prédire la consommation énergétique des bâtiments non résidentiels à Seattle (8 colonnes)"
    }
)

print("✅ Modèle 8 colonnes sauvegardé avec succès dans BentoML")
