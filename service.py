# import bentoml
# import pandas as pd
# from pydantic import BaseModel, Field
# from typing import List

# # --------------------------------------------------
# # 1. Charger le modèle depuis BentoML
# # --------------------------------------------------

# MODEL_TAG = "seattle_energy_model:latest"
# model_ref = bentoml.sklearn.get(MODEL_TAG) # Récupère le modèle depuis le Model Store BentoML
# #model = model_ref # Charge le modèle
# # V3 du truc/  model = model_ref.to_runner().run  # ou check comment accéder au pipeline
# model = model_ref.to_runner()
# # --------------------------------------------------
# # 2. Schéma Pydantic : validation des entrées
# # --------------------------------------------------
# # Définition du schéma Pydantic pour la validation des entrées et empêche les valeurs incohérentes
# class BuildingInput(BaseModel):
#     PropertyGFATotal: float = Field(..., gt=0, description="Surface totale du bâtiment (GFA)")   # BaseModel : classe de base pour créer des modèles Pydantic
#     NumberofFloors: int = Field(..., gt=0, description="Nombre d'étages")                        # Field : définit les champs du modèle
#     NumberofBuildings: int = Field(..., gt=0, description="Nombre de bâtiments")                 # le : less than or equal to (inférieur ou égal à)
#     PropertyGFAParking: float = Field(..., ge=0, description="Surface de parking")               # ge : greater than or equal to (supérieur ou égal à)
#     BuildingAge: int = Field(..., ge=0, le=1000, description="Âge du bâtiment")                   # gt : greater than (supérieur à)
#     FloorsPer1000GFA: float = Field(..., gt=0, description="Nombre d'étages par 1000 GFA")       # ... : indique que le champ est obligatoire
#     IsLargeBuilding: int = Field(..., ge=0, le=1, description="1 si grand bâtiment, sinon 0")    # description : description du champ
#     NumUseTypes: int = Field(..., ge=1, le=100, description="Nombre de types d'usage")                   

# # --------------------------------------------------
# # 3. Définition du service BentoML
# # --------------------------------------------------
# # bentoml va lancer un serveur web localement pour tester l'API, exposer swagger pour tester l'API, gerer les requettes http
# service = bentoml.Service(
#     name="seattle_energy_api"
# )

# # --------------------------------------------------
# # 4. Endpoint de prédiction
# # --------------------------------------------------

# @service.api(
#     input=bentoml.io.JSON(),
#     output=bentoml.io.JSON()
# )
# # crée automatiquement une route /predict pour la prédiction, une documentation swagger pour tester l'API et validation automatique des entrées
# def predict(input_data: BuildingInput):
#     """
#     Prédit la consommation énergétique (kBtu)
#     à partir des caractéristiques du bâtiment.
#     """
#     # input_data: BuildingInput est la validation des entrées
#     # Conversion de l'entrée validée en DataFrame
#     df = pd.DataFrame([input_data.dict()])

#     # Prédiction
#     prediction = model.predict(df)[0]

#     # Réponse JSON
#     return {
#         "prediction_kbtu": float(prediction),
#         "target": "SiteEnergyUse_clipped",
#         "unit": "kBtu"
#     }



# service.py
# API pour prédire la consommation énergétique des bâtiments de Seattle
# Utilisation de BentoML + Pydantic pour validation des données

import bentoml
from bentoml.io import JSON
import pandas as pd
from pydantic import BaseModel, Field, validator


# -----------------------------
# Charger le modèle
# -----------------------------
model_runner = bentoml.sklearn.get("seattle_energy_model_8cols:latest").to_runner()

# -----------------------------
# Service BentoML
# -----------------------------

service = bentoml.Service("seattle_energy_service", runners=[model_runner])


# -----------------------------
# Pydantic pour validation des inputs
# -----------------------------
class BuildingData(BaseModel):

    PropertyGFATotal: float = Field(..., gt=0, description="Surface totale du bâtiment (GFA)")   # BaseModel : classe de base pour créer des modèles Pydantic
    NumberofFloors: int = Field(..., gt=0, description="Nombre d'étages")                        # Field : définit les champs du modèle
    NumberofBuildings: int = Field(..., gt=0, description="Nombre de bâtiments")                 # le : less than or equal to (inférieur ou égal à)
    PropertyGFAParking: float = Field(..., ge=0, description="Surface de parking")               # ge : greater than or equal to (supérieur ou égal à)
    BuildingAge: int = Field(..., ge=0, le=1000, description="Âge du bâtiment")                   # gt : greater than (supérieur à)
    FloorsPer1000GFA: int = Field(..., gt=0, description="Nombre d'étages par 1000 GFA")       # ... : indique que le champ est obligatoire
    IsLargeBuilding: int = Field(..., ge=0, le=1, description="1 si grand bâtiment, sinon 0")    # description : description du champ
    NumUseTypes: int = Field(..., ge=1, le=100, description="Nombre de types d'usage")
    
    @validator('PropertyGFAParking')
    def parking_must_be_less_than_total(cls, v, values):
        if 'PropertyGFATotal' in values and v > values['PropertyGFATotal']:
            raise ValueError("PropertyGFAParking ne peut pas dépasser PropertyGFATotal")
        return v

# -----------------------------
# Endpoint /predict
# -----------------------------
@service.api(input=JSON(pydantic_model=BuildingData), output=JSON())
def predict(building: BuildingData):
    input_df = pd.DataFrame([building.dict()])
    prediction = model_runner.run(input_df)
    return {"prediction_kBtu": float(prediction[0])}
