import bentoml  # Pour créer et exposer des services ML
from bentoml.io import JSON  # Pour gérer les entrées/sorties au format JSON
import pandas as pd  # Pour manipuler les données sous forme de DataFrame
from pydantic import BaseModel, Field, validator  # Pour la validation des inputs. Pydantic sert à valider les données d’entrée



# Charger le modèle

# Chargement du modèle stocké dans BentoML et .to_runner() crée un "runner" qui permet de faire des prédictions dans le service
# Un runner est un composant BentoML qui encapsule le modèle et permet d’exécuter les prédictions de manière optimisée dans le service API
model_runner = bentoml.sklearn.get("seattle_energy_model_8cols:latest").to_runner()


# Service BentoML

# Création du service BentoML nommé "seattle_energy_service" qui utilise le runner du modèle
service = bentoml.Service("seattle_energy_service", runners=[model_runner])



# Pydantic pour validation des inputs

# Définition du modèle Pydantic pour la validation des inputs. On définit les données attendues par l'API et assure leur validité
class BuildingData(BaseModel):

    PropertyGFATotal: float = Field(..., gt=0, description="Surface totale du bâtiment (GFA)")   # BaseModel : classe de base pour créer des modèles Pydantic
    NumberofFloors: int = Field(..., gt=0, description="Nombre d'étages")                        # Field : définit les champs du modèle
    NumberofBuildings: int = Field(..., gt=0, description="Nombre de bâtiments")                 # le : less than or equal to (inférieur ou égal à)
    PropertyGFAParking: float = Field(..., ge=0, description="Surface de parking")               # ge : greater than or equal to (supérieur ou égal à)
    BuildingAge: int = Field(..., ge=0, le=1000, description="Âge du bâtiment")                   # gt : greater than (supérieur à)
    FloorsPer1000GFA: int = Field(..., gt=0, description="Nombre d'étages par 1000 GFA")       # ... : indique que le champ est obligatoire
    IsLargeBuilding: int = Field(..., ge=0, le=1, description="1 si grand bâtiment, sinon 0")    # description : description du champ
    NumUseTypes: int = Field(..., ge=1, le=100, description="Nombre de types d'usage")
    
    # Validator : vérifie que la surface de parking ne dépasse pas la surface totale
    @validator('PropertyGFAParking')
    def parking_must_be_less_than_total(cls, v, values):
        if 'PropertyGFATotal' in values and v > values['PropertyGFATotal']:
            raise ValueError("PropertyGFAParking ne peut pas dépasser PropertyGFATotal")
        return v
        # cls = la classe Pydantic, dois etre present pour que Pydantic reconnaît le validator
        # v = PropertyGFAParking
        # values = = toutes les autres valeurs déjà validées


# Endpoint /predict

# Définit l'API REST pour faire des prédictions. On utilise le service BentoML créé précédemment et on définit l'input et l'output
# input=JSON(pydantic_model=BuildingData) : attend un JSON correspondant à la classe BuildingData
# output=JSON() : renvoie un JSON avec la prédiction
@service.api(input=JSON(pydantic_model=BuildingData), output=JSON()) # @service.api devient une route API du service BentoML
def predict(building: BuildingData): # building est la donnée reçue depuis l’API, BuildingData est le schéma Pydantic. Le JSON reçu doit respecter les règles définies dans BuildingData
    # Convertit l'objet BuildingData en DataFrame pour que le modèle puisse le traiter
    input_df = pd.DataFrame([building.dict()]) #.dict() le transforme en dictionnaire Python car Pydantic crée un objet Python, mais scikit-learn et pandas attendent des données sous forme de DataFrame.
    # Appelle le modèle pour faire une prédiction
    prediction = model_runner.run(input_df) # run() exécute la prédiction du modèle
    # Renvoie la prédiction sous forme de dictionnaire JSON
    return {"prediction_kBtu": float(prediction[0])} #convertit la prédiction en float pour garantir la compatibilité avec le format JSON.
