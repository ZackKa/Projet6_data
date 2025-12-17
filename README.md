# Partie 1

## Définition API et test local

### Seattle Energy Prediction - Modèle 8 colonnes
📁 Arborescence du projet

```bash
projet/
│
├── save_model.py          # Script pour sauvegarder le modèle sklearn dans BentoML
├── service.py             # API avec validation Pydantic et endpoint /predict
├── bentofile.yaml         # Recette pour créer l’image Docker
├── energy_model_8cols.joblib  # Modèle pipeline sklearn sauvegardé (8 colonnes)
├── requirements.txt       # Dépendances Python (ou via Poetry)
├── 2016_Building_Energy_Benchmarking.csv  # Csv de base
├── README.md
└── template_modelistation_supervisee-Copy1.ipynb #Analyse donnée et model retenu
```
📝 Étapes détaillées
### 1️⃣ Sauvegarde du modèle sklearn

Entraînement et sélection des 8 colonnes :

```python
cols_selected = [
    "PropertyGFATotal",
    "NumberofFloors",
    "NumberofBuildings",
    "PropertyGFAParking",
    "BuildingAge",
    "FloorsPer1000GFA",
    "IsLargeBuilding",
    "NumUseTypes"
]

X_reduced = df3_encoded[cols_selected]
y_reduced = df3_encoded['SiteEnergyUse_clipped']

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

preprocessor_reduced = ColumnTransformer(
    transformers=[('num', StandardScaler(), cols_selected)]
)

rf_model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)

pipeline_reduced = Pipeline([
    ('preprocess', preprocessor_reduced),
    ('model', rf_model)
])

pipeline_reduced.fit(X_reduced, y_reduced)
```

Sauvegarde du modèle avec joblib :

```python
import joblib

joblib.dump(pipeline_reduced, "energy_model_8cols.joblib")
print("✅ Modèle 8 colonnes sauvegardé avec succès")
```


###2️⃣ Sauvegarde du modèle dans BentoML

Fichier save_model.py :

```python
import joblib
import bentoml

MODEL_PATH = "energy_model_8cols.joblib"
BENTO_MODEL_NAME = "seattle_energy_model_8cols"

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

# Charger le modèle sklearn
model = joblib.load(MODEL_PATH)

# Sauvegarder dans BentoML
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

print("✅ Modèle sauvegardé avec succès dans BentoML")
```

Commande pour exécuter :

```bash
python save_model.py
```


###3️⃣ Création de l’API avec BentoML (service.py)

Chargement du modèle BentoML et définition du service :

```python
import bentoml
from bentoml.io import JSON
import pandas as pd
from pydantic import BaseModel, Field, validator

# Charger le modèle
model_runner = bentoml.sklearn.get("seattle_energy_model_8cols:latest").to_runner()

# Créer le service BentoML
service = bentoml.Service("seattle_energy_service", runners=[model_runner])
```

Définition de la validation Pydantic :

```python
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
```

Endpoint /predict :

```python
@service.api(input=JSON(pydantic_model=BuildingData), output=JSON())
def predict(building: BuildingData):
    input_df = pd.DataFrame([building.dict()])
    prediction = model_runner.run(input_df)
    return {"prediction_kBtu": float(prediction[0])}
```

Lancer le serveur local pour tester Swagger :

```bash
bentoml serve service.py
```

Swagger disponible : http://localhost:3000

Endpoint /predict accepte un JSON avec les 8 colonnes et renvoie prediction_kBtu.

###4️⃣ Création de l’image Docker

Fichier bentofile.yaml :

```yaml
service: "service.py:service"

labels:
  owner: "z"
  project: "seattle-energy-prediction"

include:
  - "*.py"
  - "energy_model_8cols.joblib"

python:
  pip_requirements: requirements.txt
```

Commandes :

Build l’image Docker :

```bash
bentoml build
```

Cela crée une image Docker dans le store BentoML et génère un <TAG_GENERATED>.

Tester avec BentoML (avant Docker) :

```bash
bentoml serve seattle_energy_service:<TAG_GENERATED>
```
Dans mon cas bentoml serve seattle_energy_service:wkg7lfw2qo34mbd4

###5️⃣ Exécution dans Docker

Lancer le container Docker :

```bash
docker run --rm -p 3000:3000 <IMAGE_TAG>
```
Dans mon cas docker run --rm -p 3000:3000 seattle_energy_service:s3xbaqg2q66kmbd4

Tester l’API via Swagger ou Postman : http://localhost:3000/predict.

###6️⃣ Notes

Modèle 8 colonnes : toutes les étapes n’affectent pas le modèle original à 92 colonnes.

Validation Pydantic : assure que les inputs envoyés à l’API sont corrects.

Docker : permet de déployer facilement le modèle en production.

BentoML store : stocke les versions des modèles (latest ou un tag spécifique).


# Partie 2

## 🚀 Déploiement d'un modèle Machine Learning sur Google Cloud Platform (GCP)

Ce guide décrit les étapes pour déployer un modèle Machine Learning sous forme de service REST sur Google Cloud Run, en utilisant Google Cloud Artifact Registry pour stocker l’image Docker.

###1️⃣ Installation et configuration du SDK GCP
(Permet d’installer les outils nécessaires pour interagir avec GCP depuis ton terminal)

Installer le Google Cloud SDK.

Lancer la configuration :

```bash
gcloud init
```

Exemple de sortie et interactions :

```bash
Welcome to the Google Cloud CLI! ...
You must sign in to continue. Would you like to sign in (Y/n)? Y
```

Choisir le projet à utiliser :
```bash
Pick cloud project to use: [1] project-8fcce7ef-47da-4b16-b5b [2] seattle-energy-api-481506 ...
```

Répondre 2 pour sélectionner seattle-energy-api-481506.

Configurer la région pour Compute Engine :

```bash
gcloud config set compute/region europe-west1
```

Activer les API si nécessaire :

```bash
API [compute.googleapis.com] not enabled on project ... Would you like to enable and retry? (y/N) y
```

Vérifier la configuration :

```bash
gcloud config list
```

###2️⃣ Activer les services nécessaires
(Cloud Run permet de déployer le service REST, Artifact Registry permet de stocker l’image Docker que Cloud Run utilisera pour exécuter le modèle)

Activer Cloud Run et Artifact Registry :

```bash
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

Vérifier que les services sont activés :

```bash
gcloud services list --enabled | findstr run
gcloud services list --enabled | findstr artifact
```

###3️⃣ Créer un dépôt Artifact Registry
(C’est ici que l’on stocke l’image Docker. Cloud Run ira la récupérer depuis ce dépôt pour exécuter le service)

Créer le dépôt Docker :

```bash
gcloud artifacts repositories create seattle-energy-repo \
    --repository-format=docker \
    --location=europe-west1 \
    --description="Repository pour Docker images de Seattle Energy API"
```

Vérifier le dépôt :

```bash
gcloud artifacts repositories list --location=europe-west1
```

###4️⃣ Authentification Docker
(Permet à Docker de se connecter à Artifact Registry pour pousser l’image)

Configurer Docker pour pousser les images sur Artifact Registry :

```bash
gcloud auth configure-docker europe-west1-docker.pkg.dev
```

###5️⃣ Construire et pousser l’image Docker
(On crée l’image Docker contenant le modèle et le code pour le servir, puis on la pousse dans Artifact Registry pour que Cloud Run puisse l’utiliser)

Tagger l’image Docker locale :

```bash
docker tag seattle_energy_service:s3xbaqg2q66kmbd4 \
    europe-west1-docker.pkg.dev/seattle-energy-api-481506/seattle-energy-repo/seattle-energy-service:latest
```

Pousser l’image sur Artifact Registry :

```bash
docker push europe-west1-docker.pkg.dev/seattle-energy-api-481506/seattle-energy-repo/seattle-energy-service:latest
```

Vérifier que l’image est bien dans le dépôt :

```bash
gcloud artifacts docker images list \
    europe-west1-docker.pkg.dev/seattle-energy-api-481506/seattle-energy-repo
```

###6️⃣ Déployer l’image sur Cloud Run
(Cloud Run va exécuter le service REST en utilisant l’image Docker stockée dans Artifact Registry. L’accès public permet à n’importe quel client d’envoyer des requêtes HTTP au modèle)

Déployer le service avec accès public :

```bash
gcloud run deploy seattle-energy-service \
    --image europe-west1-docker.pkg.dev/seattle-energy-api-481506/seattle-energy-repo/seattle-energy-service:latest \
    --platform managed \
    --allow-unauthenticated \
    --region europe-west1
```

###7️⃣ Tester le service REST
(On vérifie que le service fonctionne correctement en envoyant des données et en recevant la prédiction du modèle)

Exemple de requête POST pour tester le modèle :

```bash
curl -X POST https://seattle-energy-service-526618594404.europe-west1.run.app/predict \
-H "Content-Type: application/json" \
-d '{
  "PropertyGFATotal": 1000,
  "NumberofFloors": 5,
  "NumberofBuildings": 1,
  "PropertyGFAParking": 100,
  "BuildingAge": 50,
  "FloorsPer1000GFA": 5,
  "IsLargeBuilding": 1,
  "NumUseTypes": 3 
}'
```

✅ Si tout fonctionne, le service retourne la prédiction du modèle.