from pathlib import Path
import json
from datetime import datetime
from typing import List
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Contrats d'API (Entrée & Sortie)
class PredictionInput(BaseModel):
    feature_1: float = Field(..., ge=-10.0, le=10.0)
    feature_2: float = Field(..., ge=-10.0, le=10.0)
    feature_3: float = Field(..., ge=-10.0, le=10.0)

class PredictionBatchInput(BaseModel):
    instances: List[PredictionInput]

class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]

# 2. Initialisation de l'application FastAPI
app = FastAPI(
    title="MLOps Production Inférence Service",
    description="API de serving pour le modèle de classification exporté en ONNX",
    version="1.0.0"
)

# Chargement unique de la session ONNX au démarrage
MODEL_PATH = Path("models/model.onnx")
if not MODEL_PATH.exists():
    raise RuntimeError(f"Artéfact {MODEL_PATH} introuvable.")

session = ort.InferenceSession(str(MODEL_PATH))
input_name = session.get_inputs()[0].name
LOGS_DIR = Path("data/monitoring_logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "inference_logs.jsonl"


def log_inferences(records: List[dict]):
    """Écrit les entrées et sorties en JSON Lines pour la boucle de monitoring."""
    with open(LOG_FILE, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@app.get("/health")
def healthcheck():
    return {"status": "healthy", "model_loaded": True, "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionBatchInput):
    if not payload.instances:
        raise HTTPException(status_code=400, detail="La liste d'instances ne peut pas être vide.")

    # Transformation vers le tenseur d'entrée
    data = [
        [inst.feature_1, inst.feature_2, inst.feature_3]
        for inst in payload.instances
    ]
    input_tensor = np.array(data, dtype=np.float32)

    # Inférence ONNX
    raw_outputs = session.run(None, {input_name: input_tensor})
    predictions = raw_outputs[0].tolist()

    # Extraction des probabilités de la classe positive (selon la structure skl2onnx)
    probs_dict_list = raw_outputs[1]
    probabilities = [float(p[1]) for p in probs_dict_list]

    # Enregistrement pour l'observabilité (Phase 5)
    now = datetime.utcnow().isoformat()
    log_records = [
        {
            "timestamp": now,
            "feature_1": inst.feature_1,
            "feature_2": inst.feature_2,
            "feature_3": inst.feature_3,
            "prediction": pred,
            "probability": prob
        }
        for inst, pred, prob in zip(payload.instances, predictions, probabilities)
    ]
    log_inferences(log_records)

    return PredictionResponse(predictions=predictions, probabilities=probabilities)