from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import pytest

MODEL_PATH = Path("models/model.onnx")

def test_onnx_model_file_exists():
    assert MODEL_PATH.exists(), "L'artéfact models/model.onnx n'a pas été généré."

def test_onnx_inference_shape_and_latency():
    """Vérifie le contrat d'entrée/sortie et la latence maximale par requête."""
    session = ort.InferenceSession(str(MODEL_PATH))
    input_name = session.get_inputs()[0].name
    
    # Échantillon de test (batch de 10)
    dummy_input = np.random.randn(10, 3).astype(np.float32)
    
    # Mesure de latence
    start_time = time.perf_counter()
    outputs = session.run(None, {input_name: dummy_input})
    latency = (time.perf_counter() - start_time) / 10.0  # par requête
    
    predictions = outputs[0]
    assert predictions.shape == (10,), f"Shape de sortie attendue (10,), reçu {predictions.shape}"
    assert set(np.unique(predictions)).issubset({0, 1}), "Les prédictions doivent être binaires {0, 1}"
    assert latency < 0.05, f"Latence d'inférence trop élevée : {latency*1000:.2f} ms > 50 ms"