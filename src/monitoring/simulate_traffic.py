import time
import requests
import numpy as np

API_URL = "http://localhost:8000/predict"

def send_batch(data: np.ndarray, label: str):
    payload = {
        "instances": [
            {"feature_1": float(r[0]), "feature_2": float(r[1]), "feature_3": float(r[2])}
            for r in data
        ]
    }
    resp = requests.post(API_URL, json=payload)
    if resp.status_code == 200:
        print(f"✓ Vague '{label}' : {len(data)} requêtes envoyées.")
    else:
        print(f"✗ Erreur : {resp.status_code} - {resp.text}")

def run_simulation():
    np.random.seed(101)
    
    # 1. Trafic normal (distribution proche de la référence)
    normal_data = np.random.randn(150, 3)
    send_batch(normal_data, "Distribution Normale")

    time.sleep(1)

    # 2. Trafic avec Data Drift sévère (décalage de moyenne et variance)
    drifted_data = np.random.normal(loc=2.5, scale=1.8, size=(150, 3))
    drifted_data[:, 1] = drifted_data[:, 1] * 0.1  # Réduction artificielle de variance
    send_batch(drifted_data, "Distribution Driftée (Anomalie)")

if __name__ == "__main__":
    run_simulation()