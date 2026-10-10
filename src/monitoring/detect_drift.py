from pathlib import Path
import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np

from scipy.stats import ks_2samp, wasserstein_distance

def load_recent_logs(logs_path: Path, window_size: int = 300) -> pd.DataFrame:
    """Charge uniquement les N dernières lignes du fichier de logs."""
    if not logs_path.exists():
        return pd.DataFrame()

    records = []
    with open(logs_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    
    df = pd.DataFrame(records)
    if len(df) == 0:
        return df
    
    # Extraction de la fenêtre glissante (les N derniers événements)
    return df.tail(window_size).reset_index(drop=True)

def detect_drift_sliding_window(
    window_size: int = 300,
    min_samples: int = 100,
    alpha_p_value: float = 0.05,
    max_drifted_ratio: float = 0.3
):
    reference_path = Path("data/raw/dataset.csv")
    logs_path = Path("data/monitoring_logs/inference_logs.jsonl")

    if not reference_path.exists():
        print(f"✗ Erreur : Référence {reference_path} introuvable.")
        return

    # 1. Chargement de la fenêtre courante
    current_df = load_recent_logs(logs_path, window_size=window_size)
    total_logs = len(current_df)

    if total_logs < min_samples:
        print(f"ℹ Volume insuffisant pour l'analyse : {total_logs}/{min_samples} logs collectés.")
        print("  Attente d'un volume supérieur pour garantir la puissance statistique.")
        return

    # 2. Chargement du dataset de référence (entraînement)
    ref_df = pd.read_csv(reference_path)
    features = ["feature_1", "feature_2", "feature_3"]

    print(f"\n========================================================")
    print(f"📊 ANALYSE DE DRIFT SUR FENÊTRE GLISSANTE (N = {total_logs})")
    print(f"========================================================")

    drifted_count = 0
    drift_details = {}

    for feat in features:
        ref_values = ref_df[feat].values
        curr_values = current_df[feat].values

        # A. Test de Kolmogorov-Smirnov
        ks_stat, p_value = ks_2samp(ref_values, curr_values)
        
        # B. Distance de Wasserstein (Earth Mover's Distance normalisée par l'écart-type de référence)
        std_ref = np.std(ref_values) if np.std(ref_values) > 1e-6 else 1.0
        w_dist = wasserstein_distance(ref_values, curr_values) / std_ref

        # Condition de dérive : p-value significative ET magnitude de distance tangible
        is_drifted = (p_value < alpha_p_value) and (w_dist > 0.15)
        if is_drifted:
            drifted_count += 1

        drift_details[feat] = {
            "p_value": p_value,
            "wasserstein_norm": w_dist,
            "drift": is_drifted
        }

        flag = "🚨 DRIFT DÉTECTÉ" if is_drifted else "✓ Stable"
        print(f"  [{feat}] p-val: {p_value:.3e} | Dist Wasserstein: {w_dist:.3f} -> {flag}")

    drifted_ratio = drifted_count / len(features)
    print(f"\nProportion de features déviantes : {drifted_ratio:.1%} (Seuil d'alerte : {max_drifted_ratio:.1%})")

    # 3. Prise de décision opérationnelle
    if drifted_ratio >= max_drifted_ratio:
        print("\n🚨 ALERTE MLOPS : Dérive significative confirmée sur la fenêtre glissante !")
        print("-> Déclenchement automatique du pipeline de réentraînement...")
        trigger_retraining_pipeline()
    else:
        print("\n✓ Modèle sous contrôle : distributions compatibles avec l'entraînement.")


def trigger_retraining_pipeline():
    """Exécute la fusion des données, le re-versionnement DVC et le réentraînement MLflow."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    print("\n--- 1. Fusion des données de production étiquetées ---")
    res_merge = subprocess.run([sys.executable, "src/data/merge_and_label.py"], env=env, capture_output=True, text=True)
    if res_merge.returncode != 0:
        print(f"✗ Échec de la fusion : {res_merge.stderr}")
        return
    print(res_merge.stdout.strip())

    print("\n--- 2. Mise à jour de la version des données avec DVC ---")
    subprocess.run(["dvc", "add", "data/raw/dataset.csv"], check=True)
    subprocess.run(["dvc", "push"], check=True)
    print("✓ Nouveau snapshot des données synchronisé avec le remote DVC.")

    print("\n--- 3. Lancement du réentraînement MLflow (train.py) ---")
    res_train = subprocess.run([sys.executable, "src/models/train.py"], env=env, capture_output=True, text=True)
    if res_train.returncode == 0:
        print("✓ Réentraînement terminé avec succès !")
        print(res_train.stdout.strip())
    else:
        print(f"✗ Erreur lors du réentraînement :\n{res_train.stderr}")
        
if __name__ == "__main__":
    detect_drift_sliding_window(window_size=300, min_samples=100)