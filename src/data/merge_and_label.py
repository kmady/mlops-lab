import sys
from pathlib import Path

# Garantit que la racine du projet est résolue quel que soit le contexte d'exécution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import pandas as pd
import numpy as np
from src.data.generate_and_validate import validate_dataframe

def simulate_ground_truth(df_logs: pd.DataFrame) -> pd.DataFrame:
    features = ["feature_1", "feature_2", "feature_3"]
    
    # Écrêtage sur les bornes strictes du contrat de données existant
    df_clipped = df_logs[features].clip(lower=-5.0, upper=5.0)
    X = df_clipped.values
    
    noise = np.random.normal(0, 0.05, size=len(X))
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2] + noise > 0).astype(int)
    
    df_labeled = df_clipped.copy()
    df_labeled["target"] = y
    return df_labeled

def merge_datasets():
    ref_path = Path("data/raw/dataset.csv")
    logs_path = Path("data/monitoring_logs/inference_logs.jsonl")

    if not ref_path.exists() or not logs_path.exists():
        raise FileNotFoundError("Jeu de référence ou logs introuvables.")

    ref_df = pd.read_csv(ref_path)

    records = []
    with open(logs_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    
    logs_df = pd.DataFrame(records)
    if logs_df.empty:
        print("Aucun log à fusionner.")
        return

    new_data = simulate_ground_truth(logs_df)
    merged_df = pd.concat([ref_df, new_data], ignore_index=True).drop_duplicates()

    if validate_dataframe(merged_df):
        merged_df.to_csv(ref_path, index=False)
        print(f"✓ Fusion réussie : {len(ref_df)} (ref) + {len(new_data)} (prod) -> {len(merged_df)} lignes.")
        print(f"✓ Fichier {ref_path} mis à jour.")
    else:
        raise ValueError("Le dataset fusionné ne respecte pas le schéma Pydantic.")

if __name__ == "__main__":
    merge_datasets()