from pathlib import Path
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# 1. Contrat de données / Schéma strict
class DataRecord(BaseModel):
    feature_1: float = Field(ge=-5.0, le=5.0)
    feature_2: float = Field(ge=-5.0, le=5.0)
    feature_3: float = Field(ge=-5.0, le=5.0)
    target: int = Field(ge=0, le=1)

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Valide chaque ligne du DataFrame selon le schéma Pydantic."""
    for idx, row in df.iterrows():
        try:
            DataRecord(**row.to_dict())
        except ValidationError as e:
            print(f"Erreur de validation à l'index {idx}: {e}")
            return False
    return True

def generate_sample_data(output_path: Path):
    """Génère un jeu de données synthétique reproductible et le valide."""
    np.random.seed(42)
    n_samples = 500
    
    # Données synthétiques
    X = np.random.randn(n_samples, 3)
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2] > 0).astype(int)
    
    df = pd.DataFrame(X, columns=["feature_1", "feature_2", "feature_3"])
    df["target"] = y
    
    # Validation avant écriture
    if validate_dataframe(df):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Données validées et sauvegardées sous : {output_path}")
    else:
        raise ValueError("Les données ne respectent pas le contrat de schéma.")

if __name__ == "__main__":
    raw_data_file = Path("data/raw/dataset.csv")
    generate_sample_data(raw_data_file)