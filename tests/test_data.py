from pathlib import Path
import pandas as pd
import pytest
from src.data.generate_and_validate import validate_dataframe

def test_dataset_exists_and_valid():
    dataset_path = Path("data/raw/dataset.csv")
    assert dataset_path.exists(), "Le dataset versionné data/raw/dataset.csv est absent."
    
    df = pd.read_csv(dataset_path)
    assert not df.empty, "Le dataset est vide."
    assert set(["feature_1", "feature_2", "feature_3", "target"]).issubset(df.columns)
    assert validate_dataframe(df), "Certaines lignes violent le contrat de schéma Pydantic."