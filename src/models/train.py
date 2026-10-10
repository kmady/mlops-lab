from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import mlflow
import mlflow.sklearn

def run_training():
    # 1. Configuration MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("mlops-lab-experiment")

    # 2. Chargement des données versionnées
    data_path = Path("data/raw/dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} introuvable. Exécutez 'dvc pull' au préalable.")

    df = pd.read_csv(data_path)
    X = df[["feature_1", "feature_2", "feature_3"]].values.astype(np.float32)
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Paramètres du modèle
    params = {
        "C": 1.0,
        "max_iter": 200,
        "solver": "lbfgs",
        "random_state": 42
    }

    with mlflow.start_run(run_name="logistic_regression_baseline") as run:
        # A. Entraînement
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # B. Évaluation
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba))
        }

        # C. Logging MLflow (Params + Metrics)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # D. Conversion & Export ONNX
        initial_type = [("float_input", FloatTensorType([None, 3]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        onnx_dir = Path("models")
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "model.onnx"
        
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # E. Test de cohérence du runtime ONNX (Sanity Check)
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        onnx_pred = session.run(None, {input_name: X_test[:5]})[0]
        np.testing.assert_array_equal(onnx_pred, model.predict(X_test[:5]))

        # F. Enregistrement des artéfacts et du Model Registry
        mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
        
        # Enregistrement dans le Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn_model",
            registered_model_name="baseline_classifier"
        )

        print(f"✓ Entraînement terminé avec succès (Run ID: {run.info.run_id})")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  F1-Score : {metrics['f1_score']:.4f}")
        print(f"  Modèle ONNX sauvegardé sous : {onnx_path}")

if __name__ == "__main__":
    run_training()