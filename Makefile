SHELL := /bin/bash
PYTHON := python
VENV_ACTIVATE := . .venv/bin/activate

.PHONY: help setup storage-init data-gen train test serve-local docker-build docker-up docker-down simulate drift clean

help:
	@echo "======================================================================"
	@echo "                   MLOps Laboratory Automation CLI                   "
	@echo "======================================================================"
	@echo "  make setup          : Installe les dépendances Python dans l'environnement"
	@echo "  make storage-init   : Initialise le remote DVC local et les dossiers de stockage"
	@echo "  make data-gen       : Génère, valide et versionne le dataset brut via DVC"
	@echo "  make train          : Entraîne le modèle, log dans MLflow et exporte en ONNX"
	@echo "  make test           : Exécute la suite de tests et quality gates (pytest)"
	@echo "  make serve-local    : Lance l'API FastAPI localement avec rechargement à chaud"
	@echo "  make docker-build   : Construit l'image Docker du service d'inférence"
	@echo "  make docker-up      : Démarre le conteneur Docker avec volume de logs monté"
	@echo "  make docker-down    : Arrête et supprime le conteneur d'inférence"
	@echo "  make simulate       : Simule du trafic utilisateur (normal + dérivé)"
	@echo "  make drift          : Lance l'analyse de drift, fusionne les données et ré-entraîne"
	@echo "  make clean          : Nettoie les caches, logs et artéfacts temporaires"
	@echo "======================================================================"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install skl2onnx onnxruntime pytest fastapi uvicorn requests scipy

storage-init:
	mkdir -p /tmp/mlops_storage/dvc-storage
	mkdir -p /tmp/mlops_storage/mlflow-artifacts
	dvc init -f
	dvc remote add -d local-storage /tmp/mlops_storage/dvc-storage || true

data-gen:
	$(PYTHON) src/data/generate_and_validate.py
	dvc add data/raw/dataset.csv
	dvc push

train:
	$(PYTHON) src/models/train.py

test:
	$(PYTHON) -m pytest -v tests/

serve-local:
	$(PYTHON) -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t mlops-serving:v1 -f docker/Dockerfile.serving .

docker-up:
	@docker rm -f serving_instance 2>/dev/null || true
	mkdir -p data/monitoring_logs
	docker run -d --name serving_instance \
		-p 8000:8000 \
		-v "$$(pwd)/data/monitoring_logs:/app/data/monitoring_logs" \
		mlops-serving:v1
	@echo "✓ Conteneur serving_instance démarré sur http://localhost:8000"

docker-down:
	docker rm -f serving_instance 2>/dev/null || true
	@echo "✓ Conteneur serving_instance arrêté."

simulate:
	$(PYTHON) src/monitoring/simulate_traffic.py

drift:
	$(PYTHON) src/monitoring/detect_drift.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf models/*.onnx
	rm -rf data/monitoring_logs/inference_logs.jsonl
	@echo "✓ Nettoyage effectué."