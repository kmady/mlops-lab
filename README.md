# 🚀 Production MLOps Laboratory

Un laboratoire MLOps pédagogique et modulaire conçu pour servir de blueprint d'ingénierie pour les projets Machine Learning en production. Il couvre l'intégralité du cycle de vie : du versionnement des données brutes à la surveillance de la dérive statistique (*Data Drift*) avec boucle de réentraînement fermée (*Continuous Training*).

---

## 🏗️ Architecture Système (5 Couches)

Le système est découpé selon le standard d'ingénierie **5-Layer MLOps Architecture** :

```

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DataOps (Données & Contrats)                                             │
│    • DVC (Stockage distant découplé)                                        │
│    • Pydantic (Validation du contrat de données à l'ingestion)              │
└──────────────────────────────────────┬──────────────────────────────────────┘
│
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ 2. Expérimentation & Standardisation                                        │
│    • MLflow Tracking (SQLite + Artéfacts locaux)                             │
│    • ONNX Runtime (Découplage framework-runtime pour l'inférence)           │
│    • MLflow Model Registry (Gestion des versions v1, v2...)                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
│
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ 3. CI/CD & Contrôle Qualité (Quality Gates)                                 │
│    • Pytest (Validation des schémas, tests unitaires)                       │
│    • Contrôle de non-régression & latence (< 50 ms / requête)               │
└──────────────────────────────────────┬──────────────────────────────────────┘
│
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ 4. Serving & Production Runtime                                             │
│    • FastAPI (API REST asynchrone haute performance)                        │
│    • Inférence native ONNX (Zéro dépendance d'entraînement en prod)         │
│    • Docker (Conteneurisation minimale avec montage de volume pour logs)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
│
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ 5. Observabilité, Drift & Boucle Fermée                                     │
│    • Sliding Window Monitoring (Analyse sur les N dernières requêtes)       │
│    • Tests statistiques : Kolmogorov-Smirnov (p-val) + Distance Wasserstein │
│    • Feedback Loop : Fusion, ré-étiquetage, update DVC et réentraînement    │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 📊 Fondement Mathématique du Monitoring de Drift

Contrairement à un bug logiciel classique levant une exception `HTTP 500`, un modèle en production subit souvent une **défaillance silencieuse** (*Silent Failure*) : la distribution des données d'entrée change alors que le modèle continue de renvoyer des prédictions.

Le module de surveillance (`src/monitoring/detect_drift.py`) combine deux approches complémentaires sur une **fenêtre glissante** :

1. **Test d'adéquation non-paramétrique de Kolmogorov-Smirnov ($p$-value)** :
   $$D = \sup_{x} \vert{}F_{\text{ref}}(x) - F_{\text{prod}}(x)\vert{}$$
   - $H_0$ : La distribution observée en production est identique à celle de référence.
   - En MLOps, une $p$-value très faible ($p < 0.05$) signale que l'écart n'est pas dû au hasard : **rejet de $H_0$, drift confirmé**.

2. **Distance de Wasserstein (Earth Mover's Distance)** :
   Quantifie l'effort physique nécessaire pour déplacer la distribution de référence vers la distribution de production. La distance est normalisée par l'écart-type de référence ($\sigma_{\text{ref}}$) pour évaluer la magnitude concrète de la dérive indépendamment de la taille de l'échantillon.

Si plus de **30 % des variables** présentent une dérive statistique avérée, le système déclenche automatiquement :
1. L'inférence et la labellisation des nouvelles données.
2. La fusion avec l'historique et la validation par schéma strict Pydantic.
3. La mise à jour du pointeur DVC (`dvc add` + `dvc push`).
4. Le réentraînement d'un nouveau modèle enregistré dans le **Model Registry** de MLflow.

---

## 📁 Structure du Répertoire

```text
mlops-lab/
├── .github/
│   └── workflows/
│       └── ci.yml               # Pipeline CI GitHub Actions (Tests & Quality Gates)
├── data/
│   ├── raw/
│   │   ├── dataset.csv.dvc     # Pointeur DVC vers les données versionnées
│   │   └── .gitignore          # Ignore dataset.csv du commit Git
│   └── monitoring_logs/        # Logs JSONL d'inférence (montés via volume Docker)
├── docker/
│   ├── Dockerfile.serving      # Conteneur d'inférence minimal FastAPI + ONNX
│   └── docker-compose.yml      # Orchestration optionnelle
├── models/
│   └── model.onnx              # Artéfact standardisé exporté
├── src/
│   ├── data/
│   │   ├── generate_and_validate.py # Génération de données synthétiques + validation
│   │   └── merge_and_label.py       # Simulation ground-truth et fusion post-drift
│   ├── models/
│   │   └── train.py                 # Entraînement, log MLflow et export ONNX
│   ├── serving/
│   │   └── app.py                   # Serveur FastAPI d'inférence + logging JSONL
│   └── monitoring/
│       ├── simulate_traffic.py      # Simulation de trafic (normal vs drifté)
│       └── detect_drift.py          # Analyse statistique sur fenêtre glissante
├── tests/
│   ├── test_data.py            # Validation du contrat de données
│   └── test_model.py           # Intégrité du modèle ONNX et test de latence
├── Makefile                    # Interface CLI d'orchestration unifiée
├── pyproject.toml              # Configuration du projet et de pytest
├── requirements.txt            # Dépendances Python
└── README.md

```

---

## 🛠️ Installation et Prérequis

### Prérequis

* Python 3.11+
* Git
* Docker

### Installation rapide

```bash
# 1. Cloner le dépôt
git clone [https://github.com/](https://github.com/)kmady/mlops-lab.git
cd mlops-lab

# 2. Configurer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate

# 3. Installer les dépendances
make setup

```

---

## ⚡ Guide d'Exécution Pas-à-Pas (Via Makefile)

L'ensemble des opérations du laboratoire est unifié dans le `Makefile` :

### 1. Initialiser le stockage & Générer les données

```bash
# Création des répertoires distants locaux et initialisation DVC
make storage-init

# Génération des données, validation Pydantic et push DVC
make data-gen

```

### 2. Démarrer MLflow et entraîner le modèle initial

Lancez le serveur MLflow dans un terminal séparé :

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root /tmp/mlops_storage/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000

```

Dans votre terminal principal :

```bash
# Entraînement, métriques tracées et export models/model.onnx
make train

```

> Accédez à l'interface MLflow sur `http://localhost:5000` pour visualiser l'expérience et le Model Registry.

### 3. Exécuter les Quality Gates

```bash
make test

```

Valide la conformité du dataset, la validité des sorties ONNX et vérifie que la latence d'inférence reste sous le seuil critique (< 50 ms).

### 4. Déployer le service d'inférence conteneurisé

```bash
# Construction de l'image Docker
make docker-build

# Démarrage du conteneur avec volume partagé pour les logs
make docker-up

```

Vérifiez l'état de l'API :

```bash
curl http://localhost:8000/health

```

### 5. Simuler le trafic et déclencher la boucle fermée de réentraînement

```bash
# Envoie 150 requêtes normales suivies de 150 requêtes fortement driftées
make simulate

# Analyse la fenêtre glissante, détecte la dérive, fusionne et ré-entraîne
make drift

```

Dans l'interface MLflow, vous verrez immédiatement une **version 2** du modèle apparaître dans le `baseline_classifier` avec les métriques recalculées sur la distribution élargie.

### 6. Nettoyage

```bash
# Arrêter le conteneur d'inférence
make docker-down

# Nettoyer les caches et logs temporaires
make clean

```

---

## 📋 Référence des Commandes `Makefile`

| Commande | Action |
| --- | --- |
| `make setup` | Installe l'ensemble des bibliothèques nécessaires |
| `make storage-init` | Crée le remote DVC local externe et initialise DVC |
| `make data-gen` | Crée, valide et pousse `dataset.csv` sur DVC |
| `make train` | Lance l'entraînement tracé et compile en ONNX |
| `make test` | Lance la suite de tests unitaires et de latence `pytest` |
| `make serve-local` | Démarre l'API FastAPI en local avec rechargement à chaud |
| `make docker-build` | Construit l'image Docker `mlops-serving:v1` |
| `make docker-up` | Démarre le conteneur d'inférence avec volume de logs monté |
| `make docker-down` | Arrête et supprime le conteneur de serving |
| `make simulate` | Simule un trafic de production avec anomalie statistique |
| `make drift` | Calcule les scores KS/Wasserstein et déclenche le réentraînement |
| `make clean` | Nettoie les caches Python, tests et logs d'inférence |

---

## 📄 Licence

Ce projet est sous licence MIT - libre pour usage pédagogique et prototypage de production.

```

---

### Commandes pour enregistrer et pousser ce README

Ouvrez ou créez le fichier `README.md` à la racine de votre projet avec ce contenu, puis exécutez :

```bash
git add README.md
git commit -m "docs: ajout de la documentation technique complète et du guide d'exécution"
git push origin main

```