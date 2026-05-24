# Genre Classification Assignment Udacity nanodegree

A reproducible ML pipeline that classifies music tracks into 15 genres using audio features. Built with scikit-learn, MLflow, Weights & Biases, and Hydra.

## Overview

Given a track's audio attributes (danceability, energy, tempo, etc.) plus a combined text feature derived from the song title and name, a Random Forest classifier predicts one of 15 music genres:

> Dark Trap · Underground Rap · Trap Metal · Emo · Rap · RnB · Pop · Hiphop · techhouse · techno · trance · psytrance · trap · dnb · hardstyle

## Pipeline

The project is structured as a multi-step MLflow pipeline. Each step is an independent component that reads/writes versioned artifacts via Weights & Biases.

```
download → preprocess → check_data → segregate → random_forest → evaluate
```

| Step | Description |
|------|-------------|
| `download` | Streams the raw Parquet dataset from a URL and logs it as a W&B artifact |
| `preprocess` | Drops duplicates, engineers a `text_feature` column (`title + song_name`), and uploads the cleaned CSV |
| `check_data` | Validates column presence/types, value ranges, class names, and runs a Kolmogorov-Smirnov drift test against a reference dataset |
| `segregate` | Stratified train/test split; uploads each split as a separate W&B artifact |
| `random_forest` | Trains a sklearn `Pipeline` (TF-IDF for NLP, StandardScaler for numerics, OrdinalEncoder for categoricals + RandomForestClassifier); logs AUC, feature importance, and confusion matrix to W&B |
| `evaluate` | Loads the exported model artifact and scores it on the held-out test set; logs final AUC and confusion matrix |

## Features Used

| Type | Columns |
|------|---------|
| Numerical | `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms` |
| Categorical | `time_signature`, `key` |
| NLP (TF-IDF) | `text_feature` (`title` + `song_name`) |

## Project Structure

```
genre_classification/
├── main.py               # Hydra entrypoint — orchestrates all pipeline steps
├── config.yaml           # Hydra configuration (data, model hyperparameters, steps)
├── MLproject             # MLflow project definition
├── conda.yml             # Conda environment
├── requirements.txt      # Pip dependencies
├── download/             # Step 1: data download
├── preprocess/           # Step 2: cleaning & feature engineering
├── check_data/           # Step 3: data validation (pytest)
├── segregate/            # Step 4: train/test split
├── random_forest/        # Step 5: model training
└── evaluate/             # Step 6: model evaluation
```

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) or Python 3.13+
- A [Weights & Biases](https://wandb.ai) account — run `wandb login` once before your first run
- [MLflow](https://mlflow.org)

### Install dependencies

**With Conda (recommended):**
```bash
conda env create -f conda.yml
conda activate ex14_sol
```

**With pip:**
```bash
pip install -r requirements.txt
```

## Usage

### Run the full pipeline

```bash
python main.py
```

### Run specific steps

Pass a comma-separated list of step names via the `main.execute_steps` override:

```bash
# Run only download and preprocess
python main.py main.execute_steps=download,preprocess

# Re-run training and evaluation only
python main.py main.execute_steps=random_forest,evaluate
```

### Override hyperparameters

Hydra lets you override any config value on the command line:

```bash
# Change number of trees and max depth
python main.py random_forest_pipeline.random_forest.n_estimators=200 \
               random_forest_pipeline.random_forest.max_depth=10
```

## Configuration

All pipeline parameters live in [config.yaml](config.yaml):

```yaml
main:
  project_name: exercise_14       # W&B project name
  experiment_name: dev            # W&B run group
  execute_steps: [download, preprocess, check_data, segregate, random_forest, evaluate]
  random_seed: 42

data:
  test_size: 0.3                  # Fraction held out for test
  val_size: 0.3                   # Fraction of train used for validation
  ks_alpha: 0.05                  # Significance level for KS drift test
  stratify: genre

random_forest_pipeline:
  random_forest:
    n_estimators: 100
    max_depth: 13
    class_weight: "balanced"
    # ... (see config.yaml for all options)
  tfidf:
    max_features: 10
```

## Experiment Tracking

All runs, metrics, and artifacts are tracked in Weights & Biases under the project `exercise_14`. After running the pipeline, navigate to your W&B dashboard to view:

- **AUC** (macro, one-vs-one) for both validation and test splits
- **Feature importance** bar chart
- **Confusion matrix** (normalized)
- Versioned data and model artifacts

## Tech Stack

- **scikit-learn** — preprocessing pipelines and Random Forest
- **MLflow** — project packaging and model serialization
- **Weights & Biases** — experiment tracking and artifact versioning
- **Hydra** — configuration management and CLI overrides
- **pandas / pyarrow** — data handling
- **pytest** — data validation tests
