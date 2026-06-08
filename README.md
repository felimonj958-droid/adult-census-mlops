# Adult Census Income MLOps Project

This repository demonstrates a production-style MLOps workflow using the Adult / Census Income dataset. The modeling is intentionally simple so the project can focus on reproducibility, versioned data, experiment tracking, automated testing, CI/CD, and data drift monitoring.

## Objective

Build a tabular classification pipeline that predicts whether income exceeds 50K using a public dataset with mixed feature types and missing values. The deliverable is a grader-friendly repository that can be cloned, restored with DVC, tested with pytest, trained through a config file, compared with MLflow, and monitored for drift with Evidently.

## Dataset

The project uses the Adult dataset from OpenML / UCI. It contains numeric and categorical columns and naturally includes missing values in fields such as workclass, occupation, and native-country.

## Repository structure

```text
adult-census-mlops/
├── .github/workflows/ci.yml
├── configs/
│   ├── train_config.yaml
│   └── experiments/
│       ├── exp_logreg_baseline.yaml
│       ├── exp_logreg_regularized.yaml
│       ├── exp_rf_small.yaml
│       ├── exp_rf_deeper.yaml
│       └── exp_rf_balanced.yaml
├── data/
│   └── raw/
│       ├── adult.csv.dvc
│       └── adult_sample.csv
├── dvc.yaml
├── params.yaml
├── scripts/
│   └── run_experiments.py
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_experiments.py
│   └── monitor_drift.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_data_validation.py
│   └── test_model_validation.py
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd adult-census-mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc pull
pytest tests/ -v
python src/train.py --config configs/train_config.yaml
python scripts/run_experiments.py
python src/compare_experiments.py --config configs/train_config.yaml
python src/monitor_drift.py --config configs/train_config.yaml
```

## DVC workflow

1. Download the dataset with `python src/download_data.py`.
2. Track it with `dvc add data/raw/adult.csv`.
3. Commit `data/raw/adult.csv.dvc` to Git.
4. Configure a remote. This project uses a local remote for demonstration:

```bash
dvc init
dvc remote add -d localremote ./dvc_remote
dvc add data/raw/adult.csv
git add data/raw/adult.csv.dvc .dvc/config .gitignore
git commit -m "Track Adult dataset with DVC"
```

A grader should be able to clone the repository and run `dvc pull` to restore the dataset.

## Experiments

The project includes five experiment configs:
- Logistic regression baseline
- Logistic regression with stronger regularization
- Random forest small
- Random forest deeper
- Random forest with balanced class weights

Each run logs parameters, metrics, the dataset path and model artifact to MLflow.

## Tests

The test suite covers:
- Unit tests for preprocessing behavior
- Data validation tests that load the real Adult dataset or a real sample
- Model validation tests that train on a sample of the real data and verify prediction shape and minimum performance

## Drift monitoring

`src/monitor_drift.py` compares the reference training data with a simulated production slice. It saves an HTML report to `reports/drift_report.html` and exits with code 1 if drift exceeds the configured threshold.

## Grader checklist

- Public GitHub repo with clean structure
- DVC pointer file committed, raw dataset not committed
- Config-driven training
- MLflow integration with multiple runs
- Passing pytest suite
- Green GitHub Actions pipeline
- Evidently drift report and written analysis
