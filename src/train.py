import argparse
from pathlib import Path
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import load_adult_dataset
from src.preprocess import split_features_target, build_preprocessor
from src.evaluate import get_classification_metrics


def build_model(model_type: str, params: dict):
    if model_type == 'logistic_regression':
        return LogisticRegression(**params)
    if model_type == 'random_forest':
        return RandomForestClassifier(**params)
    raise ValueError(f'Unsupported model type: {model_type}')


def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    df = load_adult_dataset(config['data']['raw_path'])
    X, y = split_features_target(df, config['data']['target_column'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['project']['random_state'],
        stratify=y,
    )

    preprocessor = build_preprocessor(
        config['data']['categorical_features'],
        config['data']['numeric_features'],
    )
    model = build_model(config['model']['type'], config['model']['params'])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run():
        mlflow.log_param('config_path', config_path)
        mlflow.log_param('model_type', config['model']['type'])
        mlflow.log_param('data_path', config['data']['raw_path'])
        for key, value in config['model']['params'].items():
            mlflow.log_param(key, value)

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
        metrics = get_classification_metrics(y_test, predictions, y_prob)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path='model')

        primary_metric = config['training']['primary_metric']
        threshold = config['training']['minimum_threshold']
        if metrics[primary_metric] < threshold:
            raise SystemExit(f"Primary metric {primary_metric} below threshold: {metrics[primary_metric]:.4f} < {threshold}")

    Path('models').mkdir(exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
