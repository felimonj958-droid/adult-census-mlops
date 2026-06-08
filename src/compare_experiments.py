import argparse
import yaml
import mlflow


def main(config_path='configs/train_config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
    if experiment is None:
        raise ValueError('Experiment not found. Run training first.')
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    primary_metric = f"metrics.{config['training']['primary_metric']}"
    best = runs.sort_values(primary_metric, ascending=False).head(1)
    print(best[['run_id', primary_metric, 'params.model_type', 'params.config_path']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_config.yaml')
    args = parser.parse_args()
    main(args.config)
