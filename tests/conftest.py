from pathlib import Path
import pytest
import yaml
from src.data_loader import load_adult_dataset, write_sample_dataset


@pytest.fixture(scope='session')
def config():
    with open('configs/train_config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='session')
def adult_df(config):
    path = Path(config['data']['raw_path'])
    if not path.exists():
        raise FileNotFoundError('Expected data/raw/adult.csv. Run dvc pull or python src/download_data.py first.')
    return load_adult_dataset(str(path))


@pytest.fixture(scope='session')
def adult_sample_df(config, adult_df):
    sample_path = Path(config['data']['sample_path'])
    if not sample_path.exists():
        write_sample_dataset(config['data']['raw_path'], config['data']['sample_path'], n_rows=1500)
    return load_adult_dataset(str(sample_path))
