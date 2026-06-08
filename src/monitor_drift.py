import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.data_loader import load_adult_dataset


def make_production_slice(df: pd.DataFrame) -> pd.DataFrame:
    prod = df.sample(frac=0.3, random_state=7).copy()
    if 'hours-per-week' in prod.columns:
        prod['hours-per-week'] = prod['hours-per-week'].clip(lower=1) + 6
    if 'education-num' in prod.columns:
        prod['education-num'] = prod['education-num'].clip(upper=16) - 1
    if 'workclass' in prod.columns:
        prod.loc[prod.sample(frac=0.12, random_state=11).index, 'workclass'] = 'Private'
    if 'occupation' in prod.columns:
        prod.loc[prod.sample(frac=0.10, random_state=13).index, 'occupation'] = None
    return prod


def main(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    reference = load_adult_dataset(config['data']['raw_path'])
    current = make_production_slice(reference)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    report_path = Path(config['monitoring']['report_path'])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_path))

    result = report.as_dict()
    metric_result = result['metrics'][0]['result']
    drifted = metric_result.get('number_of_drifted_columns', 0)
    drift_share = metric_result.get('share_of_drifted_columns', 0)

    print(f'Drifted columns: {drifted}')
    print(f'Drift share: {drift_share:.3f}')

    if drift_share > config['monitoring']['drift_share_threshold']:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
