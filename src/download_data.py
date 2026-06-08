from pathlib import Path
from sklearn.datasets import fetch_openml


def main():
    output = Path('data/raw')
    output.mkdir(parents=True, exist_ok=True)
    X, y = fetch_openml('adult', version=2, as_frame=True, return_X_y=True)
    df = X.copy()
    df['income'] = y.astype(str)
    df.to_csv(output / 'adult.csv', index=False)


if __name__ == '__main__':
    main()
