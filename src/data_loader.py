from pathlib import Path
import pandas as pd


def load_adult_dataset(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
    return df


def write_sample_dataset(input_path: str, output_path: str, n_rows: int = 1000) -> None:
    df = load_adult_dataset(input_path)
    sample = df.sample(n=min(n_rows, len(df)), random_state=42)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
