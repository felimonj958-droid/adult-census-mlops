import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("Input DataFrame is empty")


def split_features_target(df: pd.DataFrame, target_column: str):
    validate_dataframe(df)
    if target_column not in df.columns:
        raise KeyError(f"Missing target column: {target_column}")
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    return X, y


def build_preprocessor(categorical_features, numeric_features):
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_features),
            ("num", num_pipeline, numeric_features),
        ],
        remainder="drop",
    )