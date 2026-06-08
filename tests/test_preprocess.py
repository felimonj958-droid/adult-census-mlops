import pandas as pd
import numpy as np
import pytest
from src.preprocess import validate_dataframe, split_features_target, build_preprocessor


def sample_df():
    return pd.DataFrame({
        'age': [25, 30, np.nan],
        'workclass': ['Private', np.nan, 'State-gov'],
        'income': ['<=50K', '>50K', '<=50K']
    })


def test_validate_dataframe_accepts_valid_df():
    validate_dataframe(sample_df())


def test_validate_dataframe_rejects_non_df():
    with pytest.raises(TypeError):
        validate_dataframe([1, 2, 3])


def test_validate_dataframe_rejects_empty_df():
    with pytest.raises(ValueError):
        validate_dataframe(pd.DataFrame())


def test_split_features_target_returns_separate_objects():
    df = sample_df()
    X, y = split_features_target(df, 'income')
    assert 'income' not in X.columns
    assert len(y) == len(df)


def test_split_features_target_does_not_modify_original_df():
    df = sample_df()
    original = df.copy(deep=True)
    split_features_target(df, 'income')
    pd.testing.assert_frame_equal(df, original)


def test_split_features_target_missing_target_raises_error():
    with pytest.raises(KeyError):
        split_features_target(sample_df(), 'target')


def test_build_preprocessor_handles_missing_and_encodes():
    df = sample_df()
    X = df.drop(columns=['income'])
    pre = build_preprocessor(['workclass'], ['age'])
    transformed = pre.fit_transform(X)
    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= 2
