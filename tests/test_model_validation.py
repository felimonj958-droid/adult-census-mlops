from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.preprocess import build_preprocessor


def test_model_prediction_shape_and_type(config, adult_sample_df):
    X = adult_sample_df.drop(columns=[config['data']['target_column']])
    y = adult_sample_df[config['data']['target_column']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ('preprocessor', build_preprocessor(config['data']['categorical_features'], config['data']['numeric_features'])),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    assert len(preds) == len(X_test)
    assert set(preds).issubset(set(config['data']['allowed_targets']))


def test_model_reaches_minimum_training_performance(config, adult_sample_df):
    X = adult_sample_df.drop(columns=[config['data']['target_column']])
    y = adult_sample_df[config['data']['target_column']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ('preprocessor', build_preprocessor(config['data']['categorical_features'], config['data']['numeric_features'])),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    assert score >= 0.70
