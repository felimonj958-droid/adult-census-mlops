def test_expected_columns_present(config, adult_df):
    expected = set(config['data']['categorical_features'] + config['data']['numeric_features'] + [config['data']['target_column']])
    assert expected.issubset(set(adult_df.columns))


def test_target_values_are_expected(config, adult_df):
    allowed = set(config['data']['allowed_targets'])
    observed = set(adult_df[config['data']['target_column']].dropna().unique())
    assert observed.issubset(allowed)


def test_numeric_ranges_are_reasonable(config, adult_df):
    for col, bounds in config['data']['numeric_ranges'].items():
        low, high = bounds
        series = adult_df[col].dropna()
        assert series.between(low, high).all(), f'{col} has values outside [{low}, {high}]'
