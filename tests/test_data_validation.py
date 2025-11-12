# tests/test_data_validation.py
import pandas as pd
import pytest
import os

DATA_PATH = "data/iris.csv"

@pytest.fixture(scope="module")
def data():
    """Load the Iris dataset once for all tests."""
    assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"
    return pd.read_csv(DATA_PATH)

def test_no_missing_values(data):
    """Ensure there are no missing values in the dataset."""
    assert not data.isnull().values.any(), "Dataset contains missing values."

def test_column_names(data):
    """Ensure dataset has the correct columns."""
    expected_columns = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'}
    assert set(data.columns) == expected_columns, f"Unexpected columns: {set(data.columns)}"

def test_species_unique_values(data):
    """Ensure species column has the expected unique classes."""
    expected_classes = {'setosa', 'versicolor', 'virginica'}
    assert set(data['species'].unique()) == expected_classes, "Species classes mismatch."

def test_data_types(data):
    """Ensure numeric columns are float or int."""
    numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(data[col]), f"{col} is not numeric."