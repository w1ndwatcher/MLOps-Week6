# tests/test_model_eval.py
import pytest
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "model/model.joblib"
DATA_PATH = "data/iris.csv"

@pytest.fixture(scope="module")
def model():
    """Load the trained model once."""
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}. Run train.py first."
    return joblib.load(MODEL_PATH)

@pytest.fixture(scope="module")
def test_data():
    """Prepare test data (same as used in training)."""
    assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"
    data = pd.read_csv(DATA_PATH)
    _, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test['species']
    return X_test, y_test

def test_model_predictions_shape(model, test_data):
    """Ensure the number of predictions equals number of samples."""
    X_test, y_test = test_data
    preds = model.predict(X_test)
    assert len(preds) == len(y_test), "Prediction count mismatch with test samples."

def test_model_accuracy(model, test_data):
    """Ensure model accuracy meets minimum threshold."""
    X_test, y_test = test_data
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    assert acc >= 0.90, f"Model accuracy too low: {acc:.3f}"

def test_model_output_classes(model, test_data):
    """Ensure predicted classes match expected species labels."""
    X_test, y_test = test_data
    preds = model.predict(X_test)
    expected_classes = {'setosa', 'versicolor', 'virginica'}
    assert set(preds).issubset(expected_classes), "Unexpected prediction labels."