import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics
from sklearn.base import ClassifierMixin


# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    # add description for the first test
    """
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, size=20)

    model = train_model(X, y)

    assert isinstance(model, ClassifierMixin), "train_model() should return a sklearn classifier"
    assert hasattr(model, "predict"), "Returned model must implement predict()"
    assert hasattr(model, "classes_"), "Model appears unfitted; missing classes_ attribute"


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    X = np.random.rand(10, 4)
    y = np.random.randint(0, 2, size=10)

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(y), \
        "inference() should output one prediction per input sample"
    assert preds.ndim == 1, "Predictions must be a 1D array"


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    for name, val in [("precision", precision), ("recall", recall), ("fbeta", fbeta)]:
        assert 0.0 <= val <= 1.0, f"{name} must be between 0 and 1"

