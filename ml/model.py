import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from ml.data import process_data
from typing import Tuple, Any
import numpy as np
import pandas as pd




def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:

    model = LogisticRegression(
        solver="liblinear",
        random_state=42,
        max_iter=1000,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: Any, X: np.ndarray) -> np.ndarray:

    return model.predict(X)


def save_model(obj: Any, path: str) -> None:

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_model(path: str) -> Any:

    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
    data: pd.DataFrame,
    column_name: str,
    slice_value,
    categorical_features: list,
    label: str,
    encoder,
    lb,
    model: Any,
) -> Tuple[float, float, float]:

    slice_df = data[data[column_name] == slice_value]
    if slice_df.empty:
        return 0.0, 0.0, 0.0

    X_slice, y_slice, _, _ = process_data(
        slice_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Predict and score
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta

