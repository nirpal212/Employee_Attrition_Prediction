"""
Load the saved pipeline and score one or more employees.

The Streamlit form and the tests both call these functions. Predictions are
model estimates from historical (here: IBM sample) patterns, not decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

import config
from src.preprocessing import prepare_feature_frame


class ModelNotFoundError(FileNotFoundError):
    """Raised when the joblib pipeline has not been trained yet."""


def load_schema(path: Path | None = None) -> dict:
    schema_path = Path(path) if path is not None else config.SCHEMA_PATH
    if not schema_path.is_file():
        raise ModelNotFoundError(
            f"Feature schema not found at {schema_path}. "
            "Train the model first: python -m src.train_model"
        )
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_pipeline(path: Path | None = None):
    pipeline_path = Path(path) if path is not None else config.PIPELINE_PATH
    if not pipeline_path.is_file():
        raise ModelNotFoundError(
            f"Trained model not found at {pipeline_path}. "
            "Train the model first: python -m src.train_model"
        )
    return joblib.load(pipeline_path)


def predict_employees(inputs: pd.DataFrame, pipeline=None, schema: dict | None = None) -> pd.DataFrame:
    """
    Return a frame with predicted label and P(Attrition=Yes) when available.

    The Attrition target must never be passed in as a feature. If it is present
    in `inputs`, it is dropped before scoring.
    """
    schema = schema or load_schema()
    pipeline = pipeline or load_pipeline()

    work = inputs.copy()
    if config.TARGET_COLUMN in work.columns:
        work = work.drop(columns=[config.TARGET_COLUMN])

    feature_order = schema["feature_order"]
    aligned = prepare_feature_frame(work, feature_order)
    labels = pipeline.predict(aligned)

    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(aligned)
        classes = list(pipeline.classes_)
        if 1 in classes:
            probabilities = proba[:, classes.index(1)]
        else:
            probabilities = proba[:, -1]

    result = pd.DataFrame(
        {
            "prediction_code": labels.astype(int),
            "prediction_label": [
                config.POSITIVE_LABEL if int(v) == 1 else config.NEGATIVE_LABEL for v in labels
            ],
        }
    )
    if probabilities is not None:
        result["probability_attrition_yes"] = probabilities.round(4)
    return result


def predict_one(employee: dict) -> dict:
    """Score a single employee dict from the Streamlit form."""
    frame = pd.DataFrame([employee])
    row = predict_employees(frame).iloc[0].to_dict()
    return row
