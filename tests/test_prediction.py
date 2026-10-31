"""Tests for prediction helpers and missing-model handling."""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from src.predict import ModelNotFoundError, load_pipeline, load_schema, predict_employees, predict_one
from src.preprocessing import build_schema, columns_to_drop, infer_feature_types, split_xy


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [30, 40, 50, 35, 28, 45],
            "MonthlyIncome": [4000, 7000, 9000, 5000, 3200, 8100],
            "Department": ["Sales", "R&D", "Sales", "R&D", "Sales", "R&D"],
            "OverTime": ["Yes", "No", "No", "Yes", "Yes", "No"],
            "EmployeeNumber": [1, 2, 3, 4, 5, 6],
            "EmployeeCount": [1, 1, 1, 1, 1, 1],
            "Gender": ["Female", "Male", "Male", "Female", "Male", "Female"],
            "Attrition": ["Yes", "No", "No", "Yes", "Yes", "No"],
        }
    )


def _fit_tiny_pipeline(tmp_path: Path):
    df = _toy_frame()
    X, y = split_xy(df)
    dropped = columns_to_drop(X)
    X = X.drop(columns=dropped)
    numeric, categorical = infer_feature_types(X)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(X, y)
    schema = build_schema(X, dropped, numeric, categorical)
    model_path = tmp_path / "pipeline.joblib"
    schema_path = tmp_path / "schema.json"
    joblib.dump(pipeline, model_path)
    schema_path.write_text(__import__("json").dumps(schema), encoding="utf-8")
    return pipeline, schema, model_path, schema_path


def test_missing_model_error(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "PIPELINE_PATH", tmp_path / "nope.joblib")
    with pytest.raises(ModelNotFoundError):
        load_pipeline()


def test_missing_schema_error(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "SCHEMA_PATH", tmp_path / "nope.json")
    with pytest.raises(ModelNotFoundError):
        load_schema()


def test_prediction_output_format(tmp_path: Path):
    pipeline, schema, _, _ = _fit_tiny_pipeline(tmp_path)
    sample = pd.DataFrame(
        [
            {
                "Age": 30,
                "MonthlyIncome": 4000,
                "Department": "Sales",
                "OverTime": "Yes",
                "Attrition": "Yes",  # must be ignored if present
            }
        ]
    )
    result = predict_employees(sample, pipeline=pipeline, schema=schema)
    assert "prediction_label" in result.columns
    assert result.iloc[0]["prediction_label"] in {"Yes", "No"}
    assert "prediction_code" in result.columns
    assert int(result.iloc[0]["prediction_code"]) in {0, 1}
    assert "probability_attrition_yes" in result.columns
    prob = float(result.iloc[0]["probability_attrition_yes"])
    assert 0.0 <= prob <= 1.0


def test_predict_one_feature_order(tmp_path: Path):
    pipeline, schema, _, _ = _fit_tiny_pipeline(tmp_path)
    employee = {
        "OverTime": "No",
        "Department": "R&D",
        "MonthlyIncome": 8100,
        "Age": 45,
    }
    out = predict_employees(pd.DataFrame([employee]), pipeline=pipeline, schema=schema)
    assert list(out.columns)[0] == "prediction_code"
    assert out.shape[0] == 1
    assert out.iloc[0]["prediction_label"] in {"Yes", "No"}
