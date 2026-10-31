"""Tests for dataset validation and preprocessing (no fake production metrics)."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import config
from src.data_loader import DatasetNotFoundError, DatasetValidationError, load_dataset, validate_dataset
from src.preprocessing import (
    build_preprocessor,
    columns_to_drop,
    infer_feature_types,
    prepare_feature_frame,
    split_xy,
)


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [30, 40, 50, 35],
            "MonthlyIncome": [4000, 7000, 9000, 5000],
            "Department": ["Sales", "R&D", "Sales", "R&D"],
            "OverTime": ["Yes", "No", "No", "Yes"],
            "EmployeeNumber": [1, 2, 3, 4],
            "EmployeeCount": [1, 1, 1, 1],
            "Gender": ["Female", "Male", "Male", "Female"],
            "Attrition": ["Yes", "No", "No", "Yes"],
        }
    )


def test_validate_requires_attrition():
    df = _toy_frame().drop(columns=["Attrition"])
    with pytest.raises(DatasetValidationError):
        validate_dataset(df)


def test_missing_dataset_raises(tmp_path: Path):
    with pytest.raises(DatasetNotFoundError):
        load_dataset(path=tmp_path / "missing.csv", download_if_missing=False)


def test_split_xy_encodes_target():
    X, y = split_xy(_toy_frame())
    assert "Attrition" not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert int(y.sum()) == 2


def test_columns_to_drop_removes_id_constant_and_gender():
    X, _ = split_xy(_toy_frame())
    dropped = columns_to_drop(X)
    assert "EmployeeNumber" in dropped
    assert "EmployeeCount" in dropped
    assert "Gender" in dropped


def test_preprocessor_fit_transform_and_unknown_category():
    X, y = split_xy(_toy_frame())
    dropped = columns_to_drop(X)
    X = X.drop(columns=dropped)
    numeric, categorical = infer_feature_types(X)
    pre = build_preprocessor(numeric, categorical)
    pipeline = Pipeline([("pre", pre)])
    transformed = pipeline.fit_transform(X)
    assert transformed.shape[0] == len(X)

    unseen = X.iloc[[0]].copy()
    unseen["Department"] = "UnknownDept"
    # handle_unknown='ignore' should not raise.
    pipeline.transform(unseen)


def test_prepare_feature_frame_preserves_order():
    order = ["Age", "Department"]
    frame = pd.DataFrame({"Department": ["Sales"], "Extra": [1]})
    aligned = prepare_feature_frame(frame, order)
    assert list(aligned.columns) == order
    assert aligned["Age"].isna().all()
