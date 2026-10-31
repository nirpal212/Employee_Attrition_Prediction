"""
Build the Scikit-learn preprocessing pipeline.

Purpose:
    - Drop identifier, constant, and excluded sensitive columns.
    - Encode categories with OneHotEncoder(handle_unknown='ignore').
    - Scale numeric columns with StandardScaler (needed for Logistic Regression;
      trees ignore scale, but a shared pipeline keeps comparison fair).
    - Impute missing values inside the pipeline so the imputer is fit on TRAIN only.
    - Never include the Attrition target as an input feature.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from src.data_loader import encode_target


def find_constant_columns(X: pd.DataFrame) -> list[str]:
    """Columns with only one unique value (after dropping NA) add no information."""
    constant = []
    for column in X.columns:
        nunique = X[column].nunique(dropna=True)
        if nunique <= 1:
            constant.append(column)
    return constant


def columns_to_drop(X: pd.DataFrame) -> list[str]:
    """
    Combine domain rules with data-driven constant detection.

    Identifier columns (EmployeeNumber) uniquely tag a person and should not
    be used as predictors. Constant columns (often EmployeeCount, Over18,
    StandardHours in this IBM sample) cannot help a classifier.
    Gender is excluded as a protected characteristic for this academic demo.
    """
    drop = []
    for name in config.IDENTIFIER_COLUMNS + config.EXCLUDED_SENSITIVE_COLUMNS:
        if name in X.columns:
            drop.append(name)
    for name in find_constant_columns(X):
        if name not in drop:
            drop.append(name)
    return drop


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) from the Attrition target (y)."""
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column {config.TARGET_COLUMN}")
    y = encode_target(df[config.TARGET_COLUMN])
    X = df.drop(columns=[config.TARGET_COLUMN])
    return X, y


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split remaining columns into numeric vs categorical."""
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [col for col in X.columns if col not in numeric]
    return numeric, categorical


def outlier_report(X: pd.DataFrame, numeric_columns: Iterable[str]) -> dict:
    """
    IQR-based outlier counts for documentation.

    We do NOT delete outlier rows. Monthly income, years of experience, and
    similar HR fields are naturally skewed. Removing them would hide real
    (in this sample: generated) employment patterns.
    """
    report = {}
    for column in numeric_columns:
        series = X[column].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            count = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((series < lower) | (series > upper)).sum())
        report[column] = {
            "iqr_outlier_count": count,
            "min": float(series.min()),
            "max": float(series.max()),
            "treatment": "retained (plausible HR range / skewed but informative)",
        }
    return report


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    ColumnTransformer applied inside a full model Pipeline.

    Numeric: median impute then StandardScaler.
    Categorical: most-frequent impute then one-hot encode.
    Unknown categories at prediction time become all-zero dummy columns
    (handle_unknown='ignore') instead of crashing the app.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_feature_frame(X: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    """
    Reindex user/training inputs to the exact column names used in training.

    Missing columns are filled with NA so the pipeline imputers can handle them.
    Extra columns are ignored.
    """
    aligned = X.copy()
    for column in feature_order:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned[feature_order]


def build_schema(
    X_train: pd.DataFrame,
    dropped_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict:
    """Save min/max/median and category lists so the Streamlit form can be generated."""
    numeric_meta = {}
    for column in numeric_features:
        series = pd.to_numeric(X_train[column], errors="coerce")
        numeric_meta[column] = {
            "min": _json_number(series.min()),
            "max": _json_number(series.max()),
            "median": _json_number(series.median()),
        }

    categorical_meta = {}
    for column in categorical_features:
        values = sorted(str(v) for v in X_train[column].dropna().unique())
        categorical_meta[column] = values

    return {
        "feature_order": numeric_features + categorical_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "numeric_meta": numeric_meta,
        "categorical_meta": categorical_meta,
        "dropped_columns": dropped_columns,
        "target_column": config.TARGET_COLUMN,
        "positive_label": config.POSITIVE_LABEL,
        "negative_label": config.NEGATIVE_LABEL,
        "notes": {
            "identifier_columns": config.IDENTIFIER_COLUMNS,
            "excluded_sensitive_columns": config.EXCLUDED_SENSITIVE_COLUMNS,
        },
    }


def _json_number(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)
