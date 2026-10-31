"""
Load and validate the IBM HR Analytics attrition dataset.

Purpose:
    - Find the CSV from config.DATASET_PATH (or a caller-supplied path).
    - Optionally download the IBM sample file if it is missing.
    - Check that the Attrition target column exists.
    - Return a pandas DataFrame with no silent fake data.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

import config


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the CSV is missing and download was not used or failed."""


class DatasetValidationError(ValueError):
    """Raised when the file exists but is not a usable attrition dataset."""


def dataset_exists(path: Path | None = None) -> bool:
    """Return True if the configured (or given) CSV file exists."""
    csv_path = Path(path) if path is not None else config.DATASET_PATH
    return csv_path.is_file()


def download_dataset(url: str | None = None, dest: Path | None = None) -> Path:
    """
    Download the IBM sample CSV into Dataset/.

    This is the original IBM fictional HR sample, not real employee records.
    """
    url = url or config.DATASET_URL
    dest = Path(dest) if dest is not None else config.DATASET_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urlopen(url, timeout=60) as response:
            content = response.read()
    except (URLError, TimeoutError, OSError) as exc:
        raise DatasetNotFoundError(
            f"Could not download the dataset from {url}. "
            "Download WA_Fn-UseC_-HR-Employee-Attrition.csv from Kaggle "
            f"({config.DATASET_KAGGLE_URL}) and place it in {config.DATASET_DIR}."
        ) from exc

    if len(content) < 1000:
        raise DatasetValidationError("Downloaded file is too small to be the IBM HR CSV.")

    dest.write_bytes(content)
    return dest


def load_dataset(path: Path | None = None, download_if_missing: bool = False) -> pd.DataFrame:
    """
    Load the attrition CSV.

    Parameters
    ----------
    path:
        Optional custom CSV path. Defaults to config.DATASET_PATH.
    download_if_missing:
        If True, try to download the IBM GitHub sample when the file is absent.
        Training uses this. The Streamlit app does not silently download.
    """
    csv_path = Path(path) if path is not None else config.DATASET_PATH

    if not csv_path.is_file():
        if download_if_missing:
            csv_path = download_dataset(dest=csv_path)
        else:
            raise DatasetNotFoundError(
                f"Dataset not found at {csv_path}. "
                "Run: python -m src.train_model   (this can download the IBM sample) "
                f"or place the CSV in {config.DATASET_DIR}."
            )

    # utf-8-sig strips a BOM if the IBM file includes one.
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(col).strip() for col in df.columns]
    return validate_dataset(df)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Confirm the frame looks like an employee attrition table."""
    if df.empty:
        raise DatasetValidationError("The dataset is empty.")

    if config.TARGET_COLUMN not in df.columns:
        raise DatasetValidationError(
            f"Expected target column '{config.TARGET_COLUMN}' was not found. "
            f"Columns present: {list(df.columns)}"
        )

    target = df[config.TARGET_COLUMN]
    allowed = {
        config.POSITIVE_LABEL,
        config.NEGATIVE_LABEL,
        0,
        1,
        "0",
        "1",
        "yes",
        "no",
        "YES",
        "NO",
        "Yes",
        "No",
        True,
        False,
    }
    unique_values = set(target.dropna().unique())
    # Convert to a comparable form for mixed types.
    unique_as_str = {str(v) for v in unique_values}
    expected_str = {"Yes", "No", "yes", "no", "YES", "NO", "0", "1", "True", "False"}
    if not unique_as_str.issubset(expected_str) and not unique_values.issubset(allowed):
        raise DatasetValidationError(
            f"Unsupported Attrition values: {sorted(unique_as_str)}. "
            "Expected Yes/No (or 0/1)."
        )

    return df


def encode_target(series: pd.Series) -> pd.Series:
    """Map Attrition labels to 1 (left) and 0 (stayed)."""
    mapping = {
        "yes": 1,
        "y": 1,
        "1": 1,
        "true": 1,
        config.POSITIVE_LABEL.lower(): 1,
        "no": 0,
        "n": 0,
        "0": 0,
        "false": 0,
        config.NEGATIVE_LABEL.lower(): 0,
    }

    def _map(value):
        if pd.isna(value):
            raise DatasetValidationError("Attrition contains missing values.")
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
        key = str(value).strip().lower()
        if key not in mapping:
            raise DatasetValidationError(f"Unknown Attrition value: {value}")
        return mapping[key]

    return series.map(_map).astype(int)


def profile_dataset(df: pd.DataFrame) -> dict:
    """Return factual summary statistics used by reports and the UI."""
    target = df[config.TARGET_COLUMN]
    yes_count = int((encode_target(target) == 1).sum())
    no_count = int((encode_target(target) == 0).sum())
    n_rows = int(len(df))

    missing = df.isna().sum()
    missing = missing[missing > 0].to_dict()

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    profile = {
        "n_rows": n_rows,
        "n_columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": dtypes,
        "missing_values_by_column": {k: int(v) for k, v in missing.items()},
        "total_missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_column": config.TARGET_COLUMN,
        "target_classes": sorted(str(v) for v in target.dropna().unique()),
        "class_counts": {
            config.POSITIVE_LABEL: yes_count,
            config.NEGATIVE_LABEL: no_count,
        },
        "attrition_rate": round(yes_count / n_rows, 4) if n_rows else None,
        "source_note": (
            "IBM HR Analytics Employee Attrition & Performance sample. "
            "Fictional records created by IBM; not real employees."
        ),
        "dataset_url": config.DATASET_URL,
        "kaggle_url": config.DATASET_KAGGLE_URL,
    }
    return profile


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
