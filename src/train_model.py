"""
Train, compare, and save the attrition classification pipeline.

Run from the project root:
    python -m src.train_model

What this script does:
    1. Load (or download) the IBM HR sample CSV.
    2. Profile the real table (rows, missing values, class balance).
    3. Drop duplicates if any exist.
    4. Stratified train/test split (test set is NOT used for tuning).
    5. Drop identifier / constant / sensitive columns using the TRAIN set.
    6. Fit three models with class_weight='balanced' because attrition is rare.
    7. Tune each model with StratifiedKFold on TRAIN only.
    8. Choose the winner by cross-validated F1 (not by test accuracy).
    9. Evaluate every model on the held-out test set for the comparison table.
   10. Save the winning Pipeline with joblib.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python src/train_model.py` as well as `python -m src.train_model`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import config
from src.data_loader import load_dataset, profile_dataset, save_json
from src.evaluate_model import compute_metrics, confusion_matrix_values
from src.preprocessing import (
    build_preprocessor,
    build_schema,
    columns_to_drop,
    infer_feature_types,
    outlier_report,
    split_xy,
)


def _candidate_models() -> dict:
    """
    Three beginner-standard classifiers.

    class_weight='balanced' up-weights the minority (Attrition=Yes) class
    using training-fold frequencies. It does not copy or invent rows, so it
    does not leak test information.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_estimators=150,
        ),
    }


def _param_grids() -> dict:
    """Small grids so training stays practical on a laptop."""
    return {
        "Logistic Regression": {
            "clf__C": [0.1, 1.0, 10.0],
        },
        "Decision Tree": {
            "clf__max_depth": [3, 5, 8, None],
            "clf__min_samples_split": [2, 10],
        },
        "Random Forest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_leaf": [1, 4],
        },
    }


def _positive_proba(model, X) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]


def _feature_importance_table(pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    clf = pipeline.named_steps["clf"]
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        return pd.DataFrame()

    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        kind = "impurity_importance"
    elif hasattr(clf, "coef_"):
        values = np.abs(np.ravel(clf.coef_))
        kind = "absolute_coefficient"
    else:
        return pd.DataFrame()

    table = pd.DataFrame(
        {
            "feature": names,
            "importance": values,
            "kind": kind,
            "model": model_name,
        }
    )
    return table.sort_values("importance", ascending=False)


def compute_eda_summary(df: pd.DataFrame) -> dict:
    """Facts for the dashboard and the academic report (computed, not guessed)."""
    from src.data_loader import encode_target

    work = df.copy()
    work["AttritionFlag"] = encode_target(work[config.TARGET_COLUMN])
    summary: dict = {"n_rows": int(len(work))}

    def rate_by(column: str) -> dict | None:
        if column not in work.columns:
            return None
        grouped = (
            work.groupby(column, observed=True)["AttritionFlag"]
            .agg(["mean", "count", "sum"])
            .rename(columns={"mean": "attrition_rate", "sum": "attrition_count"})
        )
        grouped["attrition_rate"] = grouped["attrition_rate"].round(4)
        return grouped.reset_index().to_dict(orient="records")

    for column in ["OverTime", "JobRole", "Department", "JobSatisfaction"]:
        summary[f"by_{column}"] = rate_by(column)

    if "YearsAtCompany" in work.columns:
        bins = [-0.1, 2, 5, 10, 20, 50]
        labels = ["0-2", "3-5", "6-10", "11-20", "21+"]
        work["YearsAtCompanyBand"] = pd.cut(
            work["YearsAtCompany"], bins=bins, labels=labels
        )
        summary["by_YearsAtCompanyBand"] = rate_by("YearsAtCompanyBand")

    return summary


def train_and_save(dataset_path: Path | None = None, download_if_missing: bool = True) -> dict:
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset(path=dataset_path, download_if_missing=download_if_missing)
    profile = profile_dataset(df)
    save_json(profile, config.DATASET_PROFILE_PATH)

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        df = df.drop_duplicates().reset_index(drop=True)

    eda_summary = compute_eda_summary(df)
    save_json(eda_summary, config.EDA_SUMMARY_PATH)

    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    dropped = columns_to_drop(X_train)
    X_train = X_train.drop(columns=dropped, errors="ignore")
    X_test = X_test.drop(columns=dropped, errors="ignore")

    numeric_features, categorical_features = infer_feature_types(X_train)
    outliers = outlier_report(X_train, numeric_features)

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    comparison_rows = []
    fitted_models = {}
    cv_scores = {}

    for name, estimator in _candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", estimator),
            ]
        )
        search = GridSearchCV(
            pipeline,
            param_grid=_param_grids()[name],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        fitted_models[name] = best_model
        cv_scores[name] = float(search.best_score_)

        y_pred = best_model.predict(X_test)
        y_scores = _positive_proba(best_model, X_test)
        y_proba = None
        if y_scores is not None:
            y_proba = np.column_stack([1 - y_scores, y_scores])
        test_metrics = compute_metrics(y_test, y_pred, y_proba)
        cm = confusion_matrix_values(y_test, y_pred)

        comparison_rows.append(
            {
                "model": name,
                "cv_f1": round(cv_scores[name], 4),
                "test_accuracy": round(test_metrics["accuracy"], 4),
                "test_precision": round(test_metrics["precision"], 4),
                "test_recall": round(test_metrics["recall"], 4),
                "test_f1": round(test_metrics["f1"], 4),
                "test_roc_auc": (
                    round(test_metrics["roc_auc"], 4)
                    if test_metrics["roc_auc"] is not None
                    else None
                ),
                "best_params": json.dumps(search.best_params_),
                "true_negative": cm["true_negative"],
                "false_positive": cm["false_positive"],
                "false_negative": cm["false_negative"],
                "true_positive": cm["true_positive"],
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(config.MODEL_COMPARISON_PATH, index=False)

    # Select by cross-validated F1 so the test set stays out of model choice.
    selected_name = max(cv_scores, key=cv_scores.get)
    selected_model = fitted_models[selected_name]

    schema = build_schema(X_train, dropped, numeric_features, categorical_features)
    schema["n_train"] = int(len(X_train))
    schema["n_test"] = int(len(X_test))
    schema["train_positive"] = int(y_train.sum())
    schema["test_positive"] = int(y_test.sum())
    config.SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    joblib.dump(selected_model, config.PIPELINE_PATH)

    importance = _feature_importance_table(selected_model, selected_name)
    if not importance.empty:
        importance.to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)

    selected_row = comparison.loc[comparison["model"] == selected_name].iloc[0]
    summary = {
        "selected_model": selected_name,
        "selection_rule": (
            "Highest StratifiedKFold F1 on the training set after GridSearchCV. "
            "The held-out test set was not used to pick the winner."
        ),
        "cv_f1_by_model": {k: round(v, 4) for k, v in cv_scores.items()},
        "selected_test_metrics": {
            "accuracy": float(selected_row["test_accuracy"]),
            "precision": float(selected_row["test_precision"]),
            "recall": float(selected_row["test_recall"]),
            "f1": float(selected_row["test_f1"]),
            "roc_auc": selected_row["test_roc_auc"],
        },
        "dropped_columns": dropped,
        "dropped_column_reasons": {
            "identifiers": config.IDENTIFIER_COLUMNS,
            "sensitive_excluded": config.EXCLUDED_SENSITIVE_COLUMNS,
            "constants_detected_on_train": [
                col
                for col in dropped
                if col not in config.IDENTIFIER_COLUMNS + config.EXCLUDED_SENSITIVE_COLUMNS
            ],
        },
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_size": config.TEST_SIZE,
        "random_state": config.RANDOM_STATE,
        "cv_folds": config.CV_FOLDS,
        "duplicate_rows_removed": duplicate_count,
        "missing_cells_before_impute": profile["total_missing_cells"],
        "outlier_investigation": outliers,
        "class_weight": "balanced",
        "imbalance_note": (
            "Attrition is the minority class. Models use class_weight='balanced'. "
            "Recall is discussed with precision and F1 because missing a likely "
            "departure can matter more than a slightly lower accuracy."
        ),
        "dataset_profile": profile,
        "pipeline_path": str(config.PIPELINE_PATH),
    }
    save_json(summary, config.TRAINING_SUMMARY_PATH)
    return summary


def main() -> None:
    print("Training Employee Attrition models on the IBM HR sample dataset...")
    summary = train_and_save()
    print(f"Rows: {summary['dataset_profile']['n_rows']}")
    print(f"Columns: {summary['dataset_profile']['n_columns']}")
    print(f"Selected model: {summary['selected_model']}")
    print(f"CV F1 by model: {summary['cv_f1_by_model']}")
    print(f"Test metrics: {summary['selected_test_metrics']}")
    print(f"Saved pipeline: {summary['pipeline_path']}")
    print(f"Comparison table: {config.MODEL_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
