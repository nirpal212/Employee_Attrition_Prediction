"""
Streamlit frontend for Employee Attrition Prediction & HR Analytics.

Launch from the project root:
    streamlit run app.py

Pages:
    Home, HR Analytics, Prediction, Model Performance, About.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from src.data_loader import DatasetNotFoundError, load_dataset, profile_dataset
from src.predict import ModelNotFoundError, load_pipeline, load_schema, predict_one
from src.visualizations import (
    plot_attrition_rate_by,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_years_at_company,
)

st.set_page_config(
    page_title=config.PROJECT_TITLE,
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp { background-color: #f4f7fb; }
    section[data-testid="stSidebar"] { background-color: #12263a; }
    section[data-testid="stSidebar"] * { color: #e8eef5; }
    .hero {
        background: #12263a;
        color: #f4f7fb;
        padding: 1.4rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .hero h1 { color: #ffffff; font-size: 1.7rem; margin-bottom: 0.35rem; }
    .hero p { color: #c9d6e5; margin: 0; }
    .metric-card {
        background: #ffffff;
        border: 1px solid #d7e0ea;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(18, 38, 58, 0.06);
    }
    .note {
        background: #eaf4f2;
        border-left: 4px solid #2a9d8f;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        color: #1b2430;
    }
    .warn {
        background: #fff4e8;
        border-left: 4px solid #c47b2a;
        padding: 0.8rem 1rem;
        border-radius: 6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _load_json(path: Path) -> dict | None:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


@st.cache_data(show_spinner=False)
def cached_dataset():
    return load_dataset(download_if_missing=False)


@st.cache_resource(show_spinner=False)
def cached_pipeline():
    return load_pipeline()


@st.cache_data(show_spinner=False)
def cached_schema():
    return load_schema()


def sidebar() -> str:
    st.sidebar.markdown("### HR Analytics")
    st.sidebar.caption(f"{config.STUDENT_NAME} · {config.ROLL_NUMBER}")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Home",
            "HR Analytics",
            "Predict Attrition",
            "Model Performance",
            "About",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Academic learning tool. Not for hiring or firing decisions.")
    return page


def setup_message(kind: str) -> None:
    if kind == "dataset":
        st.markdown(
            f"""
            <div class="warn">
            <b>Dataset not found.</b> Place
            <code>{config.DATASET_FILENAME}</code> in
            <code>{config.DATASET_DIR}</code>, or train once so the app can
            download the IBM sample:<br>
            <code>python -m src.train_model</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="warn">
            <b>Trained model not found.</b> Predictions are disabled until you train:
            <br><code>python -m src.train_model</code>
            </div>
            """,
            unsafe_allow_html=True,
        )



def page_home() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Employee Attrition Dashboard</h1>
            <p>HR Analytics | Employee Retention Insights</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        df = cached_dataset()
    except DatasetNotFoundError:
        setup_message("dataset")
        return

    profile = profile_dataset(df)

    total_employees = profile["n_rows"]
    attrition_yes = profile["class_counts"][config.POSITIVE_LABEL]
    active_employees = total_employees - attrition_yes
    attrition_rate = profile["attrition_rate"] * 100

    # KPI Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Employees",
            f"{total_employees:,}"
        )

    with col2:
        st.metric(
            "Attrition Rate",
            f"{attrition_rate:.1f}%"
        )

    with col3:
        st.metric(
            "Active Employees",
            f"{active_employees:,}"
        )

    st.markdown("---")

    # Attrition Charts
    st.subheader("Employee Attrition Insights")

    col1, col2 = st.columns(2)

    with col1:
        if "Department" in df.columns:
            st.pyplot(
                plot_attrition_rate_by(
                    df,
                    "Department",
                    "Attrition Rate by Department"
                ),
                use_container_width=True
            )

    with col2:
        if "JobRole" in df.columns:
            st.pyplot(
                plot_attrition_rate_by(
                    df,
                    "JobRole",
                    "Attrition Rate by Job Role"
                ),
                use_container_width=True
            )


def page_analytics() -> None:
    st.header("HR Analytics dashboard")
    st.caption("All numbers on this page are computed from the loaded dataset, not from the model.")
    try:
        df = cached_dataset()
    except DatasetNotFoundError:
        setup_message("dataset")
        return

    profile = profile_dataset(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employee records", f"{profile['n_rows']:,}")
    c2.metric("Columns", profile["n_columns"])
    c3.metric("Attrition Yes", profile["class_counts"][config.POSITIVE_LABEL])
    rate = profile["attrition_rate"]
    c4.metric("Attrition rate", f"{rate * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        st.pyplot(plot_class_distribution(df), use_container_width=True)
    with right:
        if "Department" in df.columns:
            st.pyplot(
                plot_attrition_rate_by(df, "Department", "Attrition rate by department"),
                use_container_width=True,
            )

    if "JobRole" in df.columns:
        st.pyplot(
            plot_attrition_rate_by(df, "JobRole", "Attrition rate by job role"),
            use_container_width=True,
        )
    cols = st.columns(2)
    with cols[0]:
        if "OverTime" in df.columns:
            st.pyplot(
                plot_attrition_rate_by(df, "OverTime", "Attrition rate by overtime"),
                use_container_width=True,
            )
    with cols[1]:
        fig = plot_years_at_company(df)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    if "JobSatisfaction" in df.columns:
        st.pyplot(
            plot_attrition_rate_by(df, "JobSatisfaction", "Attrition rate by job satisfaction"),
            use_container_width=True,
        )

    heat = plot_correlation_heatmap(df)
    if heat is not None:
        with st.expander("Numerical correlation heatmap"):
            st.pyplot(heat, use_container_width=True)

    st.subheader("Observations from this file")
    observations = []
    from src.data_loader import encode_target

    flag = encode_target(df[config.TARGET_COLUMN])
    observations.append(
        f"In this CSV, {int(flag.sum())} of {len(df)} records are Attrition = Yes "
        f"({flag.mean() * 100:.1f}%). The classes are imbalanced."
    )
    if "OverTime" in df.columns:
        rates = df.assign(_f=flag).groupby("OverTime")["_f"].mean()
        if set(rates.index.astype(str)) >= {"Yes", "No"}:
            observations.append(
                f"Overtime Yes attrition rate is {rates.get('Yes', float('nan')) * 100:.1f}%, "
                f"compared with {rates.get('No', float('nan')) * 100:.1f}% for Overtime No."
            )
    if "Department" in df.columns:
        dept = df.assign(_f=flag).groupby("Department")["_f"].mean().sort_values(ascending=False)
        observations.append(
            f"Highest department attrition rate in this file: {dept.index[0]} "
            f"({dept.iloc[0] * 100:.1f}%)."
        )
    for line in observations:
        st.write(f"- {line}")
    st.caption("These patterns describe the IBM sample only. They are not findings about a real workforce.")


def _form_defaults(schema: dict) -> dict:
    values = {}
    for col, meta in schema["numeric_meta"].items():
        values[col] = meta.get("median") if meta.get("median") is not None else 0
    for col, options in schema["categorical_meta"].items():
        values[col] = options[0] if options else ""
    return values


def page_predict() -> None:
    st.header("Employee attrition prediction")
    st.caption("Submit a profile. The saved Scikit-learn pipeline scores it. This is not a guarantee.")

    if not config.PIPELINE_PATH.is_file() or not config.SCHEMA_PATH.is_file():
        setup_message("model")
        return

    schema = cached_schema()
    try:
        cached_pipeline()
    except ModelNotFoundError:
        setup_message("model")
        return

    defaults = _form_defaults(schema)
    employee = {}

    with st.form("employee_form"):
        st.subheader("Employee details")
        st.caption("Fields match the columns used in training. Attrition is not an input.")

        numeric = schema["numeric_features"]
        categorical = schema["categorical_features"]

        st.markdown("**Numerical attributes**")
        n_cols = st.columns(3)
        for i, column in enumerate(numeric):
            meta = schema["numeric_meta"][column]
            low = meta["min"] if meta["min"] is not None else 0
            high = meta["max"] if meta["max"] is not None else 100
            # Allow a little room around observed min/max for new entries.
            min_value = float(min(low, defaults[column]))
            max_value = float(max(high, defaults[column]))
            if min_value == max_value:
                max_value = min_value + 1
            is_int = float(min_value).is_integer() and float(max_value).is_integer()
            with n_cols[i % 3]:
                if is_int:
                    employee[column] = st.number_input(
                        column,
                        min_value=int(min_value),
                        max_value=int(max_value),
                        value=int(round(defaults[column])),
                        step=1,
                        help=f"Observed training range: {int(low)}–{int(high)}",
                    )
                else:
                    employee[column] = st.number_input(
                        column,
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=float(defaults[column]),
                        help=f"Observed training range: {low:.1f}–{high:.1f}",
                    )

        st.markdown("**Categorical attributes**")
        c_cols = st.columns(3)
        for i, column in enumerate(categorical):
            options = schema["categorical_meta"][column]
            with c_cols[i % 3]:
                employee[column] = st.selectbox(column, options=options)

        submitted = st.form_submit_button("Estimate attrition")

    if submitted:
        result = predict_one(employee)
        label = result["prediction_label"]
        st.subheader(f"Attrition: {label}")
        if "probability_attrition_yes" in result:
            prob = float(result["probability_attrition_yes"])
            st.write(
                f"Estimated probability of Attrition = Yes: **{prob:.1%}**. "
                "This is the model's predicted chance that this profile would be "
                "labelled Yes in the IBM sample, not a personal certainty."
            )
            st.progress(min(max(prob, 0.0), 1.0))
        st.markdown(
            '<div class="note">Do not use this output to hire, fire, promote, '
            "or discipline anyone. The dataset is fictional and the model can be wrong, "
            "especially for people unlike the training sample.</div>",
            unsafe_allow_html=True,
        )


def page_performance() -> None:
    st.header("Model performance")
    summary = _load_json(config.TRAINING_SUMMARY_PATH)
    if summary is None or not config.MODEL_COMPARISON_PATH.is_file():
        setup_message("model")
        return

    st.write(
        f"Train/test split: **{int((1 - summary['test_size']) * 100)}% / "
        f"{int(summary['test_size'] * 100)}%**, stratified on Attrition, "
        f"`random_state={summary['random_state']}`. "
        f"Training rows: {summary['n_train']}. Test rows: {summary['n_test']}."
    )
    st.write(f"**Selected model:** {summary['selected_model']}")
    st.write(summary["selection_rule"])

    comparison = pd.read_csv(config.MODEL_COMPARISON_PATH)
    st.subheader("Held-out test metrics")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    selected = comparison.loc[comparison["model"] == summary["selected_model"]].iloc[0]
    fig = plot_confusion_matrix(
        int(selected["true_negative"]),
        int(selected["false_positive"]),
        int(selected["false_negative"]),
        int(selected["true_positive"]),
    )
    left, right = st.columns(2)
    with left:
        st.pyplot(fig, use_container_width=True)
    with right:
        st.subheader("How to read the scores")
        st.write(
            "**Accuracy** is the share of all test employees the model labelled correctly. "
            "It can look strong when most people stay."
        )
        st.write(
            "**Precision** asks: of those flagged as leaving, how many actually left?"
        )
        st.write(
            "**Recall** asks: of those who left, how many did the model find? "
            "Missing a likely departure can matter in attrition work, so recall is reported next to precision."
        )
        st.write(
            "**F1-score** balances precision and recall. **ROC-AUC** summarises ranking quality using probabilities."
        )

    if config.FEATURE_IMPORTANCE_PATH.is_file():
        importance = pd.read_csv(config.FEATURE_IMPORTANCE_PATH)
        fig_imp = plot_feature_importance(importance)
        if fig_imp is not None:
            st.pyplot(fig_imp, use_container_width=True)

    st.caption("Test metrics describe this IBM sample split. They are not a guarantee on new organisations.")


def page_about() -> None:
    st.header("About the project")
    st.write(f"**Student:** {config.STUDENT_NAME}  ")
    st.write(f"**Roll number:** {config.ROLL_NUMBER}  ")
    st.write(f"**Course:** {config.COURSE}")

    st.subheader("Aim")
    st.write(
        "Build a complete, explainable machine-learning pipeline that estimates "
        "employee attrition risk from HR attributes and presents dataset analytics in a dashboard."
    )
    st.subheader("Objectives")
    st.write(
        "- Load and validate a public IBM HR sample with an Attrition target.\n"
        "- Clean data without leakage and encode mixed feature types in a Pipeline.\n"
        "- Compare Logistic Regression, Decision Tree, and Random Forest.\n"
        "- Tune on training folds, then report honest test-set metrics.\n"
        "- Expose analytics and single-profile estimates in Streamlit."
    )
    st.subheader("Dataset")
    st.write(
        "IBM HR Analytics Employee Attrition & Performance (fictional sample). "
        f"Primary file URL used by training: {config.DATASET_URL}. "
        f"Kaggle listing: {config.DATASET_KAGGLE_URL}."
    )
    st.subheader("Pipeline")
    st.write(
        "CSV → validation → train/test split → ColumnTransformer (impute, scale, one-hot) "
        "→ classifier with class_weight='balanced' → GridSearchCV on F1 → joblib pipeline → Streamlit."
    )
    st.subheader("Limitations and responsible use")
    st.write(
        "The CSV is synthetic. Gender is excluded from the model. Predictions must not be used "
        "for automated employment decisions. Protected characteristics and incomplete records "
        "make individual scoring ethically unsafe outside an academic demo."
    )


def main() -> None:
    page = sidebar()
    if page == "Home":
        page_home()
    elif page == "HR Analytics":
        page_analytics()
    elif page == "Predict Attrition":
        page_predict()
    elif page == "Model Performance":
        page_performance()
    else:
        page_about()


if __name__ == "__main__":
    main()
