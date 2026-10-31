# Employee Attrition Prediction & HR Analytics

**Student:** Vaishnavi Nirpal  
**Roll Number:** CS3234  
**Course:** Third-Year B.Tech Computer Science and Engineering  
**Project Type:** Individual Machine Learning Project

An end-to-end binary-classification project that estimates whether an employee profile in IBM’s **fictional HR sample dataset** would be labelled `Attrition = Yes`. The project includes dataset inspection, preprocessing, exploratory data analysis (EDA), three Scikit-learn models, model evaluation, a Streamlit dashboard, automated tests, and deployment instructions.

> **Responsible use:** Predictions are statistical estimates based on a fictional sample. They are not guarantees about any person and must not be used to hire, fire, or otherwise make employment decisions.

## Aim

To build an end-to-end, beginner-explainable machine-learning pipeline for employee attrition classification and present dataset analytics and individual estimates through a professional dashboard.

## Objectives

1. Load and validate a public IBM HR sample dataset containing an `Attrition` target.
2. Inspect row counts, data types, missing values, duplicates, and class balance.
3. Preprocess mixed numerical and categorical features using a Scikit-learn `Pipeline` without data leakage.
4. Compare Logistic Regression, Decision Tree, and Random Forest classifiers.
5. Tune models using training folds and F1-score, then report held-out test metrics.
6. Display analytics and form-based predictions through Streamlit.

## Problem Statement

Employee attrition refers to employees leaving an organization. This academic project investigates whether job, pay, and tenure attributes in IBM’s sample dataset can help a classifier estimate whether a record is labelled **Yes** (left) or **No** (stayed).

The real-world motivation is illustrative only. The CSV is a fictional sample, and this repository does not present findings about a real workforce.

## Dataset Source

| Item | Value |
|---|---|
| Dataset name | IBM HR Analytics Employee Attrition & Performance |
| Nature | Fictional sample created by IBM data scientists |
| Training download | [IBM sample CSV](https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv) |
| IBM repository | [IBM employee-attrition-aif360](https://github.com/IBM/employee-attrition-aif360) |
| Kaggle listing | [IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| Local file | `Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv` |

If the download fails, place the Kaggle CSV in the `Dataset/` directory using the filename above, or update `DATASET_PATH` in `config.py`.

## Dataset Description

The following figures were measured from the local CSV by the project’s data-loading and training workflow.

| Property | Measured value |
|---|---:|
| Rows | 1,470 |
| Columns | 35 |
| Target | `Attrition` (`Yes` / `No`) |
| Employees labelled Yes | 237 |
| Employees labelled No | 1,233 |
| Attrition rate | 16.12% |
| Missing cells | 0 |
| Duplicate rows | 0 |

The dataset includes attributes such as Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EnvironmentSatisfaction, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, NumCompaniesWorked, OverTime, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, and Attrition.

## Feature and Target Selection

- **Target:** `Attrition` is mapped to `1` for Yes (left) and `0` for No (stayed).
- **Dropped columns:**
  - `EmployeeNumber` — identifier, not a job attribute.
  - `Gender` — excluded as a protected characteristic for this academic demonstration.
  - `EmployeeCount`, `Over18`, and `StandardHours` — constant columns that provide no useful variation for classification.
- **Remaining inputs:** 23 numerical columns and 6 categorical columns:
  `BusinessTravel`, `Department`, `EducationField`, `JobRole`, `MaritalStatus`, and `OverTime`.
- The target column is never used as a prediction input.

## Preprocessing Decisions

- **Missing values:** None were found in the supplied dataset. The pipeline still uses median imputation for numerical features and most-frequent imputation for categorical features.
- **Duplicates:** None were found. The training workflow removes duplicates if they occur.
- **Outliers:** IQR-based counts were calculated on training numerical columns. High-income and long-tenure values were treated as plausible data tails; no rows were deleted. Examples from the training data include:
  - `MonthlyIncome`: 86 IQR flags (1,009–19,973)
  - `TotalWorkingYears`: 52 IQR flags (0–40)
  - `YearsAtCompany`: 52 IQR flags (0–37)
- **Encoding:** `OneHotEncoder(handle_unknown="ignore")` handles categorical features and unseen categories.
- **Scaling:** `StandardScaler` is applied to numerical features. This is useful for Logistic Regression; tree-based models do not require feature scaling.
- **Leakage control:** The data is split using stratified `train_test_split` before fitting preprocessing steps. Imputers, scaler, encoder, and `GridSearchCV` are fitted using training data only.
- **Class imbalance:** Each classifier uses `class_weight="balanced"` to reweight the minority class based on training-fold frequencies. No synthetic employee records are created.

## Exploratory Data Analysis Findings

The following patterns were calculated from all 1,470 rows in the IBM sample:

- The dataset is imbalanced: 237 records are labelled Yes and 1,233 are labelled No.
- **Overtime:** Attrition rate was 30.53% for employees with overtime and 10.44% for employees without overtime.
- **Department:** Sales 20.63%, Human Resources 19.05%, and Research & Development 13.84%.
- **Job role:** Sales Representative had the highest rate at 39.76%; Research Director had the lowest at 2.50%.
- **Job satisfaction:** Attrition rate was 22.84% for satisfaction score 1 and 11.33% for score 4.
- **Years at company:** Attrition rate was 29.82% in the 0–2 year band and 6.67% in the 11–20 year band.

These observations describe this IBM sample only. They do not establish causation or describe a real organization.

## Algorithms and Hyperparameters

All three models use `random_state=42` and `class_weight="balanced"`.

| Model | Search space | Selected hyperparameters |
|---|---|---|
| Logistic Regression | `C ∈ {0.1, 1.0, 10.0}` | `C=0.1`, `max_iter=1000` |
| Decision Tree | `max_depth ∈ {3, 5, 8, None}`, `min_samples_split ∈ {2, 10}` | `max_depth=3`, `min_samples_split=2` |
| Random Forest | `n_estimators ∈ {100, 200}`, `max_depth ∈ {5, 10, None}`, `min_samples_leaf ∈ {1, 4}` | `n_estimators=100`, `max_depth=5`, `min_samples_leaf=1` |

Hyperparameter tuning used 5-fold `StratifiedKFold` cross-validation on the training set, with **F1-score** as the scoring metric.

## Train-Test Split

- Split: 80% training and 20% testing
- Stratification: `stratify=y`
- Random seed: `random_state=42`
- Training rows: 1,176
- Testing rows: 294
- The positive class remains approximately 16% in both sets.

## Model Comparison

The results are saved in `reports/model_comparison.csv`. Cross-validation F1 was used for model selection; test metrics were reported after selection.

| Model | CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.4924** | 0.7755 | 0.3827 | **0.6596** | 0.4844 | **0.8051** |
| Decision Tree | 0.3764 | 0.7279 | 0.3226 | 0.6383 | 0.4286 | 0.6956 |
| Random Forest | 0.4704 | **0.8299** | **0.4717** | 0.5319 | **0.5000** | 0.7730 |

Selected-model test confusion matrix for Logistic Regression:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | TN = 197 | FP = 50 |
| Actual Yes | FN = 16 | TP = 31 |

Accuracy alone favours Random Forest in this test run. Attrition analysis may also value recall because it measures how many actual positive records are identified.

## Final Model Selection

**Selected model: Logistic Regression (`C=0.1`)**

The selection rule was the highest 5-fold StratifiedKFold F1-score on the training data after `GridSearchCV`. The test set was not used to select the winning model.

Random Forest achieved a slightly higher test F1-score (0.5000 compared with 0.4844), but Logistic Regression had the higher cross-validated F1-score (0.4924 compared with 0.4704), as well as higher test recall and ROC-AUC in this run. Following the cross-validation selection rule keeps the test set as an independent evaluation.

## Model Improvement Experiments

1. Used a shared `ColumnTransformer` so all algorithms received the same encoded feature matrix.
2. Used `class_weight="balanced"` to address class imbalance instead of relying on a majority-class classifier.
3. Tuned models using F1-score rather than accuracy, giving more attention to the minority class.
4. Compared baseline and tuned settings using small, explainable hyperparameter grids.
5. Removed identifiers, constant columns, and Gender while retaining the remaining selected features.

The fitted preprocessing and classifier pipeline is stored at:

`models/attrition_pipeline.joblib`

## Streamlit Application

The project uses `app.py` to provide a Streamlit dashboard with a navy and teal theme and a light background.

- **Home:** Project summary, dataset status, and model status.
- **HR Analytics:** Counts and charts generated from the CSV, not from model predictions.
- **Predict Attrition:** Form generated from `models/feature_schema.json`; submits a profile to the saved pipeline.
- **Model Performance:** Model comparison table, confusion matrix, coefficient magnitudes, and metric explanations.
- **About:** Aim, dataset information, pipeline, and limitations.

If the dataset or model file is missing, the application displays setup instructions rather than fabricated scores.

## Project Folder Structure

```text
Employee_Attrition_Prediction/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── Dataset/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── notebooks/
│   └── EDA notebook
├── src/
│   ├── data loading
│   ├── preprocessing
│   ├── model training
│   ├── evaluation
│   ├── prediction
│   └── visualizations
├── models/
│   ├── attrition_pipeline.joblib
│   └── feature_schema.json
├── reports/
│   ├── model comparison
│   ├── EDA results
│   └── academic report
└── tests/
```

### Module Workflow

`config.py` defines paths and settings → `data_loader.py` reads and validates the CSV → `preprocessing.py` builds the `ColumnTransformer` → `train_model.py` trains and compares three pipelines and writes reports → `predict.py` loads the selected model → `app.py` displays analytics and predicts from a form → `visualizations.py` creates charts for the app and notebook.

## Installation

Run these commands in PowerShell from the project folder.

```powershell
cd C:\Users\vaish\Employee_Attrition_Prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Inspect the Dataset

```powershell
.\.venv\Scripts\python -c "from src.data_loader import load_dataset, profile_dataset; import json; print(json.dumps(profile_dataset(load_dataset()), indent=2))"
```

## Train the Models

```powershell
.\.venv\Scripts\python -m src.train_model
```

The training workflow downloads the IBM sample if the CSV is missing, trains and evaluates the three models, and writes the model and report files.

## Run Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

**Last recorded test run:** 10 passed.

## Launch the Application

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

Open the Local URL printed in the terminal, usually:

[http://localhost:8501](http://localhost:8501)

## GitHub Repository

The project repository is available here:

[Employee Attrition Prediction & HR Analytics on GitHub](https://github.com/nirpal212/Employee_Attrition_Prediction)

### Clone the Repository

```bash
git clone https://github.com/nirpal212/Employee_Attrition_Prediction.git
cd Employee_Attrition_Prediction
```

### Push Local Updates

If the repository is already connected to the local project, use:

```bash
git status
git add .
git commit -m "Update Employee Attrition Prediction project"
git push
```

Do not commit `.venv/`, `.streamlit/secrets.toml`, or personal secrets. Review the `.gitignore` file before uploading changes.

## Streamlit Community Cloud Deployment

The project has **not yet been deployed** to Streamlit Community Cloud.

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io).
3. Select the repository, branch, and main file `app.py`.
4. Streamlit Cloud installs dependencies from `requirements.txt`.
5. Keep the dataset and `models/attrition_pipeline.joblib` in the repository, or the app will display a setup message.

## Screenshots

After running the application, capture:

1. Home page showing dataset readiness and selected model.
2. HR Analytics KPI row showing 1,470 records, 237 Yes records, and a 16.1% attrition rate.
3. Prediction form showing an Attrition Yes/No result and probability.
4. Model Performance page showing the comparison table and confusion matrix.

## Limitations and Future Scope

- The dataset is fictional; model coefficients describe patterns in this IBM sample, not a real company.
- Gender is excluded, but Age and other attributes may still act as proxies for protected characteristics.
- Balanced Logistic Regression increases attention to the minority class but may produce false positives and lower accuracy than always predicting No.
- The analysis does not establish causal relationships. For example, overtime is associated with attrition in this file, but this does not prove that overtime causes employees to leave.
- Future work could include cost-sensitive thresholds, probability calibration, organization-specific data under an ethics review, and excluding additional sensitive fields.

## Viva Preparation

| File or module | One-sentence purpose |
|---|---|
| `config.py` | Stores paths, random seed, and target name. |
| `src/data_loader.py` | Reads and validates the dataset without inventing rows. |
| `src/preprocessing.py` | Drops unsuitable columns and applies imputation, scaling, and one-hot encoding. |
| `src/train_model.py` | Splits data, tunes models, compares results, and saves the selected pipeline. |
| `src/evaluate_model.py` | Calculates accuracy, precision, recall, F1, ROC-AUC, and confusion matrix. |
| `src/predict.py` | Loads the saved pipeline and predicts from a form row. |
| `src/visualizations.py` | Creates charts for the dashboard and notebook. |
| `app.py` | Runs the Streamlit application pages. |
| `tests/` | Checks missing-file errors, schema order, and prediction output shape. |
