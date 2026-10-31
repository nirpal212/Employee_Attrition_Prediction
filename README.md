# Employee Attrition Prediction & HR Analytics

**Student:** Vaishnavi Nirpal  
**Roll number:** CS3234  
**Course:** Third-Year B.Tech Computer Science and Engineering  
**Type:** Individual Machine Learning project

A working binary-classification app that estimates whether an employee profile in IBM’s **fictional** HR sample would be labelled Attrition = Yes. The project covers dataset inspection, preprocessing, EDA, three Scikit-learn models, evaluation, a Streamlit dashboard, tests, and deployment notes.

Predictions are statistical estimates from this sample. They are **not** guarantees about any person and must **not** be used to hire, fire, or otherwise decide employment.

## Aim

Build an end-to-end, beginner-explainable machine-learning pipeline for employee attrition classification and present dataset analytics plus individual estimates in a professional dashboard.

## Objectives

1. Load and validate a public IBM HR sample with an `Attrition` target.
2. Inspect real row counts, types, missing values, duplicates, and class balance.
3. Preprocess mixed features in a Scikit-learn `Pipeline` without leaking the test set.
4. Compare Logistic Regression, Decision Tree, and Random Forest.
5. Tune on training folds using F1, then report held-out test metrics.
6. Serve analytics and form-based predictions in Streamlit.

## Problem statement

Human-resources teams care about people leaving (attrition). This academic project asks: given job, pay, and tenure attributes in IBM’s sample table, can a classifier estimate whether a record is labelled **Yes** (left) or **No** (stayed)?

Real-world motivation is only analogical. The CSV is **synthetic**. Nothing in this repository is a finding about a real workforce.

## Dataset source

| Item | Value |
| --- | --- |
| Name | IBM HR Analytics Employee Attrition & Performance |
| Nature | Fictional sample created by IBM data scientists |
| Training download | https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv |
| IBM repository | https://github.com/IBM/employee-attrition-aif360 |
| Kaggle listing | https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset |
| Local file | `Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv` |

If download fails, copy the Kaggle CSV into `Dataset/` using that filename, or change `DATASET_PATH` in `config.py`.

## Dataset description (measured from the file on disk)

These figures come from `python -m src.train_model` reading the actual CSV. They are not copied from a blog.

| Property | Measured value |
| --- | --- |
| Rows | 1470 |
| Columns | 35 |
| Target | `Attrition` (`Yes` / `No`) |
| Class counts | Yes = 237, No = 1233 |
| Attrition rate | 16.12% |
| Missing cells | 0 |
| Duplicate rows | 0 |

Columns include Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EmployeeCount, EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, and Attrition.

## Feature and target selection

- **Target:** `Attrition` mapped to 1 (Yes / left) and 0 (No / stayed).
- **Dropped before modelling (with reasons):**
  - `EmployeeNumber` — identifier, not a job attribute.
  - `Gender` — protected characteristic; excluded on purpose for this academic demo.
  - `EmployeeCount`, `Over18`, `StandardHours` — constant on the training split, so they cannot help a classifier.
- **Remaining inputs:** 23 numeric columns and 6 categorical columns (`BusinessTravel`, `Department`, `EducationField`, `JobRole`, `MaritalStatus`, `OverTime`).
- The target is never used as a prediction input.

## Preprocessing decisions

- Missing values: none in this file; the pipeline still median-imputes numbers and most-frequent-imputes categories so new rows with blanks do not crash the app.
- Duplicates: none found; the trainer would drop them if they appeared.
- Outliers: IQR counts were computed on **training** numeric columns. High-income and long-tenure tails are expected in HR data. **No rows were deleted.** Examples on train: MonthlyIncome 86 IQR flags (1009–19973), TotalWorkingYears 52 flags (0–40), YearsAtCompany 52 flags (0–37).
- Encoding: `OneHotEncoder(handle_unknown='ignore')` so unseen categories become zeros instead of errors.
- Scaling: `StandardScaler` on numeric columns (required for logistic regression; trees are unaffected).
- Leakage control: `train_test_split` first (stratified). Imputers, scaler, encoder, and GridSearchCV see **training data only**.
- Imbalance: `class_weight='balanced'` inside each classifier. This reweights the minority class using training-fold frequencies. It does not invent synthetic employees.

## EDA findings (computed from all 1470 rows)

- The sample is imbalanced: 237 leavers vs 1233 stayers.
- **Overtime:** attrition rate 30.53% when OverTime = Yes vs 10.44% when No.
- **Department:** Sales 20.63%, Human Resources 19.05%, Research & Development 13.84%.
- **Job role:** highest rate Sales Representative 39.76%; lowest Research Director 2.50%.
- **Job satisfaction:** rate 22.84% at score 1 vs 11.33% at score 4.
- **Years at company:** 29.82% in the 0–2 year band vs 6.67% in 11–20 years.

These numbers describe this IBM file only.

## Algorithms and parameters

All three models use `random_state=42` and `class_weight='balanced'`.

| Model | Search space | Selected hyperparameters |
| --- | --- | --- |
| Logistic Regression | `C ∈ {0.1, 1.0, 10.0}` | `C=0.1`, `max_iter=1000` |
| Decision Tree | `max_depth ∈ {3,5,8,None}`, `min_samples_split ∈ {2,10}` | `max_depth=3`, `min_samples_split=2` |
| Random Forest | `n_estimators ∈ {100,200}`, `max_depth ∈ {5,10,None}`, `min_samples_leaf ∈ {1,4}` | `n_estimators=100`, `max_depth=5`, `min_samples_leaf=1` |

Tuning used 5-fold **StratifiedKFold** on the training set, scoring **F1**.

## Train-test split

- 80% train / 20% test, `stratify=y`, `random_state=42`
- Train rows: 1176
- Test rows: 294
- Positive (Yes) class remains about 16% in both splits

## Model comparison (held-out test set)

Saved in `reports/model_comparison.csv`. CV F1 is the selection score; test columns are reported after selection.

| Model | CV F1 | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | **0.4924** | 0.7755 | 0.3827 | **0.6596** | 0.4844 | **0.8051** |
| Decision Tree | 0.3764 | 0.7279 | 0.3226 | 0.6383 | 0.4286 | 0.6956 |
| Random Forest | 0.4704 | 0.8299 | 0.4717 | 0.5319 | 0.5000 | 0.7730 |

Selected-model test confusion matrix (Logistic Regression): TN 197, FP 50, FN 16, TP 31.

Accuracy alone would favour Random Forest (0.8299). Attrition work also cares about **recall** (finding people who left). Logistic Regression had the best **cross-validated F1** and the highest test recall and ROC-AUC in this run.

## Final model-selection justification

**Winner: Logistic Regression (`C=0.1`).**

Rule used: highest StratifiedKFold F1 on the training set after GridSearchCV. The test set was **not** used to pick the winner. Random Forest had a slightly higher test F1 (0.5000 vs 0.4844) but a lower CV F1 (0.4704 vs 0.4924). Sticking to the CV rule keeps the test numbers an honest check.

## Model improvement experiments

1. Shared `ColumnTransformer` so every algorithm saw the same encoded matrix.
2. `class_weight='balanced'` vs a naive majority classifier (majority accuracy would be ~83.9% by always predicting No, with recall 0 — rejected).
3. GridSearchCV on F1 rather than accuracy so the minority class mattered during tuning.
4. Small hyperparameter grids (see table) comparing baseline defaults to tuned settings; the saved pipeline is the tuned refit model.
5. Feature filtering: identifiers, constants, and Gender removed; remaining columns kept.

The fitted preprocessing + classifier pipeline is stored at `models/attrition_pipeline.joblib`.

## Frontend

`app.py` is a Streamlit dashboard (navy / teal, light background, sidebar):

- **Home** — project summary, dataset status, model status
- **HR Analytics** — counts and charts from the CSV (not from the model)
- **Predict Attrition** — form generated from `models/feature_schema.json`; calls the joblib pipeline
- **Model Performance** — comparison table, confusion matrix, coefficient magnitudes, metric explanations
- **About** — aim, dataset, pipeline, limitations

If the CSV or the model file is missing, the UI shows setup instructions instead of invented scores.

## Project folder structure

```
Employee_Attrition_Prediction/
    app.py
    config.py
    requirements.txt
    README.md
    .gitignore
    Dataset/          # IBM sample CSV
    notebooks/        # EDA notebook (uses src/)
    src/              # load, preprocess, train, evaluate, predict, plots
    models/           # joblib pipeline + feature schema
    reports/          # metrics, EDA JSON, academic report
    tests/
```

How modules connect: `config.py` holds paths → `data_loader.py` reads the CSV → `preprocessing.py` builds the ColumnTransformer → `train_model.py` fits three pipelines and writes reports → `predict.py` loads the winner → `app.py` displays analytics and scores one form row → `visualizations.py` draws charts.

## Installation

PowerShell, from the project folder:

```powershell
cd C:\Users\vaish\Employee_Attrition_Prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Inspect the dataset

```powershell
.\.venv\Scripts\python -c "from src.data_loader import load_dataset, profile_dataset; import json; print(json.dumps(profile_dataset(load_dataset()), indent=2))"
```

## Train

```powershell
.\.venv\Scripts\python -m src.train_model
```

This downloads the IBM sample if the CSV is missing, trains the three models, and writes `models/` and `reports/`.

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

Last run on this machine: **10 passed**.

## Launch the app

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

Then open the Local URL printed in the terminal (usually http://localhost:8501).

## GitHub publishing (not done from this session)

Git was not available on the PATH of this PC when the project was built, so **no GitHub upload was performed**. After installing [Git for Windows](https://git-scm.com/download/win):

```powershell
cd C:\Users\vaish\Employee_Attrition_Prediction
git init
git add .
git commit -m "Add employee attrition prediction ML project"
gh repo create Employee_Attrition_Prediction --private --source=. --push
```

Do not commit `.venv/`, `.streamlit/secrets.toml`, or personal secrets. The IBM sample CSV and the trained joblib file may be committed so Streamlit Community Cloud can run without retraining.

## Streamlit Community Cloud

1. Push the repo to GitHub.
2. At https://share.streamlit.io choose the repo, branch, and main file `app.py`.
3. The Cloud installer reads `requirements.txt`.
4. Keep `Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv` and `models/attrition_pipeline.joblib` in the repo, or the app will show the setup message.

This project has **not** been deployed to Streamlit Cloud from this session.

## Screenshots

After `streamlit run app.py`, capture:

1. Home with Dataset Ready and selected model Logistic Regression
2. HR Analytics KPI row (1470 records, 237 Yes, 16.1% rate)
3. Prediction form result (Attrition Yes/No plus probability)
4. Model Performance table matching `reports/model_comparison.csv`

## Limitations and future scope

- The dataset is fictional; coefficients describe IBM’s generator, not a company.
- Gender is excluded; Age and other attributes can still proxy protected traits.
- Logistic Regression with balanced weights raises recall and lowers precision/accuracy versus always predicting No.
- No causal claims. Overtime correlates with attrition in this file; it does not prove overtime causes quits.
- Future work: cost-sensitive thresholds, calibration plots, organisation-specific real data under an ethics review, and excluding additional sensitive fields.

## Viva map (what to say for each module)

| File | One-sentence purpose |
| --- | --- |
| `config.py` | Paths, random seed, target name |
| `src/data_loader.py` | Read/validate CSV; never invent rows |
| `src/preprocessing.py` | Drop bad columns; impute, scale, one-hot |
| `src/train_model.py` | Split, tune, compare, save pipeline |
| `src/evaluate_model.py` | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| `src/predict.py` | Load joblib; score a form row |
| `src/visualizations.py` | Charts for the app and notebook |
| `app.py` | Streamlit pages |
| `tests/` | Missing-file errors, schema order, prediction shape |
