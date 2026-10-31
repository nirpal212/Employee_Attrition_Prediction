"""
Project configuration.

Beginner note:
    Keep paths and training settings in one place so app.py, training, and
    tests all use the same values. You can change DATASET_PATH if your CSV
    lives somewhere else.
"""

from pathlib import Path

# Root folder of this project (the folder that contains app.py).
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "Dataset"
# Official IBM sample filename used by many public mirrors of this dataset.
DATASET_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
DATASET_PATH = DATASET_DIR / DATASET_FILENAME

# IBM's published sample CSV (fictional HR records, not real employees).
# Source repository: https://github.com/IBM/employee-attrition-aif360
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/"
    "master/data/emp_attrition.csv"
)

# Kaggle listing of the same well-known sample dataset.
DATASET_KAGGLE_URL = (
    "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"
)

TARGET_COLUMN = "Attrition"
POSITIVE_LABEL = "Yes"  # employee left
NEGATIVE_LABEL = "No"   # employee stayed

# Domain knowledge: unique employee id, not a useful predictor.
IDENTIFIER_COLUMNS = ["EmployeeNumber"]

# Protected characteristic: excluded from the model on purpose.
# The column remains in the CSV for dataset documentation only.
EXCLUDED_SENSITIVE_COLUMNS = ["Gender"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "attrition_pipeline.joblib"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
TRAINING_SUMMARY_PATH = PROJECT_ROOT / "reports" / "training_summary.json"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
DATASET_PROFILE_PATH = PROJECT_ROOT / "reports" / "dataset_profile.json"
EDA_SUMMARY_PATH = PROJECT_ROOT / "reports" / "eda_summary.json"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "feature_importance.csv"

STUDENT_NAME = "Vaishnavi Nirpal"
ROLL_NUMBER = "CS3234"
COURSE = "Third-Year B.Tech Computer Science and Engineering"
PROJECT_TITLE = "Employee Attrition Prediction & HR Analytics"
