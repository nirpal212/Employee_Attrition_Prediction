# Dataset folder

Place the IBM HR Analytics Employee Attrition CSV here:

- File name expected by `config.py`: `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- Target column: `Attrition` with values `Yes` / `No`

## Preferred automatic source

Training (`python -m src.train_model`) downloads IBM's published sample from:

https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv

Repository: https://github.com/IBM/employee-attrition-aif360

That file is **fictional** sample data created by IBM. It is not a record of real employees.

## Manual download

If the automatic download fails:

1. Open the Kaggle listing: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
2. Download `WA_Fn-UseC_-HR-Employee-Attrition.csv`
3. Copy it into this `Dataset/` folder using the filename above

You can also point `DATASET_PATH` in `config.py` at another location.

Do not replace this file with made-up rows and present them as the IBM dataset.
