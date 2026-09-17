# Customer Churn Prediction

An end-to-end machine-learning project that predicts whether a telecom customer is likely to churn. The project includes data cleaning, exploratory data analysis, three classification models, model comparison, persisted model pipelines, and an interactive Streamlit dashboard.

## Features

- Cleans and validates the Telco Customer Churn dataset.
- Explores churn patterns with reusable visualizations.
- Trains Logistic Regression, Random Forest, and XGBoost models.
- Evaluates models using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Selects the best model by ROC-AUC.
- Provides customer-level churn prediction through a Streamlit web app.
- Displays dataset summaries, EDA charts, model comparisons, and prediction results.

## Project structure

```text
churn/
├── app.py                         # Streamlit dashboard
├── clean_data.py                  # Cleans the original CSV
├── check_data.py                  # Inspects data quality and target distribution
├── eda.py                         # Generates exploratory visualizations
├── logistic_model.py              # Trains Logistic Regression
├── random_forest_model.py         # Trains Random Forest
├── xgboost_model.py               # Trains XGBoost
├── compare_models.py              # Compares metrics and selects best model
├── data/
│   └── cleaned_telco_churn.csv   # Cleaned input dataset
├── models/
│   ├── *_metrics.json             # Saved evaluation metrics
│   └── model_comparison.csv       # Combined model results
└── plots/                         # EDA and evaluation charts
```

Model binaries (`*.joblib`), the local virtual environment, and generated plots are excluded from Git by [.gitignore](./.gitignore). They are created locally by the training scripts.

## Results

The current comparison uses a stratified 80/20 train-test split with `random_state=42`.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 73.81% | 50.43% | **78.34%** | **61.36%** | **84.16%** |
| XGBoost | **79.91%** | **65.22%** | 52.14% | 57.95% | 83.88% |
| Random Forest | 76.93% | 55.76% | 63.37% | 59.32% | 82.27% |

Logistic Regression currently provides the highest ROC-AUC, recall, and F1-score. XGBoost provides the highest accuracy and precision. The preferred model depends on whether the business prioritizes finding more potential churners or minimizing false alerts.

## Requirements

- Python 3.10 or newer
- pandas
- scikit-learn
- xgboost
- joblib
- matplotlib
- seaborn
- streamlit

## Installation

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install pandas scikit-learn xgboost joblib matplotlib seaborn streamlit
```

If PowerShell blocks script activation, run the commands with the virtual-environment interpreter directly:

```powershell
.\venv\Scripts\python.exe -m pip install pandas scikit-learn xgboost joblib matplotlib seaborn streamlit
```

## Usage

Run the following commands from the project root.

### 1. Clean the raw dataset

Place the original Telco CSV at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`, then run:

```powershell
.\venv\Scripts\python.exe clean_data.py
```

This creates `data/cleaned_telco_churn.csv`, converts `TotalCharges` to numeric, fills new-customer charge values, and converts `Churn` to `0/1`.

### 2. Check the data

```powershell
.\venv\Scripts\python.exe check_data.py
```

### 3. Generate EDA plots

```powershell
.\venv\Scripts\python.exe eda.py
```

### 4. Train the models

```powershell
.\venv\Scripts\python.exe logistic_model.py
.\venv\Scripts\python.exe random_forest_model.py
.\venv\Scripts\python.exe xgboost_model.py
```

Each script saves a model pipeline and a metrics JSON file in `models/`.

### 5. Compare models

```powershell
.\venv\Scripts\python.exe compare_models.py
```

This writes `models/model_comparison.csv` and copies the highest-ROC-AUC model to `models/best_model.joblib`.

### 6. Start the dashboard

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## How the prediction pipeline works

1. Load the cleaned customer data.
2. Separate the `Churn` target from customer features.
3. Detect categorical and numerical columns.
4. Apply one-hot encoding to categorical features.
5. Scale numerical features for Logistic Regression.
6. Train the selected classifier.
7. Generate a churn class and probability for a new customer.

The saved model is a complete scikit-learn pipeline, so preprocessing used during training is reused automatically during prediction.

## Business interpretation

Churn prediction can help a telecom provider identify customers who may leave and prioritize retention activities such as contract upgrades, service support, or targeted offers. Recall is particularly important when the cost of missing a potential churner is high; precision is more important when retention outreach is expensive.

## Limitations and future improvements

- The current evaluation is based on one stratified train-test split.
- The default classification threshold is used; threshold tuning may improve the precision-recall trade-off.
- Hyperparameter tuning and cross-validation should be added before production use.
- The dataset is historical and may not represent current customer behavior.
- Future work could include calibration, Precision-Recall curves, SHAP explanations, and monitoring for data drift.

## Resume-ready summary

Built an end-to-end telecom customer churn prediction system using Python, Pandas, Scikit-learn, XGBoost, and Streamlit. Compared three classification models using five evaluation metrics, selected the best model by ROC-AUC, persisted reusable preprocessing pipelines, and deployed an interactive dashboard for customer-level churn risk prediction.

## Dataset

This project uses the public IBM Telco Customer Churn dataset. Review the dataset's original license and usage terms before redistributing the raw CSV.
