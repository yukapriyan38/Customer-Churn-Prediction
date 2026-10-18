# Customer Churn Prediction

A full-stack machine learning project that predicts whether a telecom customer is likely to churn. The repository includes a data-cleaning pipeline, exploratory data analysis, model training, a FastAPI inference service, and a React + Vite dashboard for end-user prediction.

## Overview

This project uses the IBM Telco Customer Churn dataset to train and compare multiple classifiers, then deploys the best model through a lightweight API and web UI. The app helps business teams quickly assess customer churn risk and make retention decisions based on churn probability.

## Features

- Cleans and validates the Telco Customer Churn dataset
- Produces exploratory charts and data-quality summaries
- Trains and compares three models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Selects the best model using ROC-AUC
- Saves model artifacts and comparison metrics to `models/`
- Exposes predictions via a FastAPI backend
- Provides a React dashboard for interactive churn prediction
- Includes a CLI prediction script for quick single-record predictions

## Project structure

```text
churn/
├── README.md                     # Project overview and setup guide
├── clean_data.py                 # Cleans the raw Telco dataset
├── check_data.py                 # Data inspection and quality checks
├── eda.py                        # Generates exploratory visualizations
├── logistic_model.py             # Logistic Regression training
├── random_forest_model.py        # Random Forest training
├── xgboost_model.py              # XGBoost training
├── compare_models.py              # Compares metrics and chooses best model
├── predict.py                    # CLI-based single-record churn predictor
├── backend/
│   ├── main.py                   # FastAPI app for prediction, overview and model metadata
│   └── requirements.txt          # Python API dependencies
├── frontend/
│   ├── src/                     # React frontend source
│   ├── package.json             # Frontend dependencies and scripts
│   ├── vite.config.js           # Vite configuration
│   └── README.md                # Default Vite template text
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original raw dataset
│   └── cleaned_telco_churn.csv                # Cleaned data used by models
├── models/
│   ├── best_model.joblib        # Best saved model based on ROC-AUC
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── logistic_regression_metrics.json
│   ├── random_forest_metrics.json
│   ├── xgboost_metrics.json
│   └── model_comparison.csv
├── plots/                       # EDA and evaluation visualizations
├── venv/                        # Local virtual environment (generated)
├── .gitignore                   # Ignore generated artifacts and local envs
└── .git/                        # Git metadata
```

## Tech stack

- Python 3.10+
- pandas
- scikit-learn
- XGBoost
- joblib
- matplotlib
- seaborn
- FastAPI
- Uvicorn
- React
- Vite

## Model performance

The current model comparison uses a stratified 80/20 train-test split with `random_state=42`.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 73.81% | 50.43% | 78.34% | 61.36% | 84.16% |
| XGBoost | 79.91% | 65.22% | 52.14% | 57.95% | 83.88% |
| Random Forest | 76.93% | 55.76% | 63.37% | 59.32% | 82.27% |

The best-performing model is currently Logistic Regression because it achieves the highest ROC-AUC, as well as the strongest recall and F1-score. This makes it a strong choice when the business priority is to catch more likely churners.

## Setup

### 1. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

### 3. Install frontend dependencies

```powershell
cd frontend
npm install
```

## Data preparation

Place the raw Telco CSV in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`, then run:

```powershell
python clean_data.py
```

This script:

- loads the raw dataset
- cleans missing or inconsistent values
- converts `TotalCharges` to numeric
- fills missing charge values for new customers
- converts `Churn` into binary labels (`0`/`1`)
- saves the cleaned data to `data/cleaned_telco_churn.csv`

## Exploratory analysis and plotting

Run the EDA workflow from the root folder:

```powershell
python check_data.py
python eda.py
```

This creates summary tables and visualizations in `plots/` such as churn distribution, tenure analysis, feature importance, and correlation charts.

## Training the models

Train the three classifiers:

```powershell
python logistic_model.py
python random_forest_model.py
python xgboost_model.py
```

Each training script saves a model file and per-model metrics JSON under `models/`.

## Selecting the best model

After all training runs are complete, run:

```powershell
python compare_models.py
```

This script:

- reads all model metric files
- ranks models by ROC-AUC
- saves the comparison table to `models/model_comparison.csv`
- copies the top model to `models/best_model.joblib`

## Running the API

Start the FastAPI backend from the project root:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at:

- `http://localhost:8000/api/health`
- `http://localhost:8000/api/overview`
- `http://localhost:8000/api/models`
- `http://localhost:8000/api/predict`

### API example

`POST /api/predict` accepts a customer payload like:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.7,
  "TotalCharges": 848.4
}
```

## Running the frontend

In a separate terminal:

```powershell
cd frontend
npm run dev
```

Then open the local Vite URL, typically:

```text
http://localhost:5173
```

The frontend loads overview metrics from the backend and lets you enter customer details to generate a churn prediction.

## CLI prediction

You can also run a quick single-record prediction in the terminal:

```powershell
python predict.py
```

This script prompts for customer attributes and prints the predicted churn class and probability using the saved best model.

## How the prediction pipeline works

1. Load the cleaned Telco customer data
2. Separate the target (`Churn`) from the feature columns
3. Identify categorical and numeric variables
4. Apply one-hot encoding to categorical data
5. Scale numeric values for algorithms that require them
6. Train the selected classifier
7. Save the trained pipeline and evaluation metrics
8. Use the saved model to score a new customer record

Because the model is saved as a complete scikit-learn pipeline, preprocessing is preserved automatically during inference.

## Business value

This churn model can help telecom teams identify customers at risk of leaving and prioritize retention campaigns, contract adjustments, service outreach, or targeted offers. In a churn context, recall matters when missing a potential churner is costly; precision becomes more important when outreach is expensive.

## Limitations and future improvements

- The current evaluation uses a single stratified train-test split
- Default classification thresholds are used without tuning
- Hyperparameter tuning and cross-validation would improve model reliability
- Model performance should be monitored over time as customer behavior changes
- Future enhancements may include threshold optimization, calibration, SHAP explanations, and monitoring for data drift

## Dataset

This project uses the public IBM Telco Customer Churn dataset. Review the original license and usage terms before redistributing the raw CSV.
