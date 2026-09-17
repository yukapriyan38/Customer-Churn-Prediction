import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_PATH = Path("data/cleaned_telco_churn.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    y = df["Churn"]
    if y.dtype == "object":
        y = y.map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Churn"])
    return X, y.astype(int)


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": "XGBoost",
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    joblib.dump(pipeline, MODEL_DIR / "xgboost.joblib")
    with open(MODEL_DIR / "xgboost_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("XGBOOST RESULTS")
    for key, value in metrics.items():
        if key != "Model":
            print(f"{key}: {value:.6f}")
    print("Saved model to models/xgboost.joblib")


if __name__ == "__main__":
    main()
