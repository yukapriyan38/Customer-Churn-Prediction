
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="Customer Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL_PATH = ROOT / "models" / "best_model.joblib"
DATA_PATH = ROOT / "data" / "cleaned_telco_churn.csv"
METRICS_PATH = ROOT / "models" / "model_comparison.csv"

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=100)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/api/overview")
def overview():
    if not DATA_PATH.exists():
        raise HTTPException(404, "Dataset not found")

    df = pd.read_csv(DATA_PATH)
    total = len(df)
    churned = int((df["Churn"] == 1).sum())

    return {
        "total_customers": total,
        "churned_customers": churned,
        "churn_rate": round(churned / total * 100, 2)
        if total else 0,
        "preview": df.head(10).fillna("").to_dict("records"),
    }


@app.get("/api/models")
def models():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "Model comparison file not found")

    df = pd.read_csv(METRICS_PATH)
    df = df.astype(object).where(pd.notna(df), None)

    return {"models": df.to_dict("records")}


@app.post("/api/predict")
def predict(customer: Customer):
    try:
        data = pd.DataFrame([customer.model_dump()])
        prediction = int(model.predict(data)[0])
        probability = float(model.predict_proba(data)[0][1])

        return {
            "prediction": (
                "Likely to Churn"
                if prediction == 1
                else "Likely to Stay"
            ),
            "churn_probability": round(probability * 100, 2),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
