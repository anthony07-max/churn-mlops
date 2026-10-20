"""FastAPI service for churn scoring."""
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ART = joblib.load(Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib")
YN = Literal["Yes", "No"]
app = FastAPI(title="Telco churn scoring API", version=ART["version"])

class Customer(BaseModel):
    tenure: int = Field(ge=0, le=120)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    gender: Literal["Male", "Female"]
    SeniorCitizen: YN
    Partner: YN
    Dependents: YN
    PhoneService: YN
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YN
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]

@app.get("/health")
def health():
    return {"status": "ok", "model_version": ART["version"]}

@app.post("/predict")
def predict(customers: list[Customer]):
    X = pd.DataFrame([c.model_dump() for c in customers])[ART["features"]]
    p = ART["model"].predict_proba(X)[:, 1]
    return [{"churn_probability": round(float(v), 4), "contact_for_retention": bool(v >= ART["threshold"])} for v in p]
