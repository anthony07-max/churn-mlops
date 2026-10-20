import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
BASE = dict(tenure=2, MonthlyCharges=90.0, TotalCharges=180.0, gender="Female", SeniorCitizen="No", Partner="No",
            Dependents="No", PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic", OnlineSecurity="No",
            OnlineBackup="No", DeviceProtection="No", TechSupport="No", StreamingTV="Yes", StreamingMovies="Yes",
            Contract="Month-to-month", PaperlessBilling="Yes", PaymentMethod="Electronic check")

def test_health():
    assert client.get("/health").json()["status"] == "ok"

def test_predict_shape_and_range():
    r = client.post("/predict", json=[BASE]); assert r.status_code == 200
    assert 0 <= r.json()[0]["churn_probability"] <= 1

def test_risk_ordering():
    loyal = {**BASE, "tenure": 70, "TotalCharges": 6000.0, "Contract": "Two year", "PaymentMethod": "Bank transfer (automatic)",
             "OnlineSecurity": "Yes", "TechSupport": "Yes"}
    r = client.post("/predict", json=[BASE, loyal]).json()
    assert r[0]["churn_probability"] > r[1]["churn_probability"]

def test_validation_rejects_bad_input():
    assert client.post("/predict", json=[{**BASE, "Contract": "Lifetime"}]).status_code == 422
