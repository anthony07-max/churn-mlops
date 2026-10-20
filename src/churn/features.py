"""Schema and cleaning shared by training and the API, so both see identical features."""
import pandas as pd

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
               "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Blank TotalCharges only occurs for brand-new customers (tenure 0): they have not been billed yet.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes", "0": "No", "1": "Yes"}).fillna(df["SeniorCitizen"])
    return df
