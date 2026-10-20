"""Download the IBM Telco Customer Churn sample dataset (public, 7,043 customers)."""
from pathlib import Path
import requests

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, timeout=60); r.raise_for_status()
    (RAW / "telco_churn.csv").write_bytes(r.content)
    print("saved", len(r.content), "bytes")

if __name__ == "__main__":
    main()
