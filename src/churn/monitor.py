"""Population Stability Index (PSI) drift check: compare a new batch against the training reference."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from churn.features import NUMERIC, CATEGORICAL, clean

def psi(ref, new, bins=10):
    if ref.dtype.kind in "if":
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref); n = np.histogram(new, edges)[0] / len(new)
    else:
        cats = sorted(set(ref) | set(new))
        r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).values
        n = new.value_counts(normalize=True).reindex(cats, fill_value=0).values
    r, n = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))

def report(ref: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(ref[c], new[c])} for c in NUMERIC + CATEGORICAL]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False)
    out["status"] = pd.cut(out.psi, [-1, 0.1, 0.25, np.inf], labels=["stable", "watch", "drift"])
    return out

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    ref = pd.read_csv(root / "data" / "processed" / "reference_sample.csv")
    new = clean(pd.read_csv(sys.argv[1])) if len(sys.argv) > 1 else None
    if new is None:
        # Demo: a realistic shifted batch = only month-to-month, fibre customers from the real data
        full = clean(pd.read_csv(root / "data" / "raw" / "telco_churn.csv"))
        new = full[(full.Contract == "Month-to-month") & (full.InternetService == "Fiber optic")]
        print("Demo batch: real month-to-month fibre customers only (simulated segment shift)")
    r = report(ref, new); print(r.to_string(index=False))
    r.to_csv(root / "monitoring" / "drift_report_demo.csv", index=False)
