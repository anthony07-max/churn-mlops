"""Train, compare and save the churn model with evaluation artefacts."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from churn.features import clean, NUMERIC, CATEGORICAL, FEATURES

ROOT = Path(__file__).resolve().parents[2]
C, M, P = ROOT / "charts", ROOT / "models", ROOT / "data" / "processed"
for d in (C, M, P): d.mkdir(parents=True, exist_ok=True)

def build(kind):
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    clf = LogisticRegression(max_iter=2000, C=0.5) if kind == "logreg" else \
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, l2_regularization=1.0, random_state=0)
    return Pipeline([("pre", pre), ("clf", clf)])

def main():
    df = clean(pd.read_csv(ROOT / "data" / "raw" / "telco_churn.csv"))
    y = (df.Churn == "Yes").astype(int)
    X = df[FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    res = {}
    for kind in ["logreg", "gbm"]:
        res[kind] = float(cross_val_score(build(kind), Xtr, ytr, cv=cv, scoring="roc_auc").mean())
    best = max(res, key=res.get)
    model = build(best).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]

    # Retention-campaign threshold. Assumptions: £50 per contacted customer (call + discount), a saved customer is worth 12 months of charges, 25% of contacted churners are saved
    value = 12 * Xte.MonthlyCharges.values * 0.25
    ths = np.linspace(0.05, 0.9, 86)
    profit = [((p >= t) * (yte.values * value - 50)).sum() for t in ths]
    t_best = float(ths[int(np.argmax(profit))])

    metrics = {"cv_auc": res, "chosen_model": best, "test_auc": roc_auc_score(yte, p),
               "test_pr_auc": average_precision_score(yte, p), "test_brier": brier_score_loss(yte, p),
               "churn_rate": float(y.mean()), "n_customers": int(len(df)), "threshold": t_best,
               "campaign_profit_at_threshold": float(max(profit)),
               "recall_at_threshold": float(((p >= t_best) & (yte == 1)).sum() / yte.sum()),
               "contact_share_at_threshold": float((p >= t_best).mean())}
    seg = df.groupby("Contract").Churn.apply(lambda s: (s == "Yes").mean()).round(3).to_dict()
    metrics["churn_by_contract"] = seg
    metrics["churn_first_year"] = float((df[df.tenure <= 12].Churn == "Yes").mean())
    metrics["churn_after_4y"] = float((df[df.tenure > 48].Churn == "Yes").mean())

    imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=0)
    fi = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
    fi.to_csv(P / "feature_importance.csv")

    fpr, tpr, _ = roc_curve(yte, p)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set_title(f"ROC (test AUC {metrics['test_auc']:.3f})")
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[1].plot(ths, np.array(profit) / 1000); ax[1].axvline(t_best, color="r", ls="--")
    ax[1].set_title("Retention campaign value by threshold (test set)"); ax[1].set_xlabel("Probability threshold"); ax[1].set_ylabel("Net value, £k")
    fig.tight_layout(); fig.savefig(C / "01_roc_and_threshold.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 6)); fi.tail(12).plot.barh(ax=ax, color="#3d5a80")
    ax.set_title("Permutation importance (drop in test AUC)"); fig.tight_layout(); fig.savefig(C / "02_feature_importance.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    (df.groupby("Contract").Churn.apply(lambda s: (s == "Yes").mean()) * 100).plot.bar(ax=ax[0], color="#ee6c4d", rot=0)
    ax[0].set_ylabel("Churn rate, %"); ax[0].set_title("Churn by contract type")
    b = pd.cut(df.tenure, [-1, 6, 12, 24, 36, 48, 60, 72])
    (df.groupby(b, observed=True).Churn.apply(lambda s: (s == "Yes").mean()) * 100).plot.bar(ax=ax[1], color="#98c1d9", rot=30)
    ax[1].set_title("Churn by tenure (months)"); fig.tight_layout(); fig.savefig(C / "03_churn_drivers.png", dpi=150); plt.close()

    # Calibration
    fig, ax = plt.subplots(figsize=(5, 5)); q = pd.qcut(p, 10, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": yte.values}).groupby(q, observed=True).mean()
    ax.plot(cal.p, cal.y, "o-"); ax.plot([0, 1], [0, 1], "k--"); ax.set_title("Calibration (test deciles)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); fig.tight_layout(); fig.savefig(C / "04_calibration.png", dpi=150); plt.close()

    joblib.dump({"model": model, "threshold": t_best, "features": FEATURES, "version": "1.0.0"}, M / "churn_model.joblib")
    # Reference distribution for drift monitoring
    Xtr.to_csv(P / "reference_sample.csv", index=False)
    (P / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
