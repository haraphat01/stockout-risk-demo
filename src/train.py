"""Train and evaluate the late-shipment model, then save artifacts for the app.

Run:  python -m src.train
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC, load, time_split

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
N_RISK_BANDS = 4


def gbm():
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=0,
    )


def logreg():
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ]
    )
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))


def scores(y, p):
    return {
        "roc_auc": round(roc_auc_score(y, p), 3),
        "pr_auc": round(average_precision_score(y, p), 3),
        "brier": round(brier_score_loss(y, p), 4),
    }


def top_k_recall(y, p, frac=0.10):
    """Share of late shipments caught if planners review the riskiest `frac` of shipments."""
    k = int(len(p) * frac)
    top = np.argsort(-p)[:k]
    return round(y.iloc[top].sum() / max(y.sum(), 1), 3)


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = load()
    train, val, test = time_split(df)
    X = lambda d: d[FEATURES]

    # Baselines a planner could use today without a model.
    country_rate = train.groupby("country", observed=True)["late"].mean()
    base_rate = train["late"].mean()
    baselines = {
        "Historical late rate by country": test["country"].map(country_rate).astype(float).fillna(base_rate),
        "Late rate by country x mode": test.set_index(["country", "shipment_mode"])
        .index.map(train.groupby(["country", "shipment_mode"], observed=True)["late"].mean())
        .to_series(index=test.index)
        .astype(float)
        .fillna(base_rate),
    }

    models = {"Logistic regression": logreg(), "Gradient boosting": gbm()}
    results = {}
    for name, p in baselines.items():
        results[name] = {**scores(test["late"], p), "recall_top10": top_k_recall(test["late"], p.values)}
    for name, m in models.items():
        m.fit(X(train), train["late"])
        p_val = m.predict_proba(X(val))[:, 1]
        p_test = m.predict_proba(X(test))[:, 1]
        results[name] = {
            **scores(test["late"], p_test),
            "recall_top10": top_k_recall(test["late"], p_test),
            "val_roc_auc": round(roc_auc_score(val["late"], p_val), 3),
        }

    # Pick the model on validation only; the test set is reported, never used for choosing.
    chosen = max(models, key=lambda n: results[n]["val_roc_auc"])
    model = models[chosen]
    p_test = model.predict_proba(X(test))[:, 1]

    # Risk bands from validation predictions; each band keeps its empirical delay distribution
    # so the stock-out simulator samples realistic delays, not a single point estimate.
    p_val = model.predict_proba(X(val))[:, 1]
    edges = np.quantile(p_val, np.linspace(0, 1, N_RISK_BANDS + 1)[1:-1]).tolist()
    trainval = pd.concat([train, val])
    p_tv = model.predict_proba(X(trainval))[:, 1]
    bands = np.digitize(p_tv, edges)
    delay_by_band = {
        int(b): trainval["delay_days"].clip(lower=0)[bands == b].astype(int).tolist() for b in range(N_RISK_BANDS)
    }
    test_bands = np.digitize(p_test, edges)
    band_check = (
        pd.DataFrame({"band": test_bands, "late": test["late"].values, "delay": test["delay_days"].clip(lower=0).values})
        .groupby("band")
        .agg(shipments=("late", "size"), late_rate=("late", "mean"), mean_delay_days=("delay", "mean"))
        .round(3)
        .reset_index()
        .to_dict(orient="records")
    )

    joblib.dump(model, ARTIFACTS / "late_model.joblib")
    meta = {
        "model": chosen,
        "features": FEATURES,
        "categories": {c: sorted(df[c].cat.categories.tolist()) for c in CATEGORICAL},
        "risk_band_edges": edges,
        "delay_by_band": delay_by_band,
        "train_late_rate": round(float(base_rate), 4),
        "test_late_rate": round(float(test["late"].mean()), 4),
        "split": {"train": "2006-2012", "validation": "2013", "test": "2014-2015"},
        "n": {"train": len(train), "validation": len(val), "test": len(test)},
        "results": results,
        "test_band_check": band_check,
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))

    # Calibration plot for the README.
    frac_pos, mean_pred = calibration_curve(test["late"], p_test, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 0.4], [0, 0.4], ls="--", c="grey", lw=1)
    ax.plot(mean_pred, frac_pos, marker="o", c="#1f6f8b")
    ax.set(xlabel="Predicted probability of >7-day delay", ylabel="Observed share late",
           title="Calibration on 2014-2015 shipments")
    fig.tight_layout()
    fig.savefig(REPORTS / "calibration.png", dpi=150)

    print(json.dumps({"model": chosen, "n": meta["n"], "test_late_rate": meta["test_late_rate"], "results": results,
                      "band_check": band_check}, indent=2))


if __name__ == "__main__":
    main()
