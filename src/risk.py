"""Stock-out risk for a facility waiting on a resupply shipment.

Combines two uncertainties with Monte Carlo simulation:
  1. When will the shipment really arrive?  -> sampled from the empirical delays of
     past shipments in the same model risk band (real SCMS data).
  2. How much will the facility consume until then? -> gamma-distributed demand
     built from the facility's own average monthly consumption and its variability.
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
BAND_LABELS = ["Low", "Moderate", "Elevated", "High"]


@lru_cache
def load_artifacts():
    model = joblib.load(ARTIFACTS / "late_model.joblib")
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    return model, meta


def late_probability(shipment: dict) -> tuple[float, int]:
    """Probability the shipment is >7 days late, and its risk band (0 = lowest)."""
    model, meta = load_artifacts()
    row = pd.DataFrame([{f: shipment.get(f, np.nan) for f in meta["features"]}])
    for c, cats in meta["categories"].items():
        row[c] = pd.Categorical(row[c].astype(str), categories=cats)
    p = float(model.predict_proba(row)[:, 1][0])
    return p, int(np.digitize(p, meta["risk_band_edges"]))


@dataclass
class StockoutResult:
    p_stockout: float
    expected_gap_units: float  # mean shortfall when a stock-out happens
    days_of_cover: float
    p90_arrival_day: float
    top_up_units_p90: int  # extra stock needed to stay covered in 90% of scenarios


def simulate(
    stock_on_hand: float,
    monthly_consumption: float,
    consumption_cv: float,
    days_to_scheduled_arrival: int,
    risk_band: int,
    n: int = 20_000,
    seed: int = 0,
) -> StockoutResult:
    _, meta = load_artifacts()
    rng = np.random.default_rng(seed)
    delays = np.asarray(meta["delay_by_band"][str(risk_band)])
    arrival = days_to_scheduled_arrival + rng.choice(delays, size=n)

    # Gamma demand over the waiting period; variance grows linearly with time.
    mean = monthly_consumption * arrival / 30
    var = (consumption_cv * monthly_consumption) ** 2 * arrival / 30
    shape = np.where(var > 0, mean**2 / np.maximum(var, 1e-9), 1e9)
    demand = rng.gamma(shape, np.where(shape > 0, mean / shape, 0))

    gap = demand - stock_on_hand
    out = gap > 0
    return StockoutResult(
        p_stockout=float(out.mean()),
        expected_gap_units=float(gap[out].mean()) if out.any() else 0.0,
        days_of_cover=float(stock_on_hand / max(monthly_consumption, 1e-9) * 30),
        p90_arrival_day=float(np.quantile(arrival, 0.9)),
        top_up_units_p90=int(max(0, np.ceil(np.quantile(gap, 0.9)))),
    )
