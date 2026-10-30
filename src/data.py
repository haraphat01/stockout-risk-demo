"""Load and clean the USAID SCMS Delivery History dataset.

Only columns that are known when a shipment is *scheduled* are kept as features,
so the model can be used before the shipment arrives (no leakage from actual
delivery dates, freight invoices, etc.).
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "scms_delivery_history.csv"

LATE_THRESHOLD_DAYS = 7  # "materially late": more than a week past schedule
RECENT_WINDOW_DAYS = 180

CATEGORICAL = [
    "country",
    "shipment_mode",
    "fulfill_via",
    "inco_term",
    "product_group",
    "sub_classification",
    "vendor",
    "dosage_form",
    "first_line",
]
NUMERIC = [
    "log_qty",
    "log_value",
    "log_pack_price",
    "planned_lead_days",
    "po_lead_days",
    "sched_month",
    "country_recent_late",
    "global_recent_late",
]
FEATURES = CATEGORICAL + NUMERIC


def _date(s: pd.Series, fmt: str) -> pd.Series:
    return pd.to_datetime(s, format=fmt, errors="coerce")


def load(path: Path = RAW) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")

    sched = _date(raw["Scheduled Delivery Date"], "%d-%b-%y")
    delivered = _date(raw["Delivered to Client Date"], "%d-%b-%y")
    pq_sent = _date(raw["PQ First Sent to Client Date"], "%m/%d/%y")
    po_sent = _date(raw["PO Sent to Vendor Date"], "%m/%d/%y")

    # Collapse the long tail of vendors so rare ones don't become noise.
    top_vendors = raw["Vendor"].value_counts().head(15).index
    vendor = raw["Vendor"].where(raw["Vendor"].isin(top_vendors), "Other")

    df = pd.DataFrame(
        {
            "id": raw["ID"],
            "asn": raw["ASN/DN #"],
            "scheduled": sched,
            "delivered": delivered,
            "country": raw["Country"],
            "shipment_mode": raw["Shipment Mode"].fillna("Unknown"),
            "fulfill_via": raw["Fulfill Via"],
            "inco_term": raw["Vendor INCO Term"],
            "product_group": raw["Product Group"],
            "sub_classification": raw["Sub Classification"],
            "vendor": vendor,
            "dosage_form": raw["Dosage Form"],
            "first_line": raw["First Line Designation"],
            "item": raw["Item Description"],
            "log_qty": np.log1p(raw["Line Item Quantity"].clip(lower=0)),
            "log_value": np.log1p(raw["Line Item Value"].clip(lower=0)),
            "log_pack_price": np.log1p(raw["Pack Price"].clip(lower=0)),
            "planned_lead_days": (sched - pq_sent).dt.days,
            "po_lead_days": (sched - po_sent).dt.days,
            "sched_month": sched.dt.month,
        }
    )
    df["delay_days"] = (delivered - sched).dt.days
    df["late"] = (df["delay_days"] > LATE_THRESHOLD_DAYS).astype(int)
    df = df.dropna(subset=["scheduled", "delivered"]).reset_index(drop=True)
    df["country_recent_late"] = _recent_late_rate(df, by="country")
    df["global_recent_late"] = _recent_late_rate(df, by=None)
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).astype("category")
    return df


def _recent_late_rate(df: pd.DataFrame, by: str | None, window_days: int = RECENT_WINDOW_DAYS) -> pd.Series:
    """Late share among shipments *already delivered* in the `window_days` before each scheduled date.

    Uses only information a planner would have on the scheduling date, so it is leakage-free.
    Returns NaN when there is no recent history (the model handles missing values).
    """
    out = pd.Series(np.nan, index=df.index)
    groups = df.groupby(by, observed=True) if by else [(None, df)]
    for _, g in groups:
        done = g.sort_values("delivered")
        d_dates = done["delivered"].values
        cum_late = np.concatenate([[0], np.cumsum(done["late"].values)])
        for idx, s in g["scheduled"].items():
            hi = np.searchsorted(d_dates, np.datetime64(s), side="left")
            lo = np.searchsorted(d_dates, np.datetime64(s - pd.Timedelta(days=window_days)), side="left")
            if hi - lo >= 5:
                out[idx] = (cum_late[hi] - cum_late[lo]) / (hi - lo)
    return out


def time_split(df: pd.DataFrame):
    """Train on 2006-2012, validate on 2013, test on 2014-2015 (forward in time)."""
    yr = df["scheduled"].dt.year
    return df[yr <= 2012], df[yr == 2013], df[yr >= 2014]
