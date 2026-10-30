"""Stock-out early warning: will this facility run out before its resupply lands?

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from src.briefing import briefing
from src.risk import BAND_LABELS, ARTIFACTS, late_probability, load_artifacts, simulate

st.set_page_config(page_title="Stock-out early warning", layout="wide")
_, meta = load_artifacts()
cats = meta["categories"]


def pick(label, field, default):
    opts = cats[field]
    return st.selectbox(label, opts, index=opts.index(default) if default in opts else 0)


st.title("Stock-out early warning")
st.caption(
    "Estimates the chance a health facility runs out of an essential medicine before its next shipment arrives. "
    "Shipment delay risk is learned from 10,324 real USAID SCMS deliveries (2006-2015)."
)

tab_tool, tab_model = st.tabs(["Risk check", "How good is the model?"])

with tab_tool:
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.subheader("1. Incoming shipment")
        shipment = {
            "country": pick("Destination country", "country", "Nigeria"),
            "shipment_mode": pick("Shipment mode", "shipment_mode", "Truck"),
            "product_group": pick("Product group", "product_group", "ARV"),
            "sub_classification": pick("Sub-classification", "sub_classification", "Adult"),
            "fulfill_via": pick("Fulfilled via", "fulfill_via", "From RDC"),
            "inco_term": pick("Incoterm", "inco_term", "N/A - From RDC"),
            "vendor": pick("Vendor", "vendor", "SCMS from RDC"),
            "dosage_form": pick("Dosage form", "dosage_form", "Tablet - FDC"),
            "first_line": pick("First-line regimen?", "first_line", "Yes"),
        }
        qty = st.number_input("Packs ordered", 1, 1_000_000, 10_000, step=500)
        pack_price = st.number_input("Price per pack (USD)", 0.0, 1_000.0, 20.0)
        days_to_arrival = st.slider("Days until scheduled arrival", 0, 120, 30)
        with st.expander("Optional planning details"):
            lead = st.number_input("Days between quote and scheduled delivery (0 = unknown)", 0, 1000, 0)
        shipment |= {
            "log_qty": np.log1p(qty),
            "log_value": np.log1p(qty * pack_price),
            "log_pack_price": np.log1p(pack_price),
            "planned_lead_days": lead or np.nan,
            "sched_month": pd.Timestamp.today().month,
        }

        st.subheader("2. Facility stock")
        soh = st.number_input("Stock on hand (packs)", 0, 1_000_000, 1_200, step=50)
        amc = st.number_input("Average monthly consumption (packs)", 1, 1_000_000, 800, step=50)
        cv = st.slider("Month-to-month consumption variability", 0.0, 1.0, 0.3,
                       help="Coefficient of variation. 0.3 means a typical month is within about 30% of average.")

    with right:
        p_late, band = late_probability(shipment)
        res = simulate(soh, amc, cv, days_to_arrival, band)

        st.subheader("Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Stock-out risk before resupply", f"{res.p_stockout:.0%}")
        c2.metric("Shipment >7 days late", f"{p_late:.0%}", f"{BAND_LABELS[band]} band", delta_color="off")
        c3.metric("Days of stock left", f"{res.days_of_cover:.0f}")

        # Sensitivity: how risk changes if the shipment is scheduled sooner or later.
        days = np.arange(0, 121, 5)
        curve = pd.DataFrame(
            {"Days until scheduled arrival": days,
             "Stock-out risk": [simulate(soh, amc, cv, int(d), band, n=4000).p_stockout for d in days]}
        ).set_index("Days until scheduled arrival")
        st.line_chart(curve, height=220)
        st.caption("How the risk would change if the shipment were scheduled earlier or later.")

        ctx = {
            "stock-out probability": round(res.p_stockout, 3),
            "days of cover": round(res.days_of_cover, 1),
            "days to scheduled arrival": days_to_arrival,
            "delay risk band": BAND_LABELS[band],
            "probability >7 days late": round(p_late, 3),
            "90th percentile arrival day": round(res.p90_arrival_day, 1),
            "top-up for 90% cover (units)": res.top_up_units_p90,
        }
        st.subheader("Briefing")
        if st.button("Write briefing"):
            with st.spinner("Writing..."):
                text, source = briefing(ctx)
            st.markdown(text)
            st.caption("Written by Claude from the numbers above." if source == "claude"
                       else "Template briefing (set ANTHROPIC_API_KEY to have Claude write it).")

with tab_model:
    st.subheader("Evaluation on unseen 2014-2015 shipments")
    st.write(
        f"Trained on {meta['n']['train']:,} shipments (2006-2012), model chosen on {meta['n']['validation']:,} "
        f"(2013), tested on {meta['n']['test']:,} (2014-2015). "
        f"{meta['test_late_rate']:.1%} of test shipments were more than 7 days late. Chosen model: **{meta['model']}**."
    )
    table = pd.DataFrame(meta["results"]).T.rename(columns={
        "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier (lower is better)",
        "recall_top10": "Late shipments caught in riskiest 10%", "val_roc_auc": "Validation ROC-AUC"})
    st.dataframe(table, width="stretch")
    st.subheader("Do the risk bands mean something?")
    bands = pd.DataFrame(meta["test_band_check"]).assign(band=lambda d: d["band"].map(dict(enumerate(BAND_LABELS))))
    st.dataframe(bands.rename(columns={"band": "Risk band", "shipments": "Shipments", "late_rate": "Share >7 days late",
                                       "mean_delay_days": "Mean delay (days)"}), hide_index=True)
    st.image(str(ARTIFACTS.parent / "reports" / "calibration.png"), width=420)
    st.info(
        "Limits: delays are only moderately predictable from order details (ROC-AUC about 0.7), and the data "
        "covers HIV commodities from 2006-2015. Consumption is entered by the user, not forecast. Use this to "
        "decide which facilities to check first, not as a guarantee."
    )
