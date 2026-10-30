import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
BENCHMARK_PATH = ROOT / "outputs" / "benchmark.csv"
AUDIT_PATH = ROOT / "outputs" / "data_audit.json"
SUBGROUP_PATH = ROOT / "outputs" / "subgroup_audit.csv"

st.set_page_config(page_title="Trust and Retention Intelligence", layout="wide")
st.title("Trust and Retention Intelligence")
st.markdown("[Open the live app](https://trust-and-retention-intelligence.streamlit.app/)")
st.caption(
    "Public-data research demo. The source has no observation date or future churn "
    "horizon, so scores are not validated predictions for a real bank."
)


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_table(path):
    return pd.read_csv(path)


if not MODEL_PATH.exists():
    st.error(
        "Model artifact is missing. Run all cells in Model.ipynb from the project folder first."
    )
    st.stop()

artifact = load_artifact()
model = artifact["model"]
threshold = float(artifact["threshold"])

predict_tab, benchmark_tab = st.tabs(["Score a benchmark profile", "Benchmark results"])

with predict_tab:
    st.subheader("Account and activity details")
    with st.form("profile"):
        left, middle, right = st.columns(3)
        with left:
            card = st.selectbox(
                "Card category", ["Blue", "Silver", "Gold", "Platinum"]
            )
            months_on_book = st.number_input(
                "Months on book", min_value=13, max_value=56, value=36
            )
            relationships = st.number_input(
                "Total relationships", min_value=1, max_value=6, value=4
            )
            inactive_months = st.number_input(
                "Months inactive in the last year", min_value=0, max_value=6, value=2
            )
        with middle:
            contacts = st.number_input(
                "Contacts in the last year", min_value=0, max_value=6, value=2
            )
            credit_limit = st.number_input(
                "Credit limit",
                min_value=1438.3,
                max_value=34516.0,
                value=4549.0,
                step=100.0,
            )
            revolving_balance = st.number_input(
                "Revolving balance",
                min_value=0.0,
                max_value=2517.0,
                value=1276.0,
                step=100.0,
            )
            amount_change = st.number_input(
                "Transaction amount change (Q4/Q1)",
                min_value=0.0,
                max_value=3.397,
                value=0.736,
                step=0.01,
            )
        with right:
            transaction_amount = st.number_input(
                "Transaction amount",
                min_value=510,
                max_value=18484,
                value=3899,
                step=100,
            )
            transaction_count = st.number_input(
                "Transaction count", min_value=10, max_value=139, value=67
            )
            count_change = st.number_input(
                "Transaction count change (Q4/Q1)",
                min_value=0.0,
                max_value=3.714,
                value=0.702,
                step=0.01,
            )
            open_to_buy = credit_limit - revolving_balance
            utilization = round(revolving_balance / credit_limit, 3)
            st.metric("Average open to buy", f"{open_to_buy:,.0f}")
            st.metric("Utilization", f"{utilization:.1%}")

        submitted = st.form_submit_button("Score profile", type="primary")

    if submitted:
        if revolving_balance > credit_limit or open_to_buy < 3:
            st.error(
                "Revolving balance must be below the credit limit, leaving at least "
                "3 units open to buy."
            )
        else:
            row = pd.DataFrame(
                [
                    {
                        "Months_on_book": months_on_book,
                        "Total_Relationship_Count": relationships,
                        "Months_Inactive_12_mon": inactive_months,
                        "Contacts_Count_12_mon": contacts,
                        "Credit_Limit": credit_limit,
                        "Total_Revolving_Bal": revolving_balance,
                        "Avg_Open_To_Buy": open_to_buy,
                        "Total_Amt_Chng_Q4_Q1": amount_change,
                        "Total_Trans_Amt": transaction_amount,
                        "Total_Trans_Ct": transaction_count,
                        "Total_Ct_Chng_Q4_Q1": count_change,
                        "Avg_Utilization_Ratio": utilization,
                        "Card_Category": card,
                    }
                ],
                columns=artifact["feature_columns"],
            )
            score = float(model.predict_proba(row)[0, 1])
            cols = st.columns(3)
            cols[0].metric("Attrited-label score", f"{score:.1%}")
            cols[1].metric(
                "Benchmark classification",
                "Attrited" if score >= threshold else "Existing",
            )
            cols[2].metric("Validation threshold", f"{threshold:.3f}")
            st.info(
                "This score estimates the public dataset label only. It has no "
                "future prediction horizon and should not be used to contact "
                "customers or make account decisions."
            )

with benchmark_tab:
    st.subheader("Locked holdout results")
    metrics = artifact["metrics"]
    kpis = st.columns(4)
    kpis[0].metric("Selected model", artifact["selected_model"])
    kpis[1].metric("Average precision", f"{metrics['holdout_average_precision']:.3f}")
    kpis[2].metric("ROC-AUC", f"{metrics['holdout_roc_auc']:.3f}")
    kpis[3].metric("F1 at validation threshold", f"{metrics['holdout_f1']:.3f}")
    st.caption(
        f"Threshold {threshold:.3f} was selected on validation data to maximize F1. "
        f"Holdout average precision 95% bootstrap interval: "
        f"{metrics['holdout_average_precision_95pct_bootstrap_ci'][0]:.3f}–"
        f"{metrics['holdout_average_precision_95pct_bootstrap_ci'][1]:.3f}."
    )

    if BENCHMARK_PATH.exists():
        benchmark = load_table(BENCHMARK_PATH)
        st.bar_chart(
            benchmark.set_index("model")["holdout_average_precision"],
            y_label="Average precision",
            alt="Holdout average precision by model",
        )
        st.dataframe(
            benchmark,
            hide_index=True,
            alt="Cross-validation and locked holdout model comparison",
        )

    st.subheader("Data and subgroup checks")
    if AUDIT_PATH.exists():
        with AUDIT_PATH.open(encoding="utf-8") as file:
            data_audit = json.load(file)
        st.write(
            f"Rows after cleaning: {data_audit['rows_after_cleaning']:,}; "
            f"missing predictor values: "
            f"{data_audit['missing_predictor_values_after_cleaning']:,}; "
            f"duplicate rows removed: {data_audit['duplicate_rows_removed']:,}; "
            f"source attrition rate: {data_audit['attrition_rate']:.1%}; "
            f"holdout rows: {metrics['holdout_rows']:,}."
        )
    if SUBGROUP_PATH.exists():
        st.dataframe(
            load_table(SUBGROUP_PATH),
            hide_index=True,
            alt="Holdout performance by gender and age audit groups",
        )

    st.warning(
        "These are public-benchmark results. A production threshold needs bank-specific "
        "costs, outreach capacity, dated data, and a controlled retention pilot."
    )
