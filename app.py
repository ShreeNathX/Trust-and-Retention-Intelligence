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
st.caption(
    "Public-data research demo. The source has no observation date or future churn "
    "horizon, so scores are not validated predictions for a real bank."
)

if not AUDIT_PATH.exists():
    st.error("Training input ranges are missing. Run all cells in Model.ipynb first.")
    st.stop()

with AUDIT_PATH.open(encoding="utf-8") as file:
    data_audit = json.load(file)

training_ranges = data_audit.get("training_input_ranges", {})
training_categories = data_audit.get("training_categories", {})
if not training_ranges or not training_categories.get("Card_Category"):
    st.error("Training input metadata is incomplete. Run all cells in Model.ipynb first.")
    st.stop()


def numeric_input(label, feature, step=1, integer=False):
    stats = training_ranges[feature]
    if integer:
        minimum, maximum = int(stats["min"]), int(stats["max"])
        value = int(round(stats["median"]))
    else:
        minimum, maximum = float(stats["min"]), float(stats["max"])
        value = float(stats["median"])
    value = min(max(value, minimum), maximum)
    return st.number_input(
        label, min_value=minimum, max_value=maximum, value=value, step=step
    )


def file_signature(path):
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


@st.cache_resource(max_entries=2)
def load_artifact(path, modified_ns, size_bytes):
    return joblib.load(path)


@st.cache_data(max_entries=20)
def load_table(path, modified_ns, size_bytes):
    return pd.read_csv(path)


if not MODEL_PATH.exists():
    st.error(
        "Model artifact is missing. Run all cells in Model.ipynb from the project folder first."
    )
    st.stop()

artifact = load_artifact(*file_signature(MODEL_PATH))
missing_metadata = (
    set(artifact["feature_columns"])
    - set(training_ranges)
    - set(training_categories)
)
if missing_metadata:
    st.error("Training input metadata is missing model fields. Run all cells in Model.ipynb.")
    st.stop()

model = artifact["model"]
threshold = float(artifact["threshold"])

predict_tab, benchmark_tab = st.tabs(["Score a benchmark profile", "Benchmark results"])

with predict_tab:
    st.subheader("Account and activity details")
    with st.form("profile"):
        st.caption("Input ranges and default values come from the notebook training split.")
        left, middle, right = st.columns(3)
        with left:
            card = st.selectbox("Card category", training_categories["Card_Category"])
            months_on_book = numeric_input(
                "Months on book", "Months_on_book", integer=True
            )
            relationships = numeric_input(
                "Total relationships", "Total_Relationship_Count", integer=True
            )
            inactive_months = numeric_input(
                "Months inactive in the last year", "Months_Inactive_12_mon",
                integer=True,
            )
        with middle:
            contacts = numeric_input(
                "Contacts in the last year", "Contacts_Count_12_mon", integer=True
            )
            credit_limit = numeric_input("Credit limit", "Credit_Limit", step=100.0)
            revolving_balance = numeric_input(
                "Revolving balance", "Total_Revolving_Bal", step=100.0
            )
            amount_change = numeric_input(
                "Transaction amount change (Q4/Q1)",
                "Total_Amt_Chng_Q4_Q1", step=0.01,
            )
        with right:
            transaction_amount = numeric_input(
                "Transaction amount", "Total_Trans_Amt", step=100, integer=True
            )
            transaction_count = numeric_input(
                "Transaction count", "Total_Trans_Ct", integer=True
            )
            count_change = numeric_input(
                "Transaction count change (Q4/Q1)",
                "Total_Ct_Chng_Q4_Q1", step=0.01,
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
        elif any(
            not training_ranges[feature]["min"] <= value <= training_ranges[feature]["max"]
            for feature, value in {
                "Avg_Open_To_Buy": open_to_buy,
                "Avg_Utilization_Ratio": utilization,
            }.items()
        ):
            st.error(
                "The calculated open-to-buy or utilization value is outside the "
                "notebook training range, so this profile cannot be scored."
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
        benchmark = load_table(*file_signature(BENCHMARK_PATH))
        st.bar_chart(
            benchmark.set_index("model")["cv_average_precision_mean"],
            y_label="Cross-validated average precision",
            alt="Mean five-fold cross-validated average precision by model",
        )
        st.caption(
            "Candidates are compared on training cross-validation. The locked holdout "
            "above is evaluated once for the selected calibrated model."
        )
        st.dataframe(
            benchmark,
            hide_index=True,
            alt="Cross-validation and locked holdout model comparison",
        )

    st.subheader("Data and subgroup checks")
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
            load_table(*file_signature(SUBGROUP_PATH)),
            hide_index=True,
            alt="Holdout performance by gender and age audit groups",
        )

    st.warning(
        "These are public-benchmark results. A production threshold needs bank-specific "
        "costs, outreach capacity, dated data, and a controlled retention pilot."
    )
