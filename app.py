# ============================================================
# FINTECH TRUST & RETENTION INTELLIGENCE — STREAMLIT GUI
# ============================================================
# HOW TO RUN:
#   streamlit run app.py
#
# REQUIRED FILES:
#   models/churn_model.pkl
#   models/scaler.pkl
#   models/feature_names.pkl
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title = "FinTech Churn Intelligence",
    layout     = "wide"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #f8f9fb; }

    [data-testid="stSidebar"] { background-color: #0a1628; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }

    input[type="number"],
    input[type="text"] {
        background-color: #ffffff !important;
        color: #0a1628 !important;
        border: 1px solid #e8eaf0 !important;
        border-radius: 8px !important;
        caret-color: #0a1628 !important;
    }
    input::placeholder {
        color: #9ca3af !important;
        opacity: 1 !important;
    }
    [data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #0a1628 !important;
    }
    [data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #0a1628 !important;
    }
    input[type="text"]:focus,
    input[type="number"]:focus {
        border: 1.5px solid #0a1628 !important;
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(10,22,40,0.15) !important;
        caret-color: #0a1628 !important;
    }
    [data-testid="stNumberInput"] button {
        background-color: #e8eaf0 !important;
        color: #0a1628 !important;
        border-radius: 6px !important;
    }
    [data-testid="stTextInput"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #0a1628 !important;
        margin-bottom: 4px !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #e8eaf0 !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] * {
        background-color: #ffffff !important;
        color: #0a1628 !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0a1628 !important;
    }
    [data-baseweb="select"] span {
        background-color: #ffffff !important;
        color: #0a1628 !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within {
        border: 1.5px solid #0a1628 !important;
        box-shadow: 0 0 0 2px rgba(10,22,40,0.15) !important;
    }
    [data-baseweb="popover"] { background-color: #ffffff !important; }
    [data-baseweb="popover"] * {
        background-color: #ffffff !important;
        color: #0a1628 !important;
    }
    [data-baseweb="menu"] {
        background-color: #ffffff !important;
        border: 1px solid #e8eaf0 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="menu"] li {
        background-color: #ffffff !important;
        color: #0a1628 !important;
    }
    [data-baseweb="menu"] li:hover {
        background-color: #f1f5f9 !important;
        color: #0a1628 !important;
    }
    [data-testid="stTabs"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #e8eaf0;
        margin-bottom: 16px;
    }
    button[data-baseweb="tab"] {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #6b7280 !important;
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    button[data-baseweb="tab"]:hover {
        background-color: #f1f5f9 !important;
        color: #0a1628 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #0a1628 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    [data-baseweb="tab-highlight"] { display: none !important; }
    [data-baseweb="tab-border"]    { display: none !important; }

    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e8eaf0;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #0a1628;
    }
    .metric-label {
        font-size: 13px;
        color: #6b7280;
        margin-top: 4px;
    }
    .risk-critical {
        background-color: #fee2e2; color: #991b1b;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 14px; display: inline-block;
    }
    .risk-high {
        background-color: #ffedd5; color: #9a3412;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 14px; display: inline-block;
    }
    .risk-medium {
        background-color: #fef9c3; color: #854d0e;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 14px; display: inline-block;
    }
    .risk-low {
        background-color: #dcfce7; color: #166534;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 600; font-size: 14px; display: inline-block;
    }
    .section-header {
        font-size: 18px; font-weight: 600; color: #0a1628;
        margin-bottom: 16px; padding-bottom: 8px;
        border-bottom: 2px solid #e8eaf0;
    }
    .sub-label {
        font-size: 13px; font-weight: 600; color: #0a1628;
        margin-bottom: 8px; letter-spacing: 1px;
        text-transform: uppercase;
    }
    .stButton > button {
        background-color: #0a1628 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        height: 48px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border: none !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #1e3a5f !important;
        color: #ffffff !important;
    }
    hr { border-color: #e8eaf0 !important; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model         = joblib.load("models/churn_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# ============================================================
# FEATURE ENGINEERING — must match training notebook exactly
# ============================================================
def engineer_features(data: dict) -> dict:
    data['engagement_score'] = (
        data['Total_Trans_Ct'] * 0.4 +
        data['Total_Relationship_Count'] * 0.3 -
        data['Months_Inactive_12_mon'] * 0.3
    )
    data['txn_decline_flag']  = int(data['Total_Amt_Chng_Q4_Q1'] < 0.7)
    data['high_contact_flag'] = int(data['Contacts_Count_12_mon'] >= 3)
    data['revolving_ratio']   = (
        data['Total_Revolving_Bal'] / (data['Credit_Limit'] + 1)
    )
    return data

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px 0;'>
        <div style='font-size:36px; font-weight:800;
                    color:#4f9cf9; letter-spacing:2px;'>FTRI</div>
        <div style='font-size:11px; color:#94a3b8;
                    letter-spacing:3px; margin-top:4px;'>
            FINTECH INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='font-size:13px; color:#94a3b8; padding:0 8px;'>
        <p><b style='color:#ffffff;'>Model</b><br>XGBoost Classifier</p>
        <p><b style='color:#ffffff;'>Accuracy</b><br>96.8%</p>
        <p><b style='color:#ffffff;'>AUC Score</b><br>0.992</p>
        <p><b style='color:#ffffff;'>F1 Score</b><br>0.898</p>
        <p><b style='color:#ffffff;'>Training Rows</b><br>10,127</p>
        <p><b style='color:#ffffff;'>Features Used</b><br>24 features</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='font-size:11px; color:#64748b;
                text-align:center; padding-top:8px;'>
        FinTech Trust & Retention Intelligence<br>v1.0.0
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown("""
<div style='background-color:#0a1628; padding:28px 32px;
            border-radius:14px; margin-bottom:24px;'>
    <div style='font-size:26px; font-weight:700; color:#ffffff;'>
        FinTech Trust &amp; Retention Intelligence
    </div>
    <div style='font-size:14px; color:#94a3b8; margin-top:6px;'>
        Real-time customer churn prediction &nbsp;|&nbsp;
        XGBoost &nbsp;|&nbsp; AUC 0.992 &nbsp;|&nbsp; Accuracy 96.8%
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS — only 2 now
# ============================================================
tab1, tab2 = st.tabs(["Predict Churn", "Model Performance"])

# ============================================================
# TAB 1 — PREDICT
# ============================================================
with tab1:

    st.markdown('<div class="section-header">Customer Profile</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="sub-label">Identity</div>',
                    unsafe_allow_html=True)
        customer_id = st.text_input("Customer ID",  value="Customer Name")
        age         = st.number_input("Age",
                                       min_value=18, max_value=100, value=45)
        gender      = st.selectbox("Gender", ["Male", "Female"])
        dependents  = st.number_input("Dependents",
                                       min_value=0, max_value=10, value=2)

    with col2:
        st.markdown('<div class="sub-label">Profile</div>',
                    unsafe_allow_html=True)
        education = st.selectbox("Education Level",
                                  ["Uneducated", "High School", "College",
                                   "Graduate", "Post-Graduate", "Doctorate"])
        marital   = st.selectbox("Marital Status",
                                  ["Single", "Married", "Divorced"])
        income    = st.selectbox("Income Category",
                                  ["Less than $40K", "$40K - $60K",
                                   "$60K - $80K", "$80K - $120K", "$120K +"])
        card      = st.selectbox("Card Category",
                                  ["Blue", "Silver", "Gold", "Platinum"])

    with col3:
        st.markdown('<div class="sub-label">Relationship</div>',
                    unsafe_allow_html=True)
        months_book     = st.number_input("Months on Book",
                                           min_value=1, max_value=60, value=36)
        relationship_ct = st.number_input("Total Relationships",
                                           min_value=1, max_value=10, value=4)
        months_inactive = st.number_input("Months Inactive (12m)",
                                           min_value=0, max_value=12, value=2)
        contacts_count  = st.number_input("Contacts Count (12m)",
                                           min_value=0, max_value=10, value=3)

    st.divider()

    st.markdown('<div class="section-header">Financial Details</div>',
                unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown('<div class="sub-label">Credit</div>',
                    unsafe_allow_html=True)
        credit_limit  = st.number_input("Credit Limit",
                                         min_value=0.0,
                                         value=12000.0, step=100.0)
        revolving_bal = st.number_input("Revolving Balance",
                                         min_value=0.0,
                                         value=800.0, step=100.0)
        open_to_buy   = st.number_input("Avg Open to Buy",
                                         min_value=0.0,
                                         value=11200.0, step=100.0)
        util_ratio    = st.number_input("Utilization Ratio (0.0 - 1.0)",
                                         min_value=0.0, max_value=1.0,
                                         value=0.20, step=0.01)

    with col5:
        st.markdown('<div class="sub-label">Transactions</div>',
                    unsafe_allow_html=True)
        trans_amt = st.number_input("Transaction Amount",
                                     min_value=0.0,
                                     value=3500.0, step=100.0)
        trans_ct  = st.number_input("Transaction Count",
                                     min_value=0, value=38)
        amt_chng  = st.number_input("Amt Change Q4/Q1",
                                     min_value=0.0,
                                     value=0.75, step=0.01)
        ct_chng   = st.number_input("Count Change Q4/Q1",
                                     min_value=0.0,
                                     value=0.60, step=0.01)

    with col6:
        st.markdown('<div class="sub-label">Summary</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#f1f5f9; border-radius:10px;
                    padding:16px; border:1px solid #e8eaf0;'>
            <div style='font-size:12px; color:#6b7280;
                        margin-bottom:10px; font-weight:600;
                        letter-spacing:1px;'>INPUT SUMMARY</div>
            <table style='width:100%; font-size:13px;
                          border-collapse:collapse;'>
                <tr>
                    <td style='color:#6b7280; padding:5px 0;'>Customer</td>
                    <td style='color:#0a1628; font-weight:600;
                               text-align:right;'>{customer_id}</td>
                </tr>
                <tr>
                    <td style='color:#6b7280; padding:5px 0;'>Age</td>
                    <td style='color:#0a1628; font-weight:600;
                               text-align:right;'>{age}</td>
                </tr>
                <tr>
                    <td style='color:#6b7280; padding:5px 0;'>Gender</td>
                    <td style='color:#0a1628; font-weight:600;
                               text-align:right;'>{gender}</td>
                </tr>
                <tr>
                    <td style='color:#6b7280; padding:5px 0;'>Card</td>
                    <td style='color:#0a1628; font-weight:600;
                               text-align:right;'>{card}</td>
                </tr>
                <tr>
                    <td style='color:#6b7280; padding:5px 0;'>Income</td>
                    <td style='color:#0a1628; font-weight:600;
                               text-align:right;'>{income}</td>
                </tr>
                <tr>
                    <td style='color:#6b7280; padding:5px 0;'>Transactions</td>
                    <td style='color:#0a1628; font-weight:600;
                               text-align:right;'>{trans_ct}</td>
                </tr>
                <tr>
                    <td style='color:#6b7280; padding:5px 0;'>
                        Inactive months</td>
                    <td style='color:#0a1628; font-weight:600;
                               text-align:right;'>{months_inactive}</td>
                </tr>
                <tr>
                    <td style='color:#6b7280; padding:5px 0;'>Contacts</td>
                    <td style='color:#0a1628; font-weight:600;
                               text-align:right;'>{contacts_count}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Encode categoricals same as training ──
    gender_enc    = 0 if gender == "Female" else 1
    education_map = {"Uneducated":0, "High School":1, "College":2,
                     "Graduate":3, "Post-Graduate":4, "Doctorate":5}
    marital_map   = {"Divorced":0, "Married":1, "Single":2}
    income_map    = {"Less than $40K":0, "$40K - $60K":1,
                     "$60K - $80K":2, "$80K - $120K":3, "$120K +":4}
    card_map      = {"Blue":0, "Gold":1, "Platinum":2, "Silver":3}

    if util_ratio <= 0.2:
        util_bucket = 0
    elif util_ratio <= 0.5:
        util_bucket = 1
    elif util_ratio <= 0.8:
        util_bucket = 2
    else:
        util_bucket = 3

    if st.button("Run Prediction", type="primary", use_container_width=True):

        input_data = {
            'Customer_Age'             : age,
            'Gender'                   : gender_enc,
            'Dependent_count'          : dependents,
            'Education_Level'          : education_map[education],
            'Marital_Status'           : marital_map[marital],
            'Income_Category'          : income_map[income],
            'Card_Category'            : card_map[card],
            'Months_on_book'           : months_book,
            'Total_Relationship_Count' : relationship_ct,
            'Months_Inactive_12_mon'   : months_inactive,
            'Contacts_Count_12_mon'    : contacts_count,
            'Credit_Limit'             : credit_limit,
            'Total_Revolving_Bal'      : revolving_bal,
            'Avg_Open_To_Buy'          : open_to_buy,
            'Total_Amt_Chng_Q4_Q1'     : amt_chng,
            'Total_Trans_Amt'          : trans_amt,
            'Total_Trans_Ct'           : trans_ct,
            'Total_Ct_Chng_Q4_Q1'      : ct_chng,
            'Avg_Utilization_Ratio'    : util_ratio,
            'utilization_bucket'       : util_bucket
        }

        input_data = engineer_features(input_data)

        row        = pd.DataFrame([input_data])[feature_names]
        row_scaled = scaler.transform(row)
        prob       = float(model.predict_proba(row_scaled)[0][1])
        label      = int(prob >= 0.5)
        confidence = prob if label == 1 else 1 - prob

        if prob >= 0.75:
            risk_level = "CRITICAL"
            risk_class = "risk-critical"
            bar_color  = "#ef4444"
        elif prob >= 0.5:
            risk_level = "HIGH"
            risk_class = "risk-high"
            bar_color  = "#f97316"
        elif prob >= 0.3:
            risk_level = "MEDIUM"
            risk_class = "risk-medium"
            bar_color  = "#eab308"
        else:
            risk_level = "LOW"
            risk_class = "risk-low"
            bar_color  = "#22c55e"

        recommendations = {
            "CRITICAL" : "Immediate call + premium loyalty offer within 24hrs",
            "HIGH"     : "Email campaign with personalised retention offer",
            "MEDIUM"   : "Enroll in engagement program + satisfaction survey",
            "LOW"      : "No action needed — continue standard engagement"
        }

        st.divider()
        st.markdown('<div class="section-header">Prediction Results</div>',
                    unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{prob*100:.1f}%</div>
                <div class="metric-label">Churn Probability</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            result_color = "#ef4444" if label == 1 else "#22c55e"
            result_text  = "Will Churn" if label == 1 else "Will Stay"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value"
                     style='color:{result_color};'>{result_text}</div>
                <div class="metric-label">Prediction</div>
            </div>""", unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{confidence*100:.1f}%</div>
                <div class="metric-label">Confidence</div>
            </div>""", unsafe_allow_html=True)

        with r4:
            st.markdown(f"""
            <div class="metric-card">
                <span class="{risk_class}">{risk_level}</span>
                <div class="metric-label" style='margin-top:10px;'>
                    Risk Level</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        st.markdown("**Churn Probability Gauge**")
        st.markdown(f"""
        <div style='background:#e8eaf0; border-radius:8px;
                    height:22px; width:100%; margin-bottom:6px;'>
            <div style='background:{bar_color}; border-radius:8px;
                        height:22px; width:{prob*100:.1f}%;'></div>
        </div>
        <div style='font-size:13px; color:#6b7280; margin-bottom:16px;'>
            {prob*100:.1f}% probability of churn
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        rec_bg    = "#fee2e2" if label == 1 else "#dcfce7"
        rec_color = "#991b1b" if label == 1 else "#166534"
        verdict   = (f"Customer {customer_id} is at risk of churning"
                     if label == 1
                     else f"Customer {customer_id} is likely to stay")

        st.markdown(f"""
        <div style='background:{rec_bg}; border-radius:10px;
                    padding:16px 20px; margin-bottom:16px;'>
            <div style='font-size:15px; font-weight:600;
                        color:{rec_color};'>{verdict}</div>
            <div style='font-size:13px; color:{rec_color}; margin-top:6px;'>
                Recommendation: {recommendations[risk_level]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="section-header">Risk Signals Detected</div>',
                    unsafe_allow_html=True)

        signals = []
        if trans_ct < 40:
            signals.append(("Low transaction count (< 40)",
                             "Transactions dropped below healthy threshold"))
        if months_inactive >= 3:
            signals.append(("High inactivity (3+ months)",
                             "Customer has not engaged for 3 or more months"))
        if contacts_count >= 3:
            signals.append(("High contact frequency (3+ times)",
                             "Repeated contacts may indicate dissatisfaction"))
        if amt_chng < 0.7:
            signals.append(("Declining spend (Q4 vs Q1 drop)",
                             "Transaction amount dropped significantly"))

        if signals:
            for title, desc in signals:
                st.markdown(f"""
                <div style='background:#fff7ed;
                            border-left:4px solid #f97316;
                            border-radius:6px; padding:12px 16px;
                            margin-bottom:10px;'>
                    <div style='font-size:14px; font-weight:600;
                                color:#9a3412;'>{title}</div>
                    <div style='font-size:13px; color:#c2410c;
                                margin-top:4px;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#f0fdf4;
                        border-left:4px solid #22c55e;
                        border-radius:6px; padding:12px 16px;'>
                <div style='font-size:14px; font-weight:600;
                            color:#166534;'>
                    No major risk signals detected
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 2 — MODEL PERFORMANCE
# ============================================================
with tab2:

    st.markdown('<div class="section-header">Model Performance Summary</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)

    for col, lbl, val in zip(
        [m1, m2, m3, m4, m5],
        ["Model", "Accuracy", "AUC Score", "F1 Score", "Recall"],
        ["XGBoost", "96.8%", "0.992", "0.898", "0.883"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-header">All Models Compared</div>',
                unsafe_allow_html=True)

    comparison = pd.DataFrame({
        'Model'     : ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy'  : ['82.1%', '94.3%', '96.8%'],
        'F1 Score'  : [0.634, 0.853, 0.898],
        'AUC'       : [0.918, 0.984, 0.992],
        'Precision' : [0.525, 0.857, 0.914],
        'Recall'    : [0.800, 0.849, 0.883]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown('<div class="section-header">Datasets Used</div>',
                unsafe_allow_html=True)

    datasets = pd.DataFrame({
        'Dataset'  : ['BankChurners.csv', 'Churn_Modelling.csv',
                      'german_credit_data.csv'],
        'Rows'     : [10127, 10000, 1000],
        'Purpose'  : ['Primary churn model', 'Supporting analysis',
                      'Credit risk analysis'],
        'Target'   : ['Attrition_Flag', 'Exited', 'Engineered risk_flag']
    })
    st.dataframe(datasets, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown('<div class="section-header">Techniques Applied</div>',
                unsafe_allow_html=True)

    t1, t2 = st.columns(2)

    with t1:
        for technique, desc in [
            ("SMOTE",               "Handled 84/16 class imbalance"),
            ("Stratified K-Fold CV","5 folds, no data leakage"),
            ("GridSearchCV",        "Hyperparameter tuning for XGBoost"),
        ]:
            st.markdown(f"""
            <div style='background:#ffffff; border:1px solid #e8eaf0;
                        border-radius:8px; padding:12px 16px;
                        margin-bottom:10px;'>
                <div style='font-size:14px; font-weight:600;
                            color:#0a1628;'>{technique}</div>
                <div style='font-size:13px; color:#6b7280;
                            margin-top:4px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with t2:
        for technique, desc in [
            ("SHAP Explainability", "Feature importance for all 3 models"),
            ("Feature Engineering", "5 new features created from raw data"),
            ("Hypothesis Validation","5 business hypotheses all confirmed"),
        ]:
            st.markdown(f"""
            <div style='background:#ffffff; border:1px solid #e8eaf0;
                        border-radius:8px; padding:12px 16px;
                        margin-bottom:10px;'>
                <div style='font-size:14px; font-weight:600;
                            color:#0a1628;'>{technique}</div>
                <div style='font-size:13px; color:#6b7280;
                            margin-top:4px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
