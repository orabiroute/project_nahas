"""
Loan Approval — Prediction App
Loads the trained pipeline from the notebook (loan_pipeline.pkl)
and predicts. No preprocessing code here — the pipeline handles it.

Run:  streamlit run app.py
Need: loan_pipeline.pkl  (run the notebook first to generate it)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Loan Prediction", page_icon="🏦", layout="centered")
st.title("🏦 Loan Approval Predictor")
st.caption("Powered by the pipeline trained in the notebook.")

# ── Load pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load("loan_pipeline.pkl")

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("❌ `loan_pipeline.pkl` not found.\n\nRun the notebook first to train and save the model.")
    st.stop()

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal**")
    age       = st.number_input("Age",                 18, 100,  28)
    income    = st.number_input("Annual Income ($)",  5000, 2_000_000, 60_000, step=1000)
    emp_exp   = st.number_input("Employment Exp (yrs)", 0, 60, 3)
    gender    = st.selectbox("Gender", ["male", "female"])
    education = st.selectbox("Education",
                             ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    home      = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

with col2:
    st.markdown("**Loan**")
    loan_amt = st.number_input("Loan Amount ($)",  500, 50_000, 10_000, step=500)
    int_rate = st.slider("Interest Rate (%)",      1.0, 25.0, 11.0, 0.1)
    pct_inc  = st.slider("Loan % of Income",       0.01, 1.0, 0.20, 0.01)
    intent   = st.selectbox("Loan Intent",
                            ["PERSONAL", "EDUCATION", "MEDICAL",
                             "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

with col3:
    st.markdown("**Credit**")
    cred_hist    = st.number_input("Credit History (yrs)", 1, 30, 5)
    credit_score = st.number_input("Credit Score",        300, 850, 650)
    prev_default = st.selectbox("Previous Default", ["No", "Yes"])

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("⚡ Predict", type="primary", use_container_width=True):

    # Build the same DataFrame the pipeline expects
    row = pd.DataFrame([{
        "person_age":                     age,
        "person_income":                  income,
        "person_emp_exp":                 emp_exp,
        "loan_amnt":                      loan_amt,
        "loan_int_rate":                  int_rate,
        "loan_percent_income":            pct_inc,
        "cb_person_cred_hist_length":     cred_hist,
        "credit_score":                   credit_score,
        "person_gender":                  gender,
        "person_education":               education,
        "person_home_ownership":          home,
        "loan_intent":                    intent,
        "previous_loan_defaults_on_file": prev_default,
    }])

    # Pipeline does all preprocessing internally
    pred  = pipeline.predict(row)[0]
    proba = pipeline.predict_proba(row)[0]

    if pred == 1:
        st.success(f"✅ **APPROVED**\n\nConfidence: {proba[1]*100:.1f}%")
    else:
        st.error(f"❌ **REJECTED**\n\nConfidence: {proba[0]*100:.1f}%")

