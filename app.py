import streamlit as st
import pandas as pd
import joblib

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6a5cff, #ff5c93);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #9aa0ac;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .section-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.2rem 1.4rem 0.6rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .result-approved {
        background: linear-gradient(120deg, #14532d, #166534);
        border: 1px solid #22c55e;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        color: white;
    }
    .result-denied {
        background: linear-gradient(120deg, #4c1d1d, #7f1d1d);
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        color: white;
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6a5cff, #ff5c93);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.4rem;
        width: 100%;
        transition: transform 0.15s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load saved artifacts
# ============================================================
@st.cache_resource
def load_artifacts():
    return {
        "model": joblib.load("loan_model.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "le": joblib.load("label_encoder.pkl"),
        "ohe": joblib.load("onehot_encoder.pkl"),
        "onehot_input_cols": joblib.load("onehot_input_cols.pkl"),
        "final_feature_order": joblib.load("final_feature_order.pkl"),
    }

art = load_artifacts()

# ============================================================
# Header
# ============================================================
st.markdown('<p class="main-header">💰 Loan Approval Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Fill in the applicant\'s details below and get an instant prediction.</p>', unsafe_allow_html=True)

# ============================================================
# Input form — grouped into sections with tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["👤 Personal", "💵 Financial", "🏠 Loan Details"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c3:
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=15000.0, step=500.0)
        credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=650)
        savings = st.number_input("Savings", min_value=0.0, value=10000.0, step=500.0)
    with c2:
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=500.0)
        dti_ratio = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=1)
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    employer_category = st.selectbox("Employer Category", ["Government", "MNC", "Private", "Unemployed"])

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=100000.0, step=1000.0)
        loan_purpose = st.selectbox("Loan Purpose", ["Car", "Education", "Home", "Personal"])
    with c2:
        collateral_value = st.number_input("Collateral Value", min_value=0.0, value=20000.0, step=1000.0)
        property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

st.write("")
predict_clicked = st.button("🔮 Predict Loan Approval", use_container_width=True)

# ============================================================
# Prediction
# ============================================================
if predict_clicked:
    raw = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Loan_Purpose": loan_purpose,
        "Gender": gender,
        "Employer_Category": employer_category,
        "Property_Area": property_area,
    }])

    # Apply the SAME preprocessing as training
    # (Education_Level is skipped here — see note above; it'll be filled
    # with 0 automatically at the reindex step below)

    encoded = art["ohe"].transform(raw[art["onehot_input_cols"]])
    encoded_df = pd.DataFrame(encoded, columns=art["ohe"].get_feature_names_out(art["onehot_input_cols"]))
    raw = pd.concat([
        raw.drop(columns=art["onehot_input_cols"]).reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ], axis=1)

    raw = raw.reindex(columns=art["final_feature_order"], fill_value=0)
    scaled = art["scaler"].transform(raw)

    pred = art["model"].predict(scaled)[0]
    proba = art["model"].predict_proba(scaled)[0] if hasattr(art["model"], "predict_proba") else None
    confidence = max(proba) * 100 if proba is not None else None

    st.write("")
    if pred == 1:
        st.markdown(f"""
        <div class="result-approved">
            <div class="result-title">✅ Loan Approved</div>
            <div>This applicant is likely to be approved.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-denied">
            <div class="result-title">❌ Loan Not Approved</div>
            <div>This applicant is unlikely to be approved based on current inputs.</div>
        </div>
        """, unsafe_allow_html=True)

    if confidence is not None:
        st.write("")
        st.caption(f"Model confidence: {confidence:.1f}%")
        st.progress(int(confidence))

    with st.expander("🔍 See what was sent to the model"):
        st.dataframe(raw)