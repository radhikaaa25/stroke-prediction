"""
app.py - Streamlit front-end for "An Ensemble Approach for Stroke Prediction".

Loads the model, scaler and metadata saved by `train.py`, collects patient
details, applies the SAME preprocessing used in training (via preprocessing.py),
and displays the predicted stroke-risk probability.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import encode_features

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "stroke_ensemble_model.joblib"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
META_PATH = ARTIFACT_DIR / "metadata.joblib"

MODEL_LABELS = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lr": "Logistic Regression",
}

st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* App Background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    color: #f1f5f9;
}

/* Header Text */
h1, h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* Glassmorphism containers */
[data-testid="stForm"], [data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    padding: 1.5rem !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

[data-testid="stForm"]:hover, [data-testid="stExpander"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4) !important;
}

/* Input fields styling */
.stTextInput>div>div>input, [data-testid="stSelectbox"] > div, .stNumberInput>div>div>input {
    background-color: rgba(255, 255, 255, 0.05) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
}

/* Labels */
label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

/* Primary Button */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #ff69b4 0%, #ec4899 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 2rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

[data-testid="stFormSubmitButton"] > button:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.5) !important;
}

/* Metrics and Results */
[data-testid="stMetricValue"] {
    font-size: 3rem !important;
    font-weight: 700 !important;
    background: -webkit-linear-gradient(45deg, #ff69b4, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Custom progress bar */
.stProgress > div > div > div > div {
    background-image: linear-gradient(90deg, #ff69b4, #ec4899, #f43f5e) !important;
}

/* Hide streamlit branding */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load model, scaler and metadata once and reuse them across reruns."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = joblib.load(META_PATH)
    return model, scaler, meta


def risk_band(probability: float, threshold: float) -> str:
    """Translate a probability into a Low / Moderate / High band."""
    if probability >= threshold:
        return "High"
    if probability >= threshold / 2:
        return "Moderate"
    return "Low"


# --------------------------------------------------------------------------- #
# Page header
# --------------------------------------------------------------------------- #
st.title("🧠 Stroke Risk Predictor")
with st.expander("ℹ️ About the Model"):
    st.write(
        "An Ensemble Approach for Stroke Prediction - Soft Voting of "
        "Random Forest, XGBoost and Logistic Regression (trained with SMOTE)."
    )

if not (MODEL_PATH.exists() and SCALER_PATH.exists() and META_PATH.exists()):
    st.error(
        "Trained model files were not found in the `artifacts/` folder. "
        "Please run `python train.py` first, then reload this page."
    )
    st.stop()

model, scaler, meta = load_artifacts()
threshold = float(meta["threshold"])

# --------------------------------------------------------------------------- #
# Input form
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Patient Profile")
    with st.form("patient_form"):
        st.subheader("Personal Info")
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        age = st.slider("Age (years)", min_value=1, max_value=120, value=50, step=1, help="Age is a significant risk factor.")
        ever_married = st.radio("Ever married", ["No", "Yes"], index=1, horizontal=True)
        work_type = st.selectbox(
            "Work type",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        )
        residence_type = st.radio("Residence type", ["Urban", "Rural"], horizontal=True)

        st.divider()

        st.subheader("Medical History")
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True, help="High blood pressure.")
        heart_disease = st.radio("Heart disease", ["No", "Yes"], horizontal=True)
        smoking_status = st.selectbox(
            "Smoking status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
        )
        avg_glucose_level = st.slider(
            "Average glucose level (mg/dL)",
            min_value=40.0, max_value=400.0, value=100.0, step=0.5,
            help="Normal level is typically under 140 mg/dL."
        )
        bmi = st.slider(
            "BMI (kg/m²)", min_value=10.0, max_value=70.0, value=25.0, step=0.1,
            help="Normal range is 18.5 to 24.9."
        )
        bmi_unknown = st.checkbox(
            "BMI is unknown (use median)"
        )

        submitted = st.form_submit_button("Predict Stroke Risk", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
# Prediction Dashboard
# --------------------------------------------------------------------------- #
if not submitted:
    st.info("👈 Please enter the patient's details in the sidebar and click **Predict Stroke Risk** to generate a risk analysis dashboard.")
else:
    patient = pd.DataFrame(
        [{
            "gender": gender,
            "age": age,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": float("nan") if bmi_unknown else bmi,
            "smoking_status": smoking_status,
        }]
    )

    # Exactly the same pipeline as training: impute -> encode -> scale.
    features = encode_features(patient, meta["bmi_median"])
    features_scaled = scaler.transform(features)

    probability = float(model.predict_proba(features_scaled)[0, 1])
    band = risk_band(probability, threshold)

    st.divider()
    
    col_res, col_chart = st.columns([1, 1])
    with col_res:
        st.subheader("Result")
        st.metric("Estimated stroke risk probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))

        if band == "High":
            st.error(
                f"**High risk** - the ensemble's probability is at or above the "
                f"decision threshold ({threshold:.0%}). Please consult a doctor."
            )
        elif band == "Moderate":
            st.warning(
                "**Moderate risk** - some risk indicators are present. "
                "Consider a routine check-up and lifestyle review."
            )
        else:
            st.success("**Low risk** - the model sees few stroke risk indicators.")
            
    with col_chart:
        st.subheader("Model Breakdown")
        breakdown = pd.DataFrame(
            {
                "Stroke probability": {
                    MODEL_LABELS.get(name, name): float(est.predict_proba(features_scaled)[0, 1])
                    for name, est in model.named_estimators_.items()
                }
            }
        )
        breakdown.loc["Soft Voting Ensemble"] = probability
        st.bar_chart(breakdown)

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is an academic project and NOT a medical device. "
    "The model was trained on a class-balanced (SMOTE) dataset, so the probability "
    "is a relative risk score rather than a calibrated clinical probability. "
    "Always consult a qualified healthcare professional."
)
