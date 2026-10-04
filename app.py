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

st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🧠", layout="centered")


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
st.caption(
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
with st.form("patient_form"):
    st.subheader("Patient details")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever married", ["No", "Yes"], index=1)
    with col2:
        work_type = st.selectbox(
            "Work type",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        )
        residence_type = st.selectbox("Residence type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input(
            "Average glucose level (mg/dL)",
            min_value=40.0, max_value=400.0, value=100.0, step=0.5,
        )
        bmi = st.number_input(
            "BMI (kg/m²)", min_value=10.0, max_value=70.0, value=25.0, step=0.1
        )
        smoking_status = st.selectbox(
            "Smoking status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
        )

    bmi_unknown = st.checkbox(
        "BMI is unknown (use the median BMI from the training data)"
    )

    submitted = st.form_submit_button("Predict stroke risk", type="primary")

# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
if submitted:
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

    # Show what each base learner thinks (transparency of the ensemble).
    breakdown = pd.DataFrame(
        {
            "Stroke probability": {
                MODEL_LABELS.get(name, name): float(est.predict_proba(features_scaled)[0, 1])
                for name, est in model.named_estimators_.items()
            }
        }
    )
    breakdown.loc["Soft Voting Ensemble"] = probability

    with st.expander("See how each model voted"):
        st.bar_chart(breakdown)
        st.dataframe(breakdown.style.format("{:.1%}"))

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is an academic project and NOT a medical device. "
    "The model was trained on a class-balanced (SMOTE) dataset, so the probability "
    "is a relative risk score rather than a calibrated clinical probability. "
    "Always consult a qualified healthcare professional."
)
