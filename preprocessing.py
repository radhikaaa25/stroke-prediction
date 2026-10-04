"""
preprocessing.py
================
Single source of truth for feature engineering.

Both `train.py` (training) and `app.py` (inference) import `encode_features`
from this module, so the Streamlit app is guaranteed to apply EXACTLY the same
encoding, column order and BMI imputation that the model saw during training.
"""

from __future__ import annotations

import pandas as pd

# Columns that are continuous / numeric in the raw dataset.
NUMERIC_COLUMNS = ["age", "avg_glucose_level", "bmi"]

# Binary categorical columns -> fixed 0/1 mapping.
BINARY_MAPS = {
    "gender": {"Female": 0, "Male": 1},
    "ever_married": {"No": 0, "Yes": 1},
    "Residence_type": {"Rural": 0, "Urban": 1},
}

# Multi-class categorical columns -> one-hot with a FIXED list of levels, so
# the output columns never depend on which categories appear in a given batch.
CATEGORY_LEVELS = {
    "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"],
}

ONE_HOT_COLUMNS = [
    f"{column}_{level}"
    for column, levels in CATEGORY_LEVELS.items()
    for level in levels
]

# Final model input order (the scaler and the model both rely on this order).
FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
] + ONE_HOT_COLUMNS


def encode_features(df: pd.DataFrame, bmi_median: float) -> pd.DataFrame:
    """
    Convert raw patient records into the numeric feature matrix used by the model.

    Steps
    -----
    1. Impute missing BMI with `bmi_median` (computed on the TRAINING set only).
    2. Map binary categoricals (gender, ever_married, Residence_type) to 0/1.
    3. One-hot encode work_type and smoking_status using fixed levels.
    4. Return columns in the exact order given by FEATURE_COLUMNS.

    Raises
    ------
    ValueError if an unseen category value is encountered.
    """
    encoded = pd.DataFrame(index=df.index)

    # Numeric columns (BMI gets median imputation).
    encoded["age"] = pd.to_numeric(df["age"], errors="coerce")
    encoded["avg_glucose_level"] = pd.to_numeric(df["avg_glucose_level"], errors="coerce")
    encoded["hypertension"] = pd.to_numeric(df["hypertension"], errors="coerce")
    encoded["heart_disease"] = pd.to_numeric(df["heart_disease"], errors="coerce")
    encoded["bmi"] = pd.to_numeric(df["bmi"], errors="coerce").fillna(bmi_median)

    # Binary categorical columns.
    for column, mapping in BINARY_MAPS.items():
        encoded[column] = df[column].map(mapping)

    # One-hot columns with fixed levels.
    for column, levels in CATEGORY_LEVELS.items():
        for level in levels:
            encoded[f"{column}_{level}"] = (df[column] == level).astype(int)

    encoded = encoded[FEATURE_COLUMNS]

    bad_columns = encoded.columns[encoded.isna().any()].tolist()
    if bad_columns:
        raise ValueError(
            f"Missing or unrecognised values found in column(s): {bad_columns}. "
            "Check for categories that were not present in the training data."
        )

    return encoded.astype(float)
