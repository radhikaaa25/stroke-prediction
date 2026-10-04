# Stroke Prediction Dataset Analysis

This document provides an overview of the Kaggle Stroke Prediction Dataset used in this project and explains how the various input parameters (features) contribute to the model's stroke predictions.

## 1. The Dataset Overview

The dataset is the **Kaggle Stroke Prediction Dataset**, which contains health and demographic records for over 5,000 patients. It is designed to train machine learning models to predict whether a patient is likely to suffer a stroke based on their individual profile.

### Key Characteristics:
* **Size**: 5,110 rows (patients) originally, trimmed to 5,109 after removing an invalid row.
* **Target Variable**: `stroke` (1 if the patient had a stroke, 0 otherwise).
* **High Class Imbalance**: The dataset is extremely unbalanced. Only about **~4.87%** of the patients in the dataset have actually suffered a stroke, while over 95% have not. This makes techniques like SMOTE (Synthetic Minority Over-sampling Technique) essential for model training to avoid predicting "No Stroke" every single time.
* **Missing Data**: The `bmi` column contains a few hundred missing values (NaNs), which the pipeline handles by imputing (filling in) the median BMI derived from the training set.

---

## 2. Parameter Explanations (Stroke Risk Factors)

The model uses 10 distinct features (parameters) to predict stroke probabilities. The following parameters are the primary indicators that generally push the model towards a **higher stroke prediction (closer to 1.0)**:

### 🔴 Primary Clinical & Physiological Risk Factors

1. **`age` (Age of the patient)**
   * **Impact on Prediction:** High. Age is universally the strongest single predictor of stroke risk. The risk of stroke doubles every decade after age 55. As the input `age` goes up, the model's predicted probability of a stroke will sharply increase.
   
2. **`hypertension` (High Blood Pressure)**
   * **Impact on Prediction:** High. A value of `1` means the patient suffers from hypertension. High blood pressure puts immense stress on blood vessel walls, which can lead to weakened arteries or clots. If this is `1`, the model heavily weights the patient towards a positive stroke prediction.
   
3. **`heart_disease` (Presence of Heart Disease)**
   * **Impact on Prediction:** High. A value of `1` indicates existing heart disease. Conditions like coronary artery disease and atrial fibrillation (irregular heartbeat) drastically increase the risk of blood clots traveling to the brain.
   
4. **`avg_glucose_level` (Average Blood Glucose Level)**
   * **Impact on Prediction:** Medium to High. Higher glucose levels are highly indicative of pre-diabetes or active diabetes. Diabetes causes pathological changes in blood vessels and accelerates atherosclerosis (narrowing of the arteries), which significantly increases stroke risk.
   
5. **`bmi` (Body Mass Index)**
   * **Impact on Prediction:** Medium. An exceptionally high BMI indicates obesity, which is closely linked to high blood pressure, high cholesterol, and diabetes (the "metabolic syndrome"). Extremely high values correlate positively with stroke probability.

### 🟡 Lifestyle & Demographic Risk Factors

6. **`smoking_status` (Smoker Status)**
   * **Categories:** `smokes`, `formerly smoked`, `never smoked`, `Unknown`.
   * **Impact on Prediction:** Medium. Patients who are encoded as `smokes` or `formerly smoked` receive a higher risk score. Smoking damages blood vessels, raises blood pressure, and makes the blood more prone to clotting.

7. **`work_type` (Occupational Status)**
   * **Categories:** `Private`, `Self-employed`, `Govt_job`, `children`, `Never_worked`.
   * **Impact on Prediction:** Low to Medium. `children` generally acts as a strong negative indicator (extremely low risk). Conversely, `Self-employed` or `Private` jobs might marginally increase the risk in the model, though this is heavily confounded with age (e.g., self-employed people in the dataset tend to be older adults).

8. **`ever_married` (Marital Status)**
   * **Categories:** `Yes` or `No`.
   * **Impact on Prediction:** Low. Often found to be correlated in models simply because people who have been married are generally older than those who haven't. The model leverages it, but it isn't an independent physiological risk.

9. **`Residence_type` (Living Area)**
   * **Categories:** `Urban` or `Rural`.
   * **Impact on Prediction:** Very Low. Differences here might relate to stress, diet variations, or pollution, but statistically, it provides very little predictive power for the final outcome. 

10. **`gender` (Male / Female)**
    * **Categories:** `Male`, `Female`.
    * **Impact on Prediction:** Very Low. While gender plays a minor role in stroke statistics globally (men have strokes at a slightly younger age, women have a higher lifetime risk due to living longer), in this specific dataset, its predictive weight is minimal compared to age and clinical health factors.

### Summary
If a user inputs a profile into your Streamlit App that features **high age**, **hypertension = 1**, **heart_disease = 1**, and a **high average glucose level**, the soft-voting ensemble model will output a dramatically higher stroke probability than for a young, healthy individual.
