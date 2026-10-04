# An Ensemble Approach for Stroke Prediction

A machine-learning mini-project that predicts the likelihood of a stroke from basic
patient information. A **Soft Voting Classifier** combining **Random Forest**,
**XGBoost** and **Logistic Regression** is trained on the Kaggle Stroke Prediction
Dataset, with **SMOTE** used to handle the severe class imbalance. A **Streamlit**
web app lets users enter patient vitals and see the predicted risk.

## Project structure

```
stroke_prediction_project/
├── data/
│   └── healthcare-dataset-stroke-data.csv   <- you download this (Step 3)
├── artifacts/                               <- created by train.py
│   ├── stroke_ensemble_model.joblib
│   ├── scaler.joblib
│   ├── metadata.joblib
│   ├── best_params.json                     # tuned hyper-parameters + CV scores
│   ├── metrics.json
│   └── confusion_matrix.png
├── preprocessing.py   # shared encoding + BMI imputation (used by train.py AND app.py)
├── train.py           # training + evaluation pipeline
├── app.py             # Streamlit front-end
├── requirements.txt
└── README.md
```

## Prerequisites

* Python **3.9 or newer** (3.10 / 3.11 recommended)
* `pip`
* A free Kaggle account (only to download the dataset)

## Step-by-step instructions

### 1. Open the project folder

```bash
cd stroke_prediction_project
```

### 2. Create a virtual environment and install dependencies

**Windows (PowerShell / CMD)**

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the dataset

1. Go to the Kaggle dataset **"Stroke Prediction Dataset"** by *fedesoriano*:
   <https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset>
2. Click **Download** and extract the ZIP.
3. Copy `healthcare-dataset-stroke-data.csv` into the `data/` folder, so the path is:

   ```
   data/healthcare-dataset-stroke-data.csv
   ```

### 4. Train the model

```bash
python train.py
```

This will:

1. Load and clean the dataset (drops `id`, drops the single `Other` gender row).
2. Split into train/test sets (80/20, stratified).
3. Impute missing BMI (training-set median) and encode categorical variables.
4. **Tune hyper-parameters** of Random Forest, XGBoost and Logistic Regression with
   `RandomizedSearchCV` (stratified 5-fold CV). Scaling and **SMOTE run inside every CV
   fold**, so validation folds stay untouched. Candidates are ranked by F2-score
   (recall counts 4x more than precision).
5. **Tune the decision threshold** on out-of-fold training predictions (F2-score).
6. Fit the scaler and apply **SMOTE to the training set only**, then train the tuned
   Soft Voting Ensemble.
7. Print **Recall, F1-Score**, Precision, ROC-AUC and the confusion matrix on the test set,
   including an untuned-vs-tuned comparison table.
8. Save everything into the `artifacts/` folder.

Tuning takes a few minutes (it depends on your CPU). For a quick trial run:

```bash
python train.py --n-iter 8 --cv-folds 3
```

Other optional arguments:

```bash
python train.py --threshold 0.5          # skip threshold tuning and force 0.5
python train.py --n-iter 50              # search more candidates (slower, more thorough)
python train.py --test-size 0.25 --seed 7
python train.py --data path/to/other.csv
```

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

The app opens automatically at <http://localhost:8501>. Enter the patient details and
click **Predict stroke risk** to see the probability, the risk band and the vote of each
individual model.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Dataset not found at 'data/...csv'` | Complete Step 3 and check the file name matches exactly. |
| App says *Trained model files were not found* | Run `python train.py` first (Step 4). |
| `ModuleNotFoundError` | Activate the virtual environment and re-run `pip install -r requirements.txt`. |
| `streamlit: command not found` | Use `python -m streamlit run app.py`. |
| Errors loading the model after upgrading packages | Re-run `python train.py` so the model is re-saved with your installed library versions. |

## Notes on methodology

* **No data leakage:** the split happens first; the BMI median, scaler and SMOTE are all
  fitted on the training data only (and inside each fold during tuning), and the test set
  is never resampled or used to choose hyper-parameters or the threshold.
* **Recall vs precision trade-off:** the tuned threshold favours Recall, so Precision, F1
  and Accuracy may drop slightly compared with a plain 0.5 threshold. Use
  `--threshold 0.5` if you prefer a more balanced operating point.
* **Why not Accuracy?** Only about 5 % of patients in the dataset had a stroke, so a model
  that always predicts "No Stroke" scores ~95 % accuracy while missing every patient.
  Recall and F1-Score are used as the primary metrics instead.
* **Disclaimer:** this is an academic project and not a medical device.
