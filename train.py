"""
train.py - Model training pipeline for
"An Ensemble Approach for Stroke Prediction".

Pipeline
--------
1.  Load the Kaggle Stroke Prediction Dataset (CSV) and clean it.
2.  Stratified train/test split (BEFORE any fitting, to avoid data leakage).
3.  Impute missing BMI (train-set median) and encode categorical variables.
4.  Hyper-parameter tuning: RandomizedSearchCV with stratified K-fold CV for each
    base model. Scaling + SMOTE happen INSIDE every CV fold (imblearn Pipeline),
    so the validation folds never see synthetic or leaked data.
5.  Build a Soft Voting Classifier from the tuned Random Forest, XGBoost and
    Logistic Regression.
6.  Decision-threshold tuning on out-of-fold training predictions (F2-score).
7.  Final fit: scaler fitted on the training set, SMOTE on the training set only.
8.  Evaluate on the untouched test set (Recall and F1 are the headline metrics),
    and compare against an untuned ensemble.
9.  Save model, scaler, metadata and metrics with joblib / json.

Usage
-----
    python train.py                         # full tuning (a few minutes)
    python train.py --n-iter 8 --cv-folds 3 # quick run
    python train.py --threshold 0.4         # force a decision threshold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend: saves figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocessing import FEATURE_COLUMNS, encode_features

REQUIRED_COLUMNS = {
    "id", "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi",
    "smoking_status", "stroke",
}

# Metrics tracked during cross-validation. F2 (recall weighted 4x more than
# precision) is used to pick the best hyper-parameters; F1 and Recall are
# recorded so we can report them.
SCORING = {
    "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
}

MODEL_LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "lr": "Logistic Regression"}

# Reasonable starting values (used for the "untuned" comparison ensemble).
DEFAULT_PARAMS = {
    "rf": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2},
    "xgb": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8},
    "lr": {"C": 1.0},
}

# Search spaces explored by RandomizedSearchCV.
SEARCH_SPACES = {
    "rf": {
        "n_estimators": [200, 300, 500],
        "max_depth": [4, 6, 8, 10, 15, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "xgb": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
        "reg_lambda": [1, 2, 5],
    },
    "lr": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
    },
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the stroke-prediction ensemble.")
    parser.add_argument("--data", type=Path,
                        default=Path("data") / "healthcare-dataset-stroke-data.csv",
                        help="Path to the Kaggle stroke dataset CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"),
                        help="Folder where the model, scaler and metrics are saved.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data held out for testing (default 0.2).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Force a decision threshold. If omitted, it is tuned "
                             "automatically on out-of-fold training predictions.")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of stratified CV folds for tuning (default 5).")
    parser.add_argument("--n-iter", type=int, default=25,
                        help="Random-search candidates per model (default 25).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV and perform basic, row-level cleaning."""
    if not path.exists():
        sys.exit(
            f"[ERROR] Dataset not found at '{path}'.\n"
            "Download 'healthcare-dataset-stroke-data.csv' from Kaggle "
            "(fedesoriano/stroke-prediction-dataset) and place it in the 'data/' folder."
        )

    df = pd.read_csv(path)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        sys.exit(f"[ERROR] Dataset is missing expected columns: {sorted(missing_cols)}")

    # `id` is just a row identifier and carries no predictive signal.
    df = df.drop(columns=["id"])

    # The dataset contains a single record with gender == 'Other'; it cannot be
    # learned from (or stratified on), so it is removed.
    n_other = int((df["gender"] == "Other").sum())
    df = df[df["gender"] != "Other"].reset_index(drop=True)

    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Rows loaded                 : {len(df)} (removed {n_other} 'Other' gender row(s))")
    print(f"Missing BMI values          : {int(df['bmi'].isna().sum())}")
    print("Class distribution (stroke) :")
    counts = df["stroke"].value_counts().sort_index()
    for label, count in counts.items():
        print(f"    class {label}: {count:5d}  ({count / len(df):.2%})")
    return df


def make_estimator(name: str, params: dict, seed: int, n_jobs: int = 1):
    """Create one base learner with the given hyper-parameters."""
    if name == "rf":
        return RandomForestClassifier(**params, random_state=seed, n_jobs=n_jobs)
    if name == "xgb":
        return XGBClassifier(**params, eval_metric="logloss", random_state=seed, n_jobs=n_jobs)
    if name == "lr":
        return LogisticRegression(**params, max_iter=2000, random_state=seed)
    raise ValueError(f"Unknown model name: {name}")


def make_ensemble(params_by_model: dict, seed: int, n_jobs: int = 1) -> VotingClassifier:
    """Soft Voting Classifier: averages the predicted probabilities of all models."""
    return VotingClassifier(
        estimators=[(name, make_estimator(name, params_by_model[name], seed, n_jobs))
                    for name in ("rf", "xgb", "lr")],
        voting="soft",
    )


def tune_model(name: str, X_train: pd.DataFrame, y_train: pd.Series,
               cv: StratifiedKFold, n_iter: int, seed: int):
    """
    Random search for one base model.

    The estimator is wrapped in Scaler -> SMOTE -> Model, so inside every CV fold the
    scaler and SMOTE are fitted on that fold's TRAINING part only; the validation part
    stays real and untouched.
    """
    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", make_estimator(name, {}, seed, n_jobs=1)),
    ])
    space = {f"clf__{key}": values for key, values in SEARCH_SPACES[name].items()}
    n_candidates = int(np.prod([len(v) for v in SEARCH_SPACES[name].values()]))

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=space,
        n_iter=min(n_iter, n_candidates),
        scoring=SCORING,
        refit="f2",          # best candidate is chosen by F2 (recall-weighted)
        cv=cv,
        n_jobs=-1,
        random_state=seed,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    best_params = {k.split("__", 1)[1]: v for k, v in search.best_params_.items()}
    idx = search.best_index_
    cv_scores = {m: float(search.cv_results_[f"mean_test_{m}"][idx]) for m in SCORING}
    return best_params, cv_scores


def tune_threshold(ensemble: VotingClassifier, X_train: pd.DataFrame, y_train: pd.Series,
                   cv: StratifiedKFold, seed: int):
    """
    Pick the probability threshold that maximises F2 on OUT-OF-FOLD training
    predictions. The test set plays no role in this choice.
    """
    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", ensemble),
    ])
    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [
        fbeta_score(y_train, (oof_proba >= t).astype(int), beta=2, zero_division=0)
        for t in grid
    ]
    best_idx = int(np.argmax(scores))
    return float(grid[best_idx]), float(scores[best_idx])


def compute_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    """Compute the evaluation metrics for one set of predicted probabilities."""
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def save_confusion_matrix(y_true, y_pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No Stroke", "Stroke"],
        cmap="Blues", values_format="d", ax=ax,
    )
    ax.set_title("Tuned Soft Voting Ensemble - Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def json_default(obj):
    """Make numpy scalars JSON-serialisable."""
    return obj.item() if hasattr(obj, "item") else str(obj)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + clean ---------------------------------------------------------
    df = load_data(args.data)
    X_raw = df.drop(columns=["stroke"])
    y = df["stroke"].astype(int)

    # 2. Stratified split BEFORE imputing / scaling / SMOTE (prevents leakage) --
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=args.test_size, stratify=y, random_state=args.seed,
    )
    print(f"\nTrain size: {len(X_train_raw)} | Test size: {len(X_test_raw)}")

    # 3. Impute BMI (train median only) + encode categoricals -------------------
    bmi_median = float(X_train_raw["bmi"].median())
    print(f"BMI median (from training set, used for imputation): {bmi_median:.2f}")
    X_train = encode_features(X_train_raw, bmi_median)
    X_test = encode_features(X_test_raw, bmi_median)

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    # 4. Hyper-parameter tuning ---------------------------------------------------
    print("\n" + "=" * 70)
    print(f"HYPER-PARAMETER TUNING  ({args.n_iter} candidates x {args.cv_folds}-fold CV per model,"
          " scored by F2)")
    print("=" * 70)
    best_params, cv_results = {}, {}
    for name in ("rf", "xgb", "lr"):
        print(f"\nTuning {MODEL_LABELS[name]} ...")
        best_params[name], cv_results[name] = tune_model(
            name, X_train, y_train, cv, args.n_iter, args.seed
        )
        print(f"  best params : {best_params[name]}")
        print("  CV scores   : " + ", ".join(f"{m}={v:.4f}" for m, v in cv_results[name].items()))

    # 5-6. Threshold tuning on out-of-fold predictions ---------------------------------
    if args.threshold is None:
        print("\nTuning decision threshold on out-of-fold training predictions (F2)...")
        threshold, oof_f2 = tune_threshold(
            make_ensemble(best_params, args.seed, n_jobs=1), X_train, y_train, cv, args.seed
        )
        print(f"  chosen threshold = {threshold:.2f}  (out-of-fold F2 = {oof_f2:.4f})")
    else:
        threshold = float(args.threshold)
        print(f"\nUsing user-supplied decision threshold = {threshold:.2f}")

    # 7. Final fit: scaler (train only) + SMOTE (train only) ------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=args.seed)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    before = {int(k): int(v) for k, v in y_train.value_counts().sort_index().items()}
    after = {int(k): int(v) for k, v in pd.Series(y_train_res).value_counts().sort_index().items()}
    print("\nClass counts in training set (before -> after SMOTE):")
    print(f"    before: {before}")
    print(f"    after : {after}")

    print("\nTraining final models...")
    default_ensemble = make_ensemble(DEFAULT_PARAMS, args.seed, n_jobs=-1)
    default_ensemble.fit(X_train_res, y_train_res)
    tuned_ensemble = make_ensemble(best_params, args.seed, n_jobs=-1)
    tuned_ensemble.fit(X_train_res, y_train_res)
    print("Training complete.")

    # 8. Evaluation on the untouched test set --------------------------------------
    results = {}
    for name in ("rf", "xgb", "lr"):
        proba = tuned_ensemble.named_estimators_[name].predict_proba(X_test_scaled)[:, 1]
        results[f"{MODEL_LABELS[name]} (tuned, thr 0.50)"] = compute_metrics(y_test, proba, 0.5)

    default_proba = default_ensemble.predict_proba(X_test_scaled)[:, 1]
    tuned_proba = tuned_ensemble.predict_proba(X_test_scaled)[:, 1]
    results["Ensemble (untuned, thr 0.50)"] = compute_metrics(y_test, default_proba, 0.5)
    results["Ensemble (tuned, thr 0.50)"] = compute_metrics(y_test, tuned_proba, 0.5)
    final_key = f"Ensemble (tuned, thr {threshold:.2f})  <- FINAL"
    results[final_key] = compute_metrics(y_test, tuned_proba, threshold)

    baseline_accuracy = float(1 - y_test.mean())

    print("\n" + "=" * 70)
    print("TEST-SET RESULTS")
    print("=" * 70)
    print(pd.DataFrame(results).T.round(4).to_string())

    final = results[final_key]
    final_pred = (tuned_proba >= threshold).astype(int)
    print(f"\n--- FINAL MODEL: tuned Soft Voting Ensemble, threshold = {threshold:.2f} ---")
    print(f"Recall (Sensitivity) : {final['recall']:.4f}   <-- primary metric")
    print(f"F1-Score             : {final['f1']:.4f}   <-- secondary metric")
    print(f"Precision            : {final['precision']:.4f}")
    print(f"ROC-AUC              : {final['roc_auc']:.4f}")
    print(f"Accuracy             : {final['accuracy']:.4f}   (shown for reference only)")
    print(f"\nFor comparison, a useless model that ALWAYS predicts 'No Stroke' "
          f"would score {baseline_accuracy:.4f} accuracy with 0.0000 recall.")

    tn, fp, fn, tp = confusion_matrix(y_test, final_pred).ravel()
    print(f"\nConfusion matrix -> TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print("\nClassification report:")
    print(classification_report(
        y_test, final_pred, target_names=["No Stroke", "Stroke"], zero_division=0
    ))

    # 9. Persist artefacts --------------------------------------------------------------
    model_path = args.output_dir / "stroke_ensemble_model.joblib"
    scaler_path = args.output_dir / "scaler.joblib"
    meta_path = args.output_dir / "metadata.joblib"
    params_path = args.output_dir / "best_params.json"
    metrics_path = args.output_dir / "metrics.json"
    cm_path = args.output_dir / "confusion_matrix.png"

    joblib.dump(tuned_ensemble, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(
        {"feature_columns": FEATURE_COLUMNS, "bmi_median": bmi_median, "threshold": threshold},
        meta_path,
    )
    with open(params_path, "w") as f:
        json.dump({"best_params": best_params, "cv_scores": cv_results},
                  f, indent=2, default=json_default)
    with open(metrics_path, "w") as f:
        json.dump(
            {"threshold": threshold,
             "always_no_stroke_baseline_accuracy": baseline_accuracy,
             "results": results},
            f, indent=2, default=json_default,
        )
    save_confusion_matrix(y_test, final_pred, cm_path)

    print("Saved artefacts:")
    for p in (model_path, scaler_path, meta_path, params_path, metrics_path, cm_path):
        print(f"    {p}")
    print("\nNext step: streamlit run app.py")


if __name__ == "__main__":
    main()
