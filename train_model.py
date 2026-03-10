import pandas as pd
import numpy as np
import joblib
import json
import shap
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    PowerTransformer, QuantileTransformer,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve,
)

# ── 1. Generate synthetic training data ───────────────────────────────────────
rng = np.random.default_rng(42)
N   = 6_000

# --- Feature distributions ---
dept_ids = rng.choice([101, 102, 103, 104, 105], size=N, p=[0.22, 0.22, 0.18, 0.20, 0.18])
cat_ids  = rng.choice([1, 2, 3, 4, 5, 6, 7, 8], size=N,
                      p=[0.20, 0.18, 0.14, 0.14, 0.12, 0.10, 0.07, 0.05])

# Amount: mix low / medium / high in realistic proportions
amount_tier = rng.choice(["low", "med", "high", "very_high"], size=N, p=[0.55, 0.30, 0.11, 0.04])
amounts = np.where(
    amount_tier == "low",      rng.uniform(100,    5_000,   N),
    np.where(
    amount_tier == "med",      rng.uniform(5_000,  50_000,  N),
    np.where(
    amount_tier == "high",     rng.uniform(50_000, 100_000, N),
                               rng.uniform(100_000, 250_000, N)
)))

# Make ~25% of amounts round numbers (multiples of 500)
round_mask = rng.random(N) < 0.25
amounts    = np.where(round_mask, (amounts // 500) * 500, amounts)
amounts    = np.round(amounts, 2)

is_round = (amounts % 500 == 0).astype(int)

# Receipt: 78% have one; expensive / no-receipt = suspicious
has_receipt = rng.random(N)
is_affidavit = np.where(
    amounts < 500,   (has_receipt > 0.65).astype(int),   # cheap: sometimes no receipt
    np.where(amounts < 5_000,  (has_receipt > 0.82).astype(int),
             np.where(amounts < 50_000, (has_receipt > 0.90).astype(int),
                                        (has_receipt > 0.95).astype(int)))
)

# Submission gap: 0 = same-day, 1-5 = quick, 6-30 = normal, 31-60 = late
gap_tier = rng.choice(["same_day", "quick", "normal", "late", "very_late"], size=N,
                      p=[0.06, 0.20, 0.56, 0.12, 0.06])
gaps = np.where(
    gap_tier == "same_day",  0,
    np.where(
    gap_tier == "quick",     rng.integers(1,  6,  N),
    np.where(
    gap_tier == "normal",    rng.integers(6,  31, N),
    np.where(
    gap_tier == "late",      rng.integers(31, 61, N),
                             rng.integers(31, 61, N)   # very_late same range, flagged separately
))))
is_weekend = (rng.random(N) < 0.18).astype(int)

# --- Label generation: rule-based risk score → binary label ---
risk = np.zeros(N, dtype=float)

# Amount risk
risk += np.where(amounts >= 100_000, 3.5, 0)
risk += np.where((amounts >= 50_000) & (amounts < 100_000), 2.0, 0)
risk += np.where((amounts >= 10_000) & (amounts < 50_000), 0.8, 0)

# No receipt is a strong risk signal (especially at higher amounts)
no_receipt_weight = np.where(amounts >= 10_000, 2.5, np.where(amounts >= 1_000, 1.5, 0.8))
risk += np.where(is_affidavit == 1, no_receipt_weight, 0)

# Same-day submission (may indicate backdating)
risk += np.where(gaps == 0, 2.0, 0)
# Very late submission
risk += np.where(gaps > 45, 1.5, np.where(gaps > 30, 0.8, 0))

# Weekend purchase
risk += np.where(is_weekend == 1, 0.6, 0)

# Round number
risk += np.where(is_round == 1, 0.7, 0)

# Meals over limit
risk += np.where((cat_ids == 1) & (amounts > 5_000), 1.0, 0)

# Add small random noise so boundary cases vary
risk += rng.normal(0, 0.3, N)

# Label = 1 (anomalous) if risk >= 3.0, with a small symmetric noise flip
labels = (risk >= 3.0).astype(int)
flip   = rng.random(N) < 0.04           # 4% label noise
labels = np.where(flip, 1 - labels, labels)

df = pd.DataFrame({
    "Exp_TotalAmount": amounts,
    "IsAffidavit":     is_affidavit,
    "ExpCat_ID":       cat_ids,
    "Dept_ID":         dept_ids,
    "Submission_Gap":  gaps,
    "Is_Weekend":      is_weekend,
    "Is_Round_Number": is_round,
    "Label":           labels,
})

print(f"Dataset: {N} rows | Flagged: {labels.sum()} ({labels.mean()*100:.1f}%)")

# ── 3b. Feature engineering ───────────────────────────────────────────────────
# Log-transform amount: compresses the ₱100–₱250k range into a roughly
# Gaussian distribution — RobustScaler then centres this well and prevents
# extreme amounts from dominating the scaler's IQR reference.
df["Log_Amount"] = np.log1p(df["Exp_TotalAmount"])

# Interaction: large amount AND no receipt — the single strongest audit signal.
# A tree split on the raw features can find this, but giving it explicitly
# lets shallower trees (max_depth=6) surface it in the first levels.
df["HighAmt_NoReceipt"] = ((df["Exp_TotalAmount"] > 10_000) & (df["IsAffidavit"] == 1)).astype(int)

# Ordinal amount tier — lets the model learn tier-specific patterns without
# having to discover all the breakpoints itself from raw amounts.
df["Amount_Tier"] = np.digitize(
    df["Exp_TotalAmount"],
    bins=[1_000, 5_000, 50_000, 100_000],
).astype(int)  # 0=<1k, 1=1k-5k, 2=5k-50k, 3=50k-100k, 4=100k+

# Interaction: same-day submission AND no receipt — strongest backdating signal.
df["SameDay_NoReceipt"] = ((df["Submission_Gap"] == 0) & (df["IsAffidavit"] == 1)).astype(int)

# Gap ratio: submission gap normalised to the 60-day policy window.
# Provides a linear view of urgency/lateness alongside raw gap days.
df["Gap_Ratio"] = df["Submission_Gap"] / 60.0

df.to_csv("smartspend_synthetic_data.csv", index=False)
print("✅ synthetic subset exported to smartspend_synthetic_data.csv")

# ── 4. Features / target ──────────────────────────────────────────────────────
# Numeric features — these have wide or skewed ranges and benefit from
# RobustScaler (uses median/IQR, so outlier amounts don't distort scaling).
NUMERIC_FEATURES = [
    "Log_Amount",      # log-compressed amount (handles ₱100–₱250k skew)
    "Submission_Gap",  # days 0–60+  (wide range, right-skewed)
    "Gap_Ratio",       # Submission_Gap / 60 (0–1+ normalised linear view)
]

# Passthrough features — binary flags and low-cardinality integers.
# XGBoost tree splits don't need these scaled; scaling binary {0,1} columns
# would just add noise, not signal.
PASSTHROUGH_FEATURES = [
    "IsAffidavit",      # binary: 0=receipt, 1=affidavit
    "ExpCat_ID",        # 1–8 category ID
    "Dept_ID",          # 101–105 dept ID
    "Is_Weekend",       # binary: purchase on weekend
    "Is_Round_Number",  # binary: amount multiple of ₱500
    "HighAmt_NoReceipt",# interaction: high amount + no receipt
    "Amount_Tier",      # ordinal 0–4 tier
    "SameDay_NoReceipt",# interaction: gap=0 + no receipt
]

ALL_FEATURE_COLS = NUMERIC_FEATURES + PASSTHROUGH_FEATURES  # 11 total

X = df[ALL_FEATURE_COLS].copy().astype(float)
y = df["Label"]

# ── 5. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 6. Build ColumnTransformer preprocessor ──────────────────────────────────
# Why ColumnTransformer over a global scaler?
#   • Numeric columns (Log_Amount, Submission_Gap, Gap_Ratio) have very
#     different scales and right-skewed distributions.  RobustScaler uses
#     median + IQR, so extreme expense amounts (₱250k outliers) don't pull
#     the scaler parameters like StandardScaler's mean/std would.
#   • Binary / low-cardinality columns (IsAffidavit, ExpCat_ID, etc.) carry
#     no benefit from scaling — scaling {0,1} flags to [-1, +1] just adds
#     floating-point noise to tree split thresholds.
#   • Keeping them separate makes each column's preprocessing intent explicit
#     and easy to swap per-column later.
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
imbalance_ratio = neg_count / pos_count
print(f"Class imbalance ratio (scale_pos_weight): {imbalance_ratio:.2f}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num",  RobustScaler(), NUMERIC_FEATURES),
        ("pass", "passthrough",  PASSTHROUGH_FEATURES),
    ],
    remainder="drop",
)

# ── 7. Train XGBoost — Pipeline: ColumnTransformer → XGBClassifier ───────────
xgb_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("xgb",        XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        scale_pos_weight=imbalance_ratio,
        subsample=0.85,          # row-sampling: reduces overfitting, adds diversity
        colsample_bytree=0.85,   # feature-sampling per tree: same benefit
        min_child_weight=3,      # min samples per leaf: guards against tiny noisy splits
        gamma=0.1,               # min loss reduction to split: light regularisation
        random_state=42, eval_metric="logloss", verbosity=0,
    )),
])
xgb_pipeline.fit(X_train, y_train)

# ── 8. Train Decision Tree — Pipeline: ColumnTransformer → DecisionTreeClassifier
# Decision Tree shares the same preprocessor so the comparison is apples-to-apples.
dt_preprocessor = ColumnTransformer(
    transformers=[
        ("num",  RobustScaler(), NUMERIC_FEATURES),
        ("pass", "passthrough",  PASSTHROUGH_FEATURES),
    ],
    remainder="drop",
)
dt_pipeline = Pipeline([
    ("preprocess", dt_preprocessor),
    ("dt",          DecisionTreeClassifier(max_depth=6, random_state=42)),
])
dt_pipeline.fit(X_train, y_train)

# ── 8b. Scaler benchmark ──────────────────────────────────────────────────────
# Compare five scaler strategies on the numeric features only.
# This helps verify that RobustScaler is the best choice for this dataset.
print("\n── Scaler Benchmark (XGBoost, numeric features only) ──")
scalers_to_test = {
    "StandardScaler":   StandardScaler(),
    "RobustScaler":     RobustScaler(),
    "MinMaxScaler":     MinMaxScaler(),
    "PowerTransformer": PowerTransformer(method="yeo-johnson"),
    "QuantileNormal":   QuantileTransformer(output_distribution="normal", random_state=42),
    "NoScaler":         "passthrough",
}
benchmark_results = {}
for scaler_name, scaler_obj in scalers_to_test.items():
    bench_pre = ColumnTransformer(
        transformers=[
            ("num",  scaler_obj,    NUMERIC_FEATURES),
            ("pass", "passthrough", PASSTHROUGH_FEATURES),
        ],
        remainder="drop",
    )
    bench_pipe = Pipeline([
        ("preprocess", bench_pre),
        ("xgb",        XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=imbalance_ratio, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=3, gamma=0.1,
            random_state=42, eval_metric="logloss", verbosity=0,
        )),
    ])
    bench_pipe.fit(X_train, y_train)
    y_p  = bench_pipe.predict_proba(X_test)[:, 1]
    auc  = round(roc_auc_score(y_test, y_p) * 100, 2)
    rec  = round(recall_score(y_test, (y_p >= 0.40).astype(int)) * 100, 2)
    benchmark_results[scaler_name] = {"auc_roc": auc, "recall": rec}
    print(f"  {scaler_name:22s} → AUC-ROC: {auc:6.2f}%   Recall: {rec:5.1f}%")

# ── 8c. Stratified K-Fold cross-validation ────────────────────────────────────
# A single 80/20 split can be lucky or unlucky.  5-fold stratified CV gives a
# ±std band showing how stable the model is across different data subsets.
print("\n── 5-Fold Stratified Cross-Validation ──")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc_scores = cross_val_score(
    xgb_pipeline, X, y,
    cv=cv, scoring="roc_auc", n_jobs=-1,
)
cv_rec_scores = cross_val_score(
    xgb_pipeline, X, y,
    cv=cv, scoring="recall", n_jobs=-1,
)
print(f"  AUC-ROC : {cv_auc_scores.mean()*100:.2f}% ± {cv_auc_scores.std()*100:.2f}%")
print(f"  Recall  : {cv_rec_scores.mean()*100:.2f}% ± {cv_rec_scores.std()*100:.2f}%")
cv_summary = {
    "auc_roc_mean": round(float(cv_auc_scores.mean() * 100), 2),
    "auc_roc_std":  round(float(cv_auc_scores.std()  * 100), 2),
    "recall_mean":  round(float(cv_rec_scores.mean() * 100), 2),
    "recall_std":   round(float(cv_rec_scores.std()  * 100), 2),
}

# ── 8d. Find optimal classification threshold from Precision-Recall curve ────
# The default threshold of 0.40 was chosen manually.  The PR curve finds the
# threshold with the highest F1 — useful for imbalanced anomaly detection where
# Recall matters more than Precision.
y_proba_train_test = xgb_pipeline.predict_proba(X_test)[:, 1]
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_proba_train_test)
f1_vals = np.where(
    (precision_vals + recall_vals) > 0,
    2 * (precision_vals * recall_vals) / (precision_vals + recall_vals),
    0.0,
)
best_f1_idx    = f1_vals[:-1].argmax()   # last element has no corresponding threshold
optimal_thresh = float(pr_thresholds[best_f1_idx])
print(f"\n── Threshold Analysis ──")
print(f"  Default  threshold (0.40) → F1: {f1_vals[np.searchsorted(pr_thresholds, 0.40, side='left')]:.3f}")
print(f"  Optimal  threshold ({optimal_thresh:.3f})  → F1: {f1_vals[best_f1_idx]:.3f}")

# ── 9. Helper: compute all metrics ───────────────────────────────────────────
def compute_metrics(pipeline, X_t, y_t, model_name, threshold=0.40):
    y_proba = pipeline.predict_proba(X_t)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    cm      = confusion_matrix(y_t, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "model":               model_name,
        "accuracy":            round(accuracy_score(y_t, y_pred) * 100, 2),
        "recall_rejected":     round(recall_score(y_t, y_pred) * 100, 2),
        "precision_approved":  round(precision_score(y_t, y_pred, pos_label=0) * 100, 2),
        "f1_score":            round(f1_score(y_t, y_pred) * 100, 2),
        "auc_roc":             round(roc_auc_score(y_t, y_proba) * 100, 2),
        "confusion_matrix":    {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "total_test":          int(len(y_t)),
        "total_flagged":       int(sum(y_pred)),
        "total_cleared":       int(len(y_pred) - sum(y_pred)),
        "false_negative_rate": round((fn / (fn + tp)) * 100, 2) if (fn + tp) > 0 else 0,
        "false_positive_rate": round((fp / (fp + tn)) * 100, 2) if (fp + tn) > 0 else 0,
    }

xgb_metrics = compute_metrics(xgb_pipeline, X_test, y_test, "XGBoost")
dt_metrics  = compute_metrics(dt_pipeline,  X_test, y_test, "Decision Tree")

print("\n── XGBoost (default threshold 0.40) ──")
print(classification_report(y_test, (xgb_pipeline.predict_proba(X_test)[:, 1] >= 0.40).astype(int), target_names=["Normal (0)", "Flagged (1)"]))
print(f"── XGBoost (optimal threshold {optimal_thresh:.3f}) ──")
print(classification_report(y_test, (xgb_pipeline.predict_proba(X_test)[:, 1] >= optimal_thresh).astype(int), target_names=["Normal (0)", "Flagged (1)"]))
print("\n── Decision Tree ──")
print(classification_report(y_test, (dt_pipeline.predict_proba(X_test)[:, 1] >= 0.40).astype(int), target_names=["Normal (0)", "Flagged (1)"]))

# ── 10. SHAP values for XGBoost ──────────────────────────────────────────────
print("Computing SHAP values...")
# The ColumnTransformer outputs numeric-scaled columns first, then passthrough.
# The XGB estimator inside the pipeline sees this transformed feature matrix,
# so we pass the transformed data to TreeExplainer — not the raw X_test.
X_test_transformed = xgb_pipeline.named_steps["preprocess"].transform(X_test)
explainer          = shap.TreeExplainer(xgb_pipeline.named_steps["xgb"])
shap_values        = explainer.shap_values(X_test_transformed)

# For binary classification shap_values may be a list [class0, class1]
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

mean_shap = np.abs(sv).mean(axis=0)
# Feature name order after ColumnTransformer: numeric first, then passthrough
shap_feature_names = ALL_FEATURE_COLS  # NUMERIC_FEATURES + PASSTHROUGH_FEATURES
shap_importance = {
    feat: round(float(val), 4)
    for feat, val in sorted(zip(shap_feature_names, mean_shap), key=lambda x: -x[1])
}

# ── 11. Business impact estimates ────────────────────────────────────────────
total            = len(y_test)
audit_flag_count = int(sum(
    1 for p in xgb_pipeline.predict_proba(X_test)[:, 1]
    if p * 100 >= 70
))
std_review_count = total - audit_flag_count

business_impact = {
    "standard_review_pct": round((std_review_count / total) * 100, 1),
    "audit_flag_pct":      round((audit_flag_count  / total) * 100, 1),
    "false_negative_rate": xgb_metrics["false_negative_rate"],
    "false_positive_rate": xgb_metrics["false_positive_rate"],
    "total_test_records":  total,
}

# ── 12. Save everything to JSON ──────────────────────────────────────────────
output = {
    "xgb":              xgb_metrics,
    "decision_tree":    dt_metrics,
    "shap_importance":  shap_importance,
    "business_impact":  business_impact,
    "cv_summary":       cv_summary,
    "scaler_benchmark": benchmark_results,
    "optimal_threshold": round(optimal_thresh, 4),
}

with open("model_metrics.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✅ model_metrics.json saved.")

# ── 13. Save models ──────────────────────────────────────────────────────────
joblib.dump(xgb_pipeline, "xgb_model.pkl")
joblib.dump(dt_pipeline,  "dt_model.pkl")
print("✅ xgb_model.pkl and dt_model.pkl saved.")

