import pandas as pd
import numpy as np
import joblib
import json
import shap
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
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
    amounts < 500,   (has_receipt < 0.65).astype(int),   # cheap: sometimes no receipt
    np.where(amounts < 5_000,  (has_receipt < 0.82).astype(int),
             np.where(amounts < 50_000, (has_receipt < 0.90).astype(int),
                                        (has_receipt < 0.95).astype(int)))
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
risk += np.where(is_affidavit == 0, no_receipt_weight, 0)

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

# ── 4. Features / target ──────────────────────────────────────────────────────
FEATURE_COLS = [
    "Exp_TotalAmount", "IsAffidavit", "ExpCat_ID", "Dept_ID",
    "Submission_Gap", "Is_Weekend", "Is_Round_Number",
]

# Both pipelines use StandardScaler — keep all features as float
X = df[FEATURE_COLS].copy().astype(float)
y = df["Label"]

# ── 5. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 6. Train XGBoost — Pipeline: StandardScaler → XGBClassifier ──────────────
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
imbalance_ratio = neg_count / pos_count
print(f"Class imbalance ratio (scale_pos_weight): {imbalance_ratio:.2f}")

xgb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb",    XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        scale_pos_weight=imbalance_ratio,
        random_state=42, eval_metric="logloss", verbosity=0,
    )),
])
xgb_pipeline.fit(X_train, y_train)

# ── 7. Train Decision Tree — Pipeline: StandardScaler → DecisionTreeClassifier ─
dt_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("dt",     DecisionTreeClassifier(max_depth=6, random_state=42)),
])
dt_pipeline.fit(X_train, y_train)

# ── 8. Helper: compute all metrics ───────────────────────────────────────────
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

print("\n── XGBoost ──")
print(classification_report(y_test, xgb_pipeline.predict(X_test), target_names=["Normal (0)", "Flagged (1)"]))
print("\n── Decision Tree ──")
print(classification_report(y_test, dt_pipeline.predict(X_test), target_names=["Normal (0)", "Flagged (1)"]))

# ── 9. SHAP values for XGBoost ───────────────────────────────────────────────
print("Computing SHAP values...")
# Extract scaled test data and the underlying XGB estimator from the pipeline
X_test_scaled = xgb_pipeline.named_steps["scaler"].transform(X_test)
explainer     = shap.TreeExplainer(xgb_pipeline.named_steps["xgb"])
shap_values   = explainer.shap_values(X_test_scaled)

# For binary classification shap_values may be a list [class0, class1]
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

mean_shap = np.abs(sv).mean(axis=0)
shap_importance = {
    feat: round(float(val), 4)
    for feat, val in sorted(zip(FEATURE_COLS, mean_shap), key=lambda x: -x[1])
}

# ── 10. Business impact estimates ─────────────────────────────────────────────
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

# ── 11. Save everything to JSON ───────────────────────────────────────────────
output = {
    "xgb":             xgb_metrics,
    "decision_tree":   dt_metrics,
    "shap_importance": shap_importance,
    "business_impact": business_impact,
}

with open("model_metrics.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✅ model_metrics.json saved.")

# ── 12. Save models ───────────────────────────────────────────────────────────
joblib.dump(xgb_pipeline, "xgb_model.pkl")
joblib.dump(dt_pipeline,  "dt_model.pkl")
print("✅ xgb_model.pkl and dt_model.pkl saved.")

