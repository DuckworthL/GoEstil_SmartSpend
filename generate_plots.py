import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# PLOT STYLE
# =====================================================================
plt.rcParams.update({
    'figure.facecolor': '#F8F9FA',
    'axes.facecolor': '#FFFFFF',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
COLORS = {'high_risk': '#E74C3C', 'low_risk': '#2ECC71', 'xgb': '#3498DB', 'dt': '#E67E22'}

# =====================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# =====================================================================
# ── CONFIG: set path to your CSV file here ──────────────────────────
CSV_PATH = 'smartspend_synthetic_data.csv'   # <- change if needed
# ────────────────────────────────────────────────────────────────────

# Load CSV — no synthetic generation, reads real data only
df = pd.read_csv(CSV_PATH)

# Drop columns not used in modeling (tracking/date fields)
df = df.drop(columns=['Exp_ID', 'Exp_Date', 'SubmittedDate'], errors='ignore')

print(f"✅ Loaded '{CSV_PATH}' — {df.shape[0]} records, {df.shape[1]} columns")
print(f"   Columns used: {list(df.columns)}\n")

df["Log_Amount"]        = np.log1p(df["Exp_TotalAmount"])
df["HighAmt_NoReceipt"] = ((df["Exp_TotalAmount"] > 10_000) & (df["IsAffidavit"] == 1)).astype(int)
df["Amount_Tier"]       = np.digitize(df["Exp_TotalAmount"], bins=[1_000, 5_000, 50_000, 100_000]).astype(int)
df["SameDay_NoReceipt"] = ((df["Submission_Gap"] == 0) & (df["IsAffidavit"] == 1)).astype(int)
df["Gap_Ratio"]         = df["Submission_Gap"] / 60.0

NUMERIC     = ["Log_Amount", "Submission_Gap", "Gap_Ratio"]
PASSTHROUGH = ["IsAffidavit", "ExpCat_ID", "Dept_ID", "Is_Weekend", "Is_Round_Number",
               "HighAmt_NoReceipt", "Amount_Tier", "SameDay_NoReceipt"]

# =====================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================
print(f"{'='*75}\n{'EXPLORATORY DATA ANALYSIS (EDA)':^75}\n{'='*75}")

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
missing_vals = df.isnull().sum().sum()
print(f"Total Missing Values: {missing_vals}\n")

normal_count = (df['Label'] == 0).sum()
fraud_count  = (df['Label'] == 1).sum()
fraud_rate   = fraud_count / len(df) * 100
print(f"Target Distribution (Label):")
print(f"  - Normal (0): {normal_count} ({100 - fraud_rate:.1f}%)")
print(f"  - Fraud/Anomaly (1): {fraud_count} ({fraud_rate:.1f}%)\n")

print(f"Expense Amount Analysis:")
print(f"  - Avg Normal Expense: ₱{df[df['Label']==0]['Exp_TotalAmount'].mean():,.2f}")
print(f"  - Avg Fraud Expense:  ₱{df[df['Label']==1]['Exp_TotalAmount'].mean():,.2f}")
print(f"  - Max Expense Logged: ₱{df['Exp_TotalAmount'].max():,.2f}\n")

print(f"Fraud Rates by High-Risk Categories:")
affidavit_fraud = df[df['IsAffidavit']==1]['Label'].mean() * 100
weekend_fraud   = df[df['Is_Weekend']==1]['Label'].mean() * 100
round_fraud     = df[df['Is_Round_Number']==1]['Label'].mean() * 100
highamt_fraud   = df[df['HighAmt_NoReceipt']==1]['Label'].mean() * 100
print(f"  - Missing Receipt (Affidavit): {affidavit_fraud:.1f}% are Fraud")
print(f"  - Weekend Purchases:           {weekend_fraud:.1f}% are Fraud")
print(f"  - Exact Round Numbers:         {round_fraud:.1f}% are Fraud")
print(f"  - High Amount + No Receipt:    {highamt_fraud:.1f}% are Fraud\n")

print(f"Top Features Correlated with Fraud (Label):")
correlations = df[NUMERIC + PASSTHROUGH + ['Label']].corr()['Label'].drop('Label').sort_values(ascending=False)
for feature, corr_val in correlations.items():
    print(f"  - {feature:<20}: {corr_val:+.3f}")
print(f"{'='*75}\n")

# =====================================================================
# GRAPH 1 — CLASS DISTRIBUTION (Section 2.3)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 1: Target Variable — Class Distribution', fontsize=14, fontweight='bold', y=1.02)

# Bar chart
bars = axes[0].bar(['Low-Risk (0)\nCompliant', 'High-Risk (1)\nViolating'],
                   [normal_count, fraud_count],
                   color=[COLORS['low_risk'], COLORS['high_risk']], width=0.5, edgecolor='white', linewidth=1.5)
axes[0].set_title('Class Frequency', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Records')
for bar, count in zip(bars, [normal_count, fraud_count]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 f'{count}\n({count/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=11)
axes[0].set_ylim(0, max(normal_count, fraud_count) * 1.2)

# Pie chart
axes[1].pie([normal_count, fraud_count],
            labels=[f'Low-Risk\n{normal_count} ({100-fraud_rate:.1f}%)', f'High-Risk\n{fraud_count} ({fraud_rate:.1f}%)'],
            colors=[COLORS['low_risk'], COLORS['high_risk']],
            startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2),
            textprops={'fontsize': 11})
axes[1].set_title('Class Proportion', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('fig1_class_distribution.png', dpi=150, bbox_inches='tight')
print("✅ Saved: fig1_class_distribution.png")

# =====================================================================
# GRAPH 2 — OUTLIERS & FEATURE SCALES (Section 2.3)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Figure 2: Feature Scale Differences & Outliers', fontsize=14, fontweight='bold')

# Expense Amount by class
low_risk_amounts  = df[df['Label']==0]['Exp_TotalAmount']
high_risk_amounts = df[df['Label']==1]['Exp_TotalAmount']
bp = axes[0].boxplot([low_risk_amounts, high_risk_amounts],
                     patch_artist=True, widths=0.5,
                     boxprops=dict(linewidth=1.5),
                     medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor(COLORS['low_risk'] + '99')
bp['boxes'][1].set_facecolor(COLORS['high_risk'] + '99')
axes[0].set_xticklabels(['Low-Risk (0)', 'High-Risk (1)'])
axes[0].set_title('Expense Amount Distribution by Class', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Expense Amount (₱)')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₱{x:,.0f}'))

# Submission Gap by class
low_risk_gap  = df[df['Label']==0]['Submission_Gap']
high_risk_gap = df[df['Label']==1]['Submission_Gap']
bp2 = axes[1].boxplot([low_risk_gap, high_risk_gap],
                      patch_artist=True, widths=0.5,
                      boxprops=dict(linewidth=1.5),
                      medianprops=dict(color='black', linewidth=2))
bp2['boxes'][0].set_facecolor(COLORS['low_risk'] + '99')
bp2['boxes'][1].set_facecolor(COLORS['high_risk'] + '99')
axes[1].set_xticklabels(['Low-Risk (0)', 'High-Risk (1)'])
axes[1].set_title('Submission Gap (Days) by Class', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Days Between Expense & Submission')
axes[1].axhline(y=60, color='red', linestyle='--', alpha=0.7, label='60-Day Policy Limit')
axes[1].legend()

plt.tight_layout()
plt.savefig('fig2_outliers_scales.png', dpi=150, bbox_inches='tight')
print("✅ Saved: fig2_outliers_scales.png")

# =====================================================================
# 3. MODEL PREPARATION & TRAINING
# =====================================================================
X = df[NUMERIC + PASSTHROUGH].astype(float)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
imbalance = (y_train==0).sum() / (y_train==1).sum()

def make_preprocessor():
    return ColumnTransformer([("num", RobustScaler(), NUMERIC),
                              ("pass", "passthrough", PASSTHROUGH)], remainder="drop")

xgb_pipe = Pipeline([("prep", make_preprocessor()),
                     ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                           scale_pos_weight=imbalance, subsample=0.85,
                                           colsample_bytree=0.85, min_child_weight=3, gamma=0.1,
                                           random_state=42, eval_metric="logloss"))])

dt_pipe = Pipeline([("prep", make_preprocessor()),
                    ("clf", DecisionTreeClassifier(max_depth=6, random_state=42))])

xgb_pipe.fit(X_train, y_train)
dt_pipe.fit(X_train, y_train)

# =====================================================================
# 4. PREDICTIONS & EVALUATION (Threshold = 0.40)
# =====================================================================
thr   = 0.40
p_xgb = xgb_pipe.predict_proba(X_test)[:, 1]
p_dt  = dt_pipe.predict_proba(X_test)[:, 1]
y_xgb = (p_xgb >= thr).astype(int)
y_dt  = (p_dt  >= thr).astype(int)

def evaluate(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return [accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_prob),
            recall_score(y_true, y_pred), precision_score(y_true, y_pred),
            f1_score(y_true, y_pred), fn/(fn+tp), fp/(fp+tn)]

m_xgb = evaluate(y_test, y_xgb, p_xgb)
m_dt  = evaluate(y_test, y_dt,  p_dt)
metrics_names = ["Accuracy", "AUC-ROC", "Recall", "Precision", "F1", "FNR", "FPR"]

# =====================================================================
# 5. METRICS OUTPUT
# =====================================================================
print(f"{'Metric':<18} | {'XGBoost (Thr=0.40)':<25} | {'Decision Tree':<25}\n{'-'*75}")
winners = []
for n, val_x, val_d in zip(metrics_names, m_xgb, m_dt):
    print(f"{n:<18} | {val_x:<25.4f} | {val_d:<25.4f}")
    is_lower_better = n in ["FNR", "FPR"]
    winner = "XGBoost" if (val_x < val_d if is_lower_better else val_x > val_d) else "Decision Tree"
    winners.append(f"{n}: {winner}")

print(f"{'-'*75}\nWINNER SUMMARY: {', '.join(winners)}\n{'='*75}")
print(f"\n{'-'*25} XGBOOST REPORT {'-'*25}\n{classification_report(y_test, y_xgb)}")
print(f"{'-'*22} DECISION TREE REPORT {'-'*22}\n{classification_report(y_test, y_dt)}")

cm_xgb = confusion_matrix(y_test, y_xgb)
cm_dt  = confusion_matrix(y_test, y_dt)
print(f"\n{'-'*25} CONFUSION MATRICES {'-'*25}")
print(f"XGBoost:\n  TN: {cm_xgb[0,0]:<5} FP: {cm_xgb[0,1]:<5}\n  FN: {cm_xgb[1,0]:<5} TP: {cm_xgb[1,1]:<5}\n")
print(f"Decision Tree:\n  TN: {cm_dt[0,0]:<5} FP: {cm_dt[0,1]:<5}\n  FN: {cm_dt[1,0]:<5} TP: {cm_dt[1,1]:<5}\n")

# =====================================================================
# GRAPH 3 — CONFUSION MATRIX HEATMAPS (Section 5)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 3: Confusion Matrices — Threshold = 0.40', fontsize=14, fontweight='bold')

labels = [['TN', 'FP'], ['FN', 'TP']]
for ax, cm, title, color in zip(axes,
                                 [cm_xgb, cm_dt],
                                 ['XGBoost (AI-Recommended)', 'Decision Tree (Baseline)'],
                                 ['Blues', 'Oranges']):
    sns.heatmap(cm, annot=False, fmt='d', cmap=color, ax=ax,
                linewidths=2, linecolor='white', cbar=False)
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.35, labels[i][j],
                    ha='center', va='center', fontsize=13, color='grey', fontweight='bold')
            ax.text(j + 0.5, i + 0.65, str(cm[i, j]),
                    ha='center', va='center', fontsize=22, fontweight='bold',
                    color='white' if cm[i,j] > cm.max()*0.5 else 'black')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('Actual Label', fontsize=11)
    ax.set_xticklabels(['Low-Risk (0)', 'High-Risk (1)'])
    ax.set_yticklabels(['Low-Risk (0)', 'High-Risk (1)'], rotation=0)

plt.tight_layout()
plt.savefig('fig3_confusion_matrices.png', dpi=150, bbox_inches='tight')
print("✅ Saved: fig3_confusion_matrices.png")

# =====================================================================
# GRAPH 4 — ROC CURVE COMPARISON (Section 5)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle('Figure 4: ROC Curve Comparison', fontsize=14, fontweight='bold')

for y_prob, label, color, auc in [
    (p_xgb, 'XGBoost', COLORS['xgb'], m_xgb[1]),
    (p_dt,  'Decision Tree', COLORS['dt'], m_dt[1])
]:
    fpr_vals, tpr_vals, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr_vals, tpr_vals, color=color, linewidth=2.5, label=f'{label} (AUC = {auc:.4f})')

ax.plot([0,1],[0,1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
ax.fill_between(*roc_curve(y_test, p_xgb)[:2], alpha=0.08, color=COLORS['xgb'])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('ROC Curve — XGBoost vs. Decision Tree', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig('fig4_roc_curve.png', dpi=150, bbox_inches='tight')
print("✅ Saved: fig4_roc_curve.png")

# =====================================================================
# GRAPH 5 — METRICS BAR COMPARISON (Section 5)
# =====================================================================
compare_metrics = ['Accuracy', 'AUC-ROC', 'Recall', 'Precision', 'F1']
xgb_vals = m_xgb[:5]
dt_vals  = m_dt[:5]

x = np.arange(len(compare_metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars1 = ax.bar(x - width/2, xgb_vals, width, label='XGBoost', color=COLORS['xgb'], alpha=0.9, edgecolor='white')
bars2 = ax.bar(x + width/2, dt_vals,  width, label='Decision Tree', color=COLORS['dt'], alpha=0.9, edgecolor='white')

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

min_val = min(min(xgb_vals), min(dt_vals))
ax.set_ylim(max(0, min_val - 0.05), 1.02)
ax.set_xticks(x)
ax.set_xticklabels(compare_metrics, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Figure 5: Model Performance Comparison (Threshold = 0.40)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('fig5_metrics_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: fig5_metrics_comparison.png")

# =====================================================================
# GRAPH 6 — FEATURE IMPORTANCE (Section 8.2)
# =====================================================================
feature_names = NUMERIC + PASSTHROUGH
importances   = xgb_pipe.named_steps['clf'].feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=True)

# Business-friendly labels
label_map = {
    'Log_Amount': 'Log Amount',
    'Submission_Gap': 'Submission Gap (Days)',
    'Gap_Ratio': 'Gap Ratio (÷60)',
    'IsAffidavit': 'Missing Receipt\n(Affidavit)',
    'ExpCat_ID': 'Expense Category',
    'Dept_ID': 'Department',
    'Is_Weekend': 'Weekend Purchase',
    'Is_Round_Number': 'Round Number Amount',
    'HighAmt_NoReceipt': 'High Amount\n+ No Receipt',
    'Amount_Tier': 'Amount Tier',
    'SameDay_NoReceipt': 'Same-Day\nNo Receipt',
}
importance_df['Label'] = importance_df['Feature'].map(label_map)

fig, ax = plt.subplots(figsize=(10, 7))
colors_imp = [COLORS['high_risk'] if v >= importance_df['Importance'].quantile(0.7)
              else COLORS['xgb'] if v >= importance_df['Importance'].quantile(0.4)
              else '#95A5A6'
              for v in importance_df['Importance']]

bars = ax.barh(importance_df['Label'], importance_df['Importance'],
               color=colors_imp, edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, importance_df['Importance']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Feature Importance Score (Gain)', fontsize=12)
ax.set_title('Figure 6: XGBoost Feature Importance\n(Section 8.2 — Most Influential Predictors)', fontsize=13, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS['high_risk'], label='High Importance'),
                   Patch(facecolor=COLORS['xgb'],       label='Medium Importance'),
                   Patch(facecolor='#95A5A6',            label='Low Importance')]
ax.legend(handles=legend_elements, fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig('fig6_feature_importance.png', dpi=150, bbox_inches='tight')
print("✅ Saved: fig6_feature_importance.png")

# =====================================================================
# GRAPH 7 — FRAUD RATE BY RISK CATEGORY (EDA)
# =====================================================================
categories   = ['Missing Receipt\n(Affidavit)', 'Weekend\nPurchase', 'Round Number\nAmount', 'High Amt\n+ No Receipt']
fraud_rates  = [affidavit_fraud, weekend_fraud, round_fraud, highamt_fraud]

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = [COLORS['high_risk'] if r > 80 else COLORS['dt'] if r > 50 else COLORS['low_risk'] for r in fraud_rates]
bars = ax.bar(categories, fraud_rates, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
ax.axhline(y=fraud_rate, color='grey', linestyle='--', alpha=0.7, label=f'Overall Fraud Rate ({fraud_rate:.1f}%)')

for bar, rate in zip(bars, fraud_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylim(0, 115)
ax.set_ylabel('Fraud Rate (%)', fontsize=12)
ax.set_title('Figure 7: Fraud Rate by High-Risk Category', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('fig7_fraud_rates_by_category.png', dpi=150, bbox_inches='tight')
print("✅ Saved: fig7_fraud_rates_by_category.png")

# =====================================================================
# 5-FOLD CROSS VALIDATION OUTPUT
# =====================================================================
print(f"\n{'-'*25} 5-FOLD STRATIFIED CV {'-'*25}")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, model in [("XGBoost", xgb_pipe), ("Decision Tree", dt_pipe)]:
    auc_cv = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    rec_cv = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall')
    cv_results[name] = {'auc': auc_cv, 'rec': rec_cv}
    print(f"{name:<18} AUC-ROC: {auc_cv.mean():.4f} ± {auc_cv.std():.4f}")
    print(f"{'':<18} Recall:  {rec_cv.mean():.4f} ± {rec_cv.std():.4f}")

# =====================================================================
# GRAPH 8 — CROSS-VALIDATION STABILITY (Section 6.3)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 8: 5-Fold Cross-Validation Stability', fontsize=14, fontweight='bold')

fold_labels = [f'Fold {i+1}' for i in range(5)]
for ax, metric_key, metric_name in zip(axes, ['auc', 'rec'], ['AUC-ROC', 'Recall']):
    xgb_scores = cv_results['XGBoost'][metric_key]
    dt_scores  = cv_results['Decision Tree'][metric_key]
    ax.plot(fold_labels, xgb_scores, 'o-', color=COLORS['xgb'], linewidth=2.5, markersize=8, label=f'XGBoost (μ={xgb_scores.mean():.4f})')
    ax.plot(fold_labels, dt_scores,  's-', color=COLORS['dt'],  linewidth=2.5, markersize=8, label=f'Decision Tree (μ={dt_scores.mean():.4f})')
    ax.fill_between(range(5),
                    xgb_scores.mean() - xgb_scores.std(),
                    xgb_scores.mean() + xgb_scores.std(),
                    alpha=0.15, color=COLORS['xgb'])
    ax.set_title(f'{metric_name} per Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(max(0, min(xgb_scores.min(), dt_scores.min()) - 0.02), 1.02)

plt.tight_layout()
plt.savefig('fig8_cross_validation.png', dpi=150, bbox_inches='tight')
print("✅ Saved: fig8_cross_validation.png")

print(f"\n{'='*75}")
print(f"{'ALL 8 FIGURES SAVED SUCCESSFULLY':^75}")
print(f"{'='*75}")
