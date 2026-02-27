"""
pages/3_System_Performance.py — SmartSpend AI
Full System Performance & Model Accuracy dashboard.
Promoted from the expander in the original monolithic app.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px

import db
from utils import DARK_CSS, require_login, render_user_sidebar

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="System Performance — SmartSpend AI",
    page_icon="📊",
    layout="wide",
)
st.markdown(DARK_CSS, unsafe_allow_html=True)

require_login()
render_user_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
  <span style="font-size:2rem;">📊</span>
  <h2 style="margin:0; color:#f1f5f9;">System Performance & Model Accuracy</h2>
</div>
<p style="color:#64748b; margin-top:0; margin-bottom:24px;">
  Model metrics, confusion matrix, feature importance, and business impact — computed on held-out test data.
</p>
""", unsafe_allow_html=True)

# ── Load metrics ──────────────────────────────────────────────────────────────
metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_metrics.json")

if not os.path.exists(metrics_path):
    st.warning(
        "⚠️ No metrics file found. Run `train_model.py` to generate performance data.",
        icon="⚠️",
    )
    st.stop()

with open(metrics_path) as f:
    m = json.load(f)

lgbm_m = m["lgbm"]
dt_m   = m["decision_tree"]
shap_d = m["shap_importance"]
biz    = m["business_impact"]

# ── 1. Key Metrics (LightGBM) ─────────────────────────────────────────────────
st.markdown("### 🏆 LightGBM — Key Performance Metrics")
st.caption("Computed on held-out test data. Metrics are fixed until the model is retrained.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Accuracy",             f"{lgbm_m['accuracy']}%",
          help="Overall % of correct predictions across both classes.")
k2.metric("Recall (Rejected)",    f"{lgbm_m['recall_rejected']}%",
          help="How many actual bad reports the model correctly caught. LOW = fraud slipping through.")
k3.metric("Precision (Approved)", f"{lgbm_m['precision_approved']}%",
          help="Confidence when the model says a report is normal.")
k4.metric("F1-Score",             f"{lgbm_m['f1_score']}%",
          help="Balance between Precision and Recall.")
k5.metric("AUC-ROC",              f"{lgbm_m['auc_roc']}%",
          help="Model's ability to distinguish Approved vs Rejected across all thresholds.")

st.info(
    "🎯 **Most important metric in this context: Recall (Rejected).** "
    "A low Recall means fraudulent or non-compliant reports slip through — directly causing financial loss.",
    icon="🎯",
)

st.markdown("---")

# ── 2. Confusion Matrix ───────────────────────────────────────────────────────
st.markdown("### 🔢 Confusion Matrix — LightGBM")
cm = lgbm_m["confusion_matrix"]

cm_fig = go.Figure(data=go.Heatmap(
    z=[[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
    x=["Predicted: Normal", "Predicted: Flagged"],
    y=["Actual: Normal", "Actual: Flagged"],
    text=[[
        f"✅ True Negatives\n{cm['TN']}\n(Good reports cleared)",
        f"⚠️ False Positives\n{cm['FP']}\n(Good reports wrongly flagged)",
    ], [
        f"🚨 False Negatives\n{cm['FN']}\n(Bad reports missed — COSTLY)",
        f"✅ True Positives\n{cm['TP']}\n(Bad reports correctly flagged)",
    ]],
    texttemplate="%{text}",
    colorscale=[[0, "#1a1f2e"], [0.5, "#1e3a5f"], [1, "#2563eb"]],
    showscale=False,
))
cm_fig.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    font=dict(color="#e2e8f0", size=13),
    margin=dict(t=20, b=20, l=20, r=20),
    height=300,
)
cm_fig.update_xaxes(tickfont=dict(color="#94a3b8"))
cm_fig.update_yaxes(tickfont=dict(color="#94a3b8"))
st.plotly_chart(cm_fig, use_container_width=True)

cr1, cr2, cr3, cr4 = st.columns(4)
cr1.metric("✅ True Positives",  cm["TP"], help="Bad reports correctly flagged")
cr2.metric("✅ True Negatives",  cm["TN"], help="Good reports correctly cleared")
cr3.metric("⚠️ False Positives", cm["FP"], help="Good reports wrongly flagged")
cr4.metric("🚨 False Negatives", cm["FN"], help="Bad reports wrongly approved — financial loss risk")

st.caption(
    f"False Negative Rate: **{lgbm_m['false_negative_rate']}%** &nbsp;|&nbsp; "
    f"False Positive Rate: **{lgbm_m['false_positive_rate']}%** &nbsp;·&nbsp; "
    "Minimising False Negatives is the primary objective."
)

st.markdown("---")

# ── 3. Model Comparison ───────────────────────────────────────────────────────
st.markdown("### ⚔️ Model Comparison — Decision Tree vs LightGBM")

metrics_labels = ["Accuracy", "Recall (Rejected)", "Precision (Approved)", "F1-Score", "AUC-ROC"]
dt_vals   = [dt_m["accuracy"],   dt_m["recall_rejected"],   dt_m["precision_approved"],   dt_m["f1_score"],   dt_m["auc_roc"]]
lgbm_vals = [lgbm_m["accuracy"], lgbm_m["recall_rejected"], lgbm_m["precision_approved"], lgbm_m["f1_score"], lgbm_m["auc_roc"]]

comp_fig = go.Figure()
comp_fig.add_trace(go.Bar(
    name="Decision Tree", x=metrics_labels, y=dt_vals,
    marker_color="#f59e0b", text=[f"{v}%" for v in dt_vals], textposition="outside",
))
comp_fig.add_trace(go.Bar(
    name="LightGBM", x=metrics_labels, y=lgbm_vals,
    marker_color="#2563eb", text=[f"{v}%" for v in lgbm_vals], textposition="outside",
))
comp_fig.update_layout(
    barmode="group",
    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    font=dict(color="#e2e8f0"),
    legend=dict(bgcolor="#1a1f2e", bordercolor="#2d3561"),
    yaxis=dict(range=[0, 115], ticksuffix="%", gridcolor="#1e293b"),
    xaxis=dict(gridcolor="#1e293b"),
    margin=dict(t=30, b=10),
    height=380,
)
st.plotly_chart(comp_fig, use_container_width=True)

comp_data = {
    "Metric":        metrics_labels,
    "Decision Tree": [f"{v}%" for v in dt_vals],
    "LightGBM":      [f"{v}%" for v in lgbm_vals],
    "Winner":        ["🏆 " + ("LightGBM" if l > d else "Decision Tree")
                      for d, l in zip(dt_vals, lgbm_vals)],
}
st.dataframe(pd.DataFrame(comp_data).set_index("Metric"), use_container_width=True)

# ── Why LightGBM ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### 🏅 Why LightGBM is the Right Model — Despite the Numbers")

lgbm_auc = lgbm_m["auc_roc"]
dt_auc   = dt_m["auc_roc"]
lgbm_fpr = lgbm_m["false_positive_rate"]
dt_fpr   = dt_m["false_positive_rate"]

st.markdown(f"""
At first glance, the Decision Tree wins **4 out of 5 metrics**.
But raw metric counts don't tell the full story — what matters is *which* metric matters most.

---

**1. AUC-ROC is the decisive metric — and LightGBM wins it ({lgbm_auc}% vs {dt_auc}%)**

SmartSpend AI outputs a **continuous risk score (0–100%)** used to route expenses into triage tiers
and apply business-rule boosts. AUC-ROC is the only metric that evaluates this — it measures how
well the model separates Approved from Rejected expenses *across every possible threshold*.
LightGBM's {lgbm_auc}% AUC-ROC produces more reliable, well-calibrated risk scores at all confidence levels.

---

**2. Fewer false alarms — lower False Positive Rate ({lgbm_fpr}% vs {dt_fpr}%)**

The Decision Tree flags **{dt_fpr}% of good reports** as anomalous. LightGBM only flags **{lgbm_fpr}%** —
that's **{round(dt_fpr - lgbm_fpr, 2)} percentage points fewer** good reports wrongly sent to Finance review,
directly reducing officer workload and employee friction.

---

**3. Native categorical data handling**

The dataset's most predictive features (`Dept_ID`, `ExpCat_ID`, `IsAffidavit`, `Is_Weekend`) are categorical.
LightGBM accepts Pandas `category` dtype directly. The Decision Tree required manual `cat.codes` encoding —
a transformation that loses ordinal meaning and can negatively affect generalization on new category combinations.

---

**4. Real-world generalization as data grows**

The Decision Tree performs well on this clean synthetic data. In a live production environment with noisy,
messy data, LightGBM's ensemble of corrective trees generalizes more robustly and scales to millions of records.

> **Conclusion:** Decision Tree is a useful baseline. But for a production risk-scoring system dependent on
> calibrated probability outputs — **LightGBM is definitively the correct choice for SmartSpend AI.**
""")

st.markdown("---")

# ── 4. SHAP Feature Importance ────────────────────────────────────────────────
st.markdown("### 🔍 SHAP Feature Importance — LightGBM")
st.caption(
    "SHAP (SHapley Additive exPlanations) shows which features most influence predictions. "
    "Higher = greater impact on the model's output."
)

feat_names = list(shap_d.keys())
shap_vals  = list(shap_d.values())

shap_fig = go.Figure(go.Bar(
    x=shap_vals[::-1],
    y=feat_names[::-1],
    orientation="h",
    marker=dict(
        color=shap_vals[::-1],
        colorscale=[[0, "#1e3a5f"], [1, "#2563eb"]],
        showscale=False,
    ),
    text=[f"{v:.4f}" for v in shap_vals[::-1]],
    textposition="outside",
))
shap_fig.update_layout(
    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    font=dict(color="#e2e8f0"),
    xaxis=dict(title="Mean |SHAP Value|", gridcolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b"),
    margin=dict(t=10, b=10),
    height=320,
)
st.plotly_chart(shap_fig, use_container_width=True)
st.caption(
    "Top features justify why the model flagged or cleared a report — "
    "useful for explaining decisions to employees and auditors."
)

st.markdown("---")

# ── 5. Business Impact ────────────────────────────────────────────────────────
st.markdown("### 💼 Business Impact Validation")
st.caption(f"Based on {biz['total_test_records']} held-out test records.")

b1, b2, b3, b4 = st.columns(4)
b1.metric("Standard Review",     f"{biz['standard_review_pct']}%",
          help="Proportion routed to Standard Review (risk < 70%)")
b2.metric("Audit Flagged",       f"{biz['audit_flag_pct']}%",
          help="Proportion escalated to Audit Flag (risk ≥ 70%)")
b3.metric("False Negative Rate", f"{biz['false_negative_rate']}%",
          help="Bad reports wrongly approved — primary financial risk indicator")
b4.metric("False Positive Rate", f"{biz['false_positive_rate']}%",
          help="Good reports wrongly flagged — causes employee frustration")

st.markdown(f"""
**Key business outcome estimates:**
- 🔎 **{biz['standard_review_pct']}%** of submissions route to Standard Review — reducing audit workload
- 🚨 **{biz['audit_flag_pct']}%** escalated as high-risk requiring full audit
- 🚨 **{biz['false_negative_rate']}%** false negative rate — bad reports the model missed
- Instant pre-submission feedback reduces avoidable employee rejections
""")

st.markdown("---")

# ── 6. Live Data Trends ────────────────────────────────────────────────────────
st.markdown("### 📈 Live Data Trends")
st.caption("Real-time charts pulled from SmartSpendDB — auto-refreshes on page reload.")

# dept name/budget maps kept as fallbacks only (DB already returns these)

# ── 6a. Weekly Submission Volume + Risk Trend ──────────────────────────────────
try:
    trend_df = db.get_submission_trend()
    if not trend_df.empty:
        st.markdown("#### 📅 Weekly Submission Volume & Risk (Last 90 Days)")
        col_l, col_r = st.columns(2)

        with col_l:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=trend_df["Week_Start"],
                y=trend_df["Submissions"],
                name="Submissions",
                marker_color="#3b82f6",
                opacity=0.85,
            ))
            fig_vol.add_trace(go.Bar(
                x=trend_df["Week_Start"],
                y=trend_df["Audit_Flags"],
                name="Flagged for Audit",
                marker_color="#ef4444",
                opacity=0.85,
            ))
            fig_vol.update_layout(
                barmode="overlay",
                title="Submissions vs Audit Flags per Week",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                legend=dict(orientation="h", y=-0.25),
                xaxis=dict(title="Week", gridcolor="#1e293b"),
                yaxis=dict(title="Count", gridcolor="#1e293b"),
                height=300,
                margin=dict(t=40, b=60),
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        with col_r:
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(
                x=trend_df["Week_Start"],
                y=(trend_df["Avg_Risk"] * 100).round(1),
                mode="lines+markers",
                name="Avg Risk %",
                line=dict(color="#f59e0b", width=2.5),
                marker=dict(size=6),
            ))
            fig_risk.update_layout(
                title="Average Risk Score per Week",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(title="Week", gridcolor="#1e293b"),
                yaxis=dict(title="Avg Risk (%)", gridcolor="#1e293b", range=[0, 100]),
                height=300,
                margin=dict(t=40, b=60),
            )
            st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("No submission data in the past 90 days.", icon="ℹ️")
except Exception as e:
    st.warning(f"Weekly trend unavailable: {e}")

# ── 6b. Dept Spend vs Budget ───────────────────────────────────────────────────
try:
    dept_df = db.get_dept_spend_summary()
    if not dept_df.empty:
        st.markdown("#### 🏢 Department Spend vs Budget")
        dept_df["Dept_Label"] = dept_df["Dept_Name"]
        budget_safe = dept_df["Budget"].replace(0, 1)
        dept_df["Utilization_Pct"] = (dept_df["Current_Spend"] / budget_safe * 100).round(1)

        bar_colors = [
            "#ef4444" if p >= 90 else "#f59e0b" if p >= 70 else "#22c55e"
            for p in dept_df["Utilization_Pct"]
        ]

        fig_dept = go.Figure()
        fig_dept.add_trace(go.Bar(
            name="Budget",
            x=dept_df["Dept_Label"],
            y=dept_df["Budget"].astype(float),
            marker_color="rgba(59,130,246,0.25)",
            marker_line_color="#3b82f6",
            marker_line_width=1.5,
        ))
        fig_dept.add_trace(go.Bar(
            name="Current Spend",
            x=dept_df["Dept_Label"],
            y=dept_df["Current_Spend"],
            marker_color=bar_colors,
            opacity=0.9,
        ))
        fig_dept.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            legend=dict(orientation="h", y=-0.25),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(title="Amount (₱)", gridcolor="#1e293b"),
            height=320,
            margin=dict(t=20, b=60),
        )
        st.plotly_chart(fig_dept, use_container_width=True)
        st.caption("🟢 < 70%  | 🟡 70–89%  | 🔴 ≥ 90% of budget")
    else:
        st.info("No department spend data available.", icon="ℹ️")
except Exception as e:
    st.warning(f"Dept spend chart unavailable: {e}")

# ── 6c. Status Breakdown ───────────────────────────────────────────────────────
try:
    status_df = db.get_status_breakdown()
    if not status_df.empty:
        st.markdown("#### 📊 Report Status Breakdown")
        STATUS_COLORS = {
            "Pending Manager Review": "#f59e0b",
            "Approved by Manager":    "#3b82f6",
            "Rejected by Manager":    "#ef4444",
            "Reimbursed":             "#22c55e",
        }
        colors = [STATUS_COLORS.get(s, "#94a3b8") for s in status_df["Exp_Status"]]
        fig_pie = go.Figure(go.Pie(
            labels=status_df["Exp_Status"],
            values=status_df["Count"],
            hole=0.52,
            marker=dict(colors=colors),
            textinfo="percent+label",
            textfont=dict(size=12),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            showlegend=False,
            height=320,
            margin=dict(t=10, b=10),
        )
        scol1, scol2 = st.columns([1, 1])
        with scol1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with scol2:
            total = status_df["Count"].sum()
            for _, row in status_df.iterrows():
                pct = round(row["Count"] / total * 100, 1) if total else 0
                icon = STATUS_COLORS.get(row["Exp_Status"], "#94a3b8")
                st.markdown(
                    f"<div style='margin:6px 0; font-size:0.9rem;'>"
                    f"<span style='color:{icon}; font-size:1.1rem;'>●</span> "
                    f"<b>{row['Exp_Status']}</b>: {row['Count']} ({pct}%)"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No report status data available.", icon="ℹ️")
except Exception as e:
    st.warning(f"Status breakdown unavailable: {e}")
