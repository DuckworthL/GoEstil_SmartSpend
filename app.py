"""
app.py — SmartSpend AI
Login gate + home dashboard.
All pages are blocked until the user authenticates.
"""

import streamlit as st
import json
import os

import db
from utils import DARK_CSS, render_user_sidebar

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartSpend AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(DARK_CSS, unsafe_allow_html=True)

# Seed demo users on first DB connection (silently skipped if already done)
if db.is_db_available():
    db.seed_demo_users()

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not st.session_state.get("logged_in"):

    # Hide sidebar navigation until the user is logged in
    st.markdown("""
    <style>
    [data-testid="stSidebarNav"]   { display: none !important; }
    [data-testid="stSidebar"]      { display: none !important; }
    section[data-testid="stSidebarContent"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # Centered login card
    _, center, _ = st.columns([1.4, 2, 1.4])
    with center:
        st.markdown("""
        <div style="text-align:center; padding:48px 0 36px 0;">
            <div style="font-size:3.2rem; margin-bottom:10px;">💳</div>
            <h1 style="font-size:2.2rem; font-weight:800; color:#f1f5f9; margin:0;">
                SmartSpend AI
            </h1>
            <p style="color:#64748b; font-size:0.9rem; margin-top:10px; line-height:1.6;">
                Intelligent expense anomaly detection<br>
                Sign in to continue
            </p>
        </div>
        """, unsafe_allow_html=True)

        if not db.is_db_available():
            st.error(
                "❌ Database not reachable. Run `db_setup.sql` and "
                "`db_add_users.sql` against **(localdb)\\\\MSSQLLocalDB** then restart.",
                icon="❌",
            )
            st.stop()

        with st.form("login_form"):
            username = st.text_input(
                "Username",
                placeholder="e.g. employee1",
                label_visibility="visible",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
            )
            login_btn = st.form_submit_button(
                "🔐  Sign In", use_container_width=True
            )

        if login_btn:
            if not username.strip() or not password.strip():
                st.error("⚠️ Please enter both username and password.")
            else:
                user = db.get_user(username.strip())
                if (
                    user
                    and user["Is_Active"]
                    and db.verify_password(password, user["Password_Hash"])
                ):
                    st.session_state.logged_in = True
                    st.session_state.user_id   = user["User_ID"]
                    st.session_state.username  = user["Username"]
                    st.session_state.full_name = user["Full_Name"]
                    st.session_state.role      = user["Role"]
                    st.rerun()
                else:
                    st.error(
                        "❌ Invalid username or password. Please try again.",
                        icon="❌",
                    )

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🧪 Demo Accounts"):
            st.markdown("""
| Username | Password | Role |
|---|---|---|
| `employee1` | `Employee@123` | Employee |
| `manager1` | `Manager@123` | Manager |
| `fo1` | `Finance@123` | Finance Officer |
""")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATED HOME
# ══════════════════════════════════════════════════════════════════════════════

render_user_sidebar()

role      = st.session_state["role"]
full_name = st.session_state["full_name"]

_ROLE_ICONS = {"Employee": "👤", "Manager": "🗂️", "Finance Officer": "🏦"}
_ROLE_BADGE = {
    "Employee":        "#3b82f6",
    "Manager":         "#f59e0b",
    "Finance Officer": "#22c55e",
}
r_icon  = _ROLE_ICONS.get(role, "👤")
r_color = _ROLE_BADGE.get(role, "#64748b")

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding:40px 0 28px 0;">
    <div style="font-size:3.4rem; margin-bottom:10px;">💳</div>
    <h1 style="font-size:2.5rem; font-weight:800; color:#f1f5f9; margin:0;">
        SmartSpend AI
    </h1>
    <p style="color:#64748b; font-size:1rem; margin-top:10px;">
        Welcome back, <b style="color:#f1f5f9;">{full_name}</b>
        &nbsp;·&nbsp;
        <span style="color:{r_color}; font-weight:600;">{r_icon} {role}</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ── Navigation cards ──────────────────────────────────────────────────────────
_CARD = """
<div style="background:#1a1f2e; border:1px solid #2d3561; border-radius:14px;
            padding:28px 20px; text-align:center; min-height:175px;
            display:flex; flex-direction:column;
            align-items:center; justify-content:center; gap:8px;">
    <div style="font-size:2.2rem;">{icon}</div>
    <div style="font-size:1rem; font-weight:700; color:#f1f5f9;">{title}</div>
    <div style="font-size:0.78rem; color:#64748b; line-height:1.55;">{desc}</div>
</div>
"""

CARDS = [
    ("📋", "Submit Expense",
     "Fill in your expense details. The AI analyses risk before you save."),
    ("📁", "Reports Log",
     "View all reports. Managers review AI risk and approve or reject. "
     "Finance Officers process reimbursements."),
    ("📊", "System Performance",
     "Model accuracy, confusion matrix, SHAP feature importance, and business impact."),
]

c1, c2, c3 = st.columns(3, gap="large")
for col, (icon, title, desc) in zip([c1, c2, c3], CARDS):
    col.markdown(_CARD.format(icon=icon, title=title, desc=desc), unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#334155; font-size:0.8rem; margin-top:8px;'>"
    "← Use the sidebar to navigate between pages"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Model at-a-glance ─────────────────────────────────────────────────────────
metrics_path = os.path.join(os.path.dirname(__file__), "model_metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        mx = json.load(f)
    lgbm = mx["lgbm"]

    st.markdown("### 🏆 Model at a Glance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Model",               "LightGBM")
    m2.metric("Accuracy",            f"{lgbm['accuracy']}%")
    m3.metric("AUC-ROC",             f"{lgbm['auc_roc']}%")
    m4.metric("Recall (Flagged)",    f"{lgbm['recall_rejected']}%")
    m5.metric("False Positive Rate", f"{lgbm['false_positive_rate']}%")

    st.caption(
        "Model trained on synthetic SmartSpend expense data. "
        "Metrics computed on held-out test split. "
        "Navigate to **System Performance** for the full breakdown."
    )
else:
    st.info(
        "📊 No metrics file found. Run `train_model.py` to generate performance data.",
        icon="📊",
    )

st.markdown("---")
st.caption(
    "⚖️ SmartSpend AI provides advisory risk scores only. "
    "All final approval and reimbursement decisions are made by authorised "
    "Managers and Finance Officers in accordance with company policy."
)
