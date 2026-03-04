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
        <style>
        /* Login page extras */
        .login-hero { animation: fadeInUp 0.6s cubic-bezier(.22,1,.36,1) both; }
        .login-card {
            background: rgba(17,24,39,0.9);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(45,73,122,0.5);
            border-radius: 20px;
            padding: 36px 32px;
            box-shadow: 0 8px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(59,130,246,0.06);
            animation: fadeInUp 0.55s cubic-bezier(.22,1,.36,1) 0.1s both;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="login-hero" style="text-align:center; padding:48px 0 28px 0;">
            <div style="font-size:3.4rem; margin-bottom:10px;
                        filter:drop-shadow(0 0 20px rgba(59,130,246,0.4));">💳</div>
            <h1 style="font-size:2.3rem; font-weight:900; margin:0;
                       background:linear-gradient(135deg,#f1f5f9 30%,#3b82f6);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                       background-clip:text;">
                SmartSpend AI
            </h1>
            <p style="color:#64748b; font-size:0.92rem; margin-top:10px; line-height:1.7;">
                Intelligent expense anomaly detection<br>
                <span style='color:#3b82f6; font-weight:600;'>Sign in to continue</span>
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
<style>
.hero-section {{ animation: fadeInUp 0.5s cubic-bezier(.22,1,.36,1) both; }}
.nav-card {{
    background: linear-gradient(145deg, #141a26 0%, #1a2235 100%);
    border: 1px solid rgba(45,73,122,0.6);
    border-radius: 18px;
    padding: 32px 24px;
    text-align: center;
    min-height: 180px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 10px;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.22s;
    cursor: pointer;
    animation: fadeInUp 0.45s ease both;
}}
.nav-card:hover {{
    border-color: rgba(59,130,246,0.55);
    box-shadow: 0 8px 36px rgba(59,130,246,0.16), inset 0 0 0 1px rgba(59,130,246,0.08);
    transform: translateY(-5px);
}}
.nav-card-icon {{ font-size: 2.4rem; filter: drop-shadow(0 0 10px rgba(59,130,246,0.3)); }}
.nav-card-title {{ font-size: 1.02rem; font-weight: 800; color: #f1f5f9; }}
.nav-card-desc  {{ font-size: 0.78rem; color: #64748b; line-height: 1.6; }}
</style>
<div class="hero-section" style="text-align:center; padding:44px 0 30px 0;">
    <div style="font-size:3.6rem; margin-bottom:12px;
                filter:drop-shadow(0 0 24px rgba(59,130,246,0.35));">💳</div>
    <h1 style="font-size:2.6rem; font-weight:900; margin:0;
               background:linear-gradient(135deg,#f1f5f9 30%,#60a5fa 70%,#818cf8);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               background-clip:text; letter-spacing:-0.03em;">
        SmartSpend AI
    </h1>
    <p style="color:#64748b; font-size:1rem; margin-top:12px;">
        Welcome back, <b style="color:#f1f5f9;">{full_name}</b>
        &nbsp;·&nbsp;
        <span style="color:{r_color}; font-weight:700;">{r_icon} {role}</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ── Navigation cards ──────────────────────────────────────────────────────────
_CARD = """
<div class="nav-card" style="animation-delay:{delay}s">
    <div class="nav-card-icon">{icon}</div>
    <div class="nav-card-title">{title}</div>
    <div class="nav-card-desc">{desc}</div>
</div>
"""

CARDS = [
    ("📋", "Submit Expense",
     "Fill in your expense details. The AI analyses risk before you save."),
    ("📁", "Reports Log",
     "View all reports. Managers approve or reject. "
     "Finance Officers process reimbursements."),
    ("📊", "System Performance",
     "Model accuracy, confusion matrix, SHAP feature importance, and business impact."),
]

c1, c2, c3 = st.columns(3, gap="large")
for col, (delay, (icon, title, desc)) in zip([c1, c2, c3], enumerate(CARDS)):
    col.markdown(_CARD.format(icon=icon, title=title, desc=desc, delay=delay*0.08), unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#1e2d45; font-size:0.8rem; margin-top:8px;'>"
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
    lgbm = mx["xgb"]

    st.markdown("### 🏆 Model at a Glance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Model",               "XGBoost")
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
