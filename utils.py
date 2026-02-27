"""
utils.py — SmartSpend AI shared constants, helpers, and CSS
Imported by all pages.
"""

import joblib
import pandas as pd

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "lgbm_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)


# ── Label maps ────────────────────────────────────────────────────────────────
DEPT_LABELS = {
    101: "Sales",
    102: "Engineering",
    103: "Marketing",
    104: "Finance",
    105: "HR/Operations",
}

CAT_LABELS = {
    1: "Meals & Entertainment",
    2: "Transportation",
    3: "Accommodation",
    4: "Office Supplies",
    5: "Software/Subscriptions",
    6: "Training & Development",
    7: "Medical/Wellness",
    8: "Miscellaneous",
}

PAYMENT_METHODS = [
    "Cash",
    "Personal Credit Card",
    "Company Credit Card",
    "Bank Transfer",
    "GCash / PayMaya",
    "Petty Cash",
]

# ── Business-rule constants ───────────────────────────────────────────────────
MEALS_CAT_ID          = 1
MEALS_DAILY_LIMIT     = 5_000.0    # ₱ per day
SUBMISSION_LIMIT_DAYS = 60         # max days after purchase (1–60 valid)
LARGE_AMOUNT_CAUTION  = 50_000.0   # ₱ — elevated scrutiny
LARGE_AMOUNT_AUDIT    = 100_000.0  # ₱ — mandatory audit flag
ML_ANOMALY_THRESHOLD  = 40.0       # raw ML % at which model considers record anomalous
ML_ANOMALY_BOOST      = 30.0       # pts added to final score to bridge ML flag → UI triage

# ── Feature columns expected by the model ────────────────────────────────────
CAT_COLS = ["Dept_ID", "ExpCat_ID", "IsAffidavit", "Is_Weekend", "Is_Round_Number"]

# ── Report lifecycle status config ────────────────────────────────────────────
# Each status maps to (display_label, hex_colour, emoji)
STATUS_CONFIG: dict[str, tuple[str, str, str]] = {
    "Pending Manager Review": ("Pending Review",       "#f59e0b", "⏳"),
    "Approved by Manager":    ("Approved — Awaiting Reimbursement", "#22c55e", "✅"),
    "Rejected by Manager":    ("Rejected",             "#ef4444", "❌"),
    "Reimbursed":             ("Reimbursed",           "#2563eb", "💸"),
}

def status_color(status: str) -> str:
    return STATUS_CONFIG.get(status, ("", "#64748b", ""))[1]

def status_emoji(status: str) -> str:
    return STATUS_CONFIG.get(status, ("", "", "❓"))[2]

def status_label(status: str) -> str:
    return STATUS_CONFIG.get(status, (status, "", ""))[0]


# ── Business rules ────────────────────────────────────────────────────────────
def apply_business_rules(score: float, row: dict) -> tuple[float, list[tuple[str, int]]]:
    """
    Apply policy-based score boosts on top of the raw ML probability.
    Returns (capped_score, list_of_(label, boost) tuples).

    Rules:
      1. No receipt / affidavit filed                → +15 pts
      2. Future or same-day purchase (gap <= 0)      → +25 pts
      3. Late submission (gap > 60 days)             → +15 pts
      4. Purchase date on a weekend                  → +20 pts
      5. Meals expense > ₱5,000 daily limit          → +20 pts
      6. Round-number amount (x00.00)                → +15 pts
      7. High-value ≥ ₱50,000                        → +15 pts
         Very high-value ≥ ₱100,000                 → +25 pts (replaces #7)
    """
    boosts: list[tuple[str, int]] = []
    gap = int(row.get("SubmissionGap", 1))
    amt = float(row.get("Exp_TotalAmount", 0))

    # 1. No receipt
    if row.get("IsAffidavit", 0) == 1:
        score += 15
        boosts.append(("No Receipt — Affidavit Filed", 15))

    # 2. Future or same-day purchase date
    if gap < 0:
        score += 25
        boosts.append(("Future Purchase Date Detected", 25))
    elif gap == 0:
        score += 25
        boosts.append(("Same-Day Purchase — Expense Filed on Purchase Day", 25))

    # 3. Late submission
    elif gap > SUBMISSION_LIMIT_DAYS:
        score += 15
        boosts.append((f"Late Submission — {gap} Days (Limit: {SUBMISSION_LIMIT_DAYS})", 15))

    # 4. Weekend purchase date (no business activity on Sat/Sun)
    if row.get("Purchase_Is_Weekend", 0) == 1:
        score += 20
        boosts.append(("Purchase Date Falls on a Weekend", 20))

    # 5. Meals over daily limit
    if row.get("ExpCat_ID") == MEALS_CAT_ID and amt > MEALS_DAILY_LIMIT:
        score += 20
        boosts.append((f"Meals Expense > ₱{MEALS_DAILY_LIMIT:,.0f} Daily Limit", 20))

    # 6. Round-number amount
    if row.get("Is_Round_Number", 0) == 1:
        score += 15
        boosts.append(("Round-Number Amount — Possible Estimation", 15))

    # 7. High-value expense (tiered)
    if amt >= LARGE_AMOUNT_AUDIT:
        score += 25
        boosts.append((f"Very High-Value Expense ≥ ₱{LARGE_AMOUNT_AUDIT:,.0f}", 25))
    elif amt >= LARGE_AMOUNT_CAUTION:
        score += 15
        boosts.append((f"High-Value Expense ≥ ₱{LARGE_AMOUNT_CAUTION:,.0f}", 15))

    # 8. ML anomaly bridge — raw model probability crossed the 40 % anomaly threshold
    if score >= ML_ANOMALY_THRESHOLD:
        score += ML_ANOMALY_BOOST
        boosts.append((f"High ML Anomaly Probability (Base Score ≥ {ML_ANOMALY_THRESHOLD}%)", int(ML_ANOMALY_BOOST)))

    return min(score, 100.0), boosts


def get_triage(score: float) -> tuple[str, str]:
    """Return (triage_label, hex_colour)."""
    if score >= 70:
        return "AUDIT FLAG", "#ef4444"
    return "STANDARD REVIEW", "#f59e0b"


def get_risk_factors(row: dict, raw_ml_prob: float = 0.0) -> list[str]:
    """Return human-readable list of all active policy violations and risk signals."""
    factors: list[str] = []
    gap = int(row.get("SubmissionGap", 1))
    amt = float(row.get("Exp_TotalAmount", 0))

    # Submission window violations
    if gap < 0:
        factors.append(
            f"Future purchase date detected (gap: {gap} days) — "
            "purchases cannot be dated in the future"
        )
    elif gap == 0:
        factors.append(
            "Same-day expense claim — purchase and submission on the same day "
            "(minimum 1 day gap required)"
        )
    elif gap > SUBMISSION_LIMIT_DAYS:
        factors.append(
            f"Late submission — {gap} days after purchase "
            f"(policy window: 1–{SUBMISSION_LIMIT_DAYS} days)"
        )

    # Weekend purchase
    if row.get("Purchase_Is_Weekend", 0) == 1:
        factors.append(
            "Purchase date falls on Saturday or Sunday — "
            "no business activity expected on weekends"
        )

    # No receipt
    if row.get("IsAffidavit", 0) == 1:
        factors.append("No receipt on file — Affidavit of Loss required for verification")

    # Meals over limit
    if row.get("ExpCat_ID") == MEALS_CAT_ID and amt > MEALS_DAILY_LIMIT:
        factors.append(
            f"Meals expense ₱{amt:,.2f} exceeds ₱{MEALS_DAILY_LIMIT:,.0f} daily limit"
        )

    # Round number
    if row.get("Is_Round_Number", 0) == 1:
        factors.append(
            "Round-number amount — uncommon in real purchases, may indicate "
            "estimated or fabricated figures"
        )

    # High-value
    if amt >= LARGE_AMOUNT_AUDIT:
        factors.append(
            f"Very high-value expense ₱{amt:,.2f} (≥ ₱{LARGE_AMOUNT_AUDIT:,.0f}) — "
            "mandatory senior audit review required"
        )
    elif amt >= LARGE_AMOUNT_CAUTION:
        factors.append(
            f"High-value expense ₱{amt:,.2f} (≥ ₱{LARGE_AMOUNT_CAUTION:,.0f}) — "
            "elevated scrutiny and full documentation required"
        )

    # ML anomaly signal
    if raw_ml_prob >= ML_ANOMALY_THRESHOLD:
        factors.append(
            "LightGBM model detected high-risk anomalous patterns in the submission data (Score >= 40%)."
        )

    return factors


def predict_risk(row: dict) -> tuple[float, float, list[tuple[str, int]]]:
    """
    Run model + business rules on a feature dict.
    Returns (raw_ml_score_pct, final_score_pct, boosts).
    """
    model = load_model()

    df = pd.DataFrame([{
        "Exp_TotalAmount":  row["Exp_TotalAmount"],
        "IsAffidavit":      row["IsAffidavit"],
        "ExpCat_ID":        row["ExpCat_ID"],
        "Dept_ID":          row["Dept_ID"],
        "Submission_Gap":   row["SubmissionGap"],
        "Is_Weekend":       row.get("Is_Weekend", 0),
        "Is_Round_Number":  row.get("Is_Round_Number", 0),
    }])

    for col in CAT_COLS:
        df[col] = df[col].astype("category")

    raw_prob   = float(model.predict_proba(df)[0][1]) * 100.0
    final, boosts = apply_business_rules(raw_prob, row)
    return raw_prob, final, boosts


# ── Auth / session helpers ────────────────────────────────────────────────────
_ROLE_ICONS  = {"Employee": "👤", "Manager": "🗂️", "Finance Officer": "🏦"}
_ROLE_COLORS = {"Employee": "#3b82f6", "Manager": "#f59e0b", "Finance Officer": "#22c55e"}


def require_login() -> None:
    """
    Block page access if the user is not authenticated.
    Call at the very top of every page (after st.set_page_config).
    """
    import streamlit as st
    if not st.session_state.get("logged_in"):
        st.error("🔒 You must be signed in to access this page.")
        st.markdown("""
        <div style="text-align:center; margin-top:32px;">
            <a href="/" target="_self"
               style="background:#2563eb; color:#fff; padding:10px 28px;
                      border-radius:8px; text-decoration:none; font-weight:600;
                      font-size:0.95rem;">
                ← Back to Login
            </a>
        </div>
        """, unsafe_allow_html=True)
        st.stop()


def render_user_sidebar() -> None:
    """
    Render the logged-in user card + Sign Out button in the sidebar.
    Call once near the top of every page (after require_login).
    """
    import streamlit as st

    role      = st.session_state.get("role", "")
    full_name = st.session_state.get("full_name", "")
    username  = st.session_state.get("username", "")

    icon  = _ROLE_ICONS.get(role, "👤")
    color = _ROLE_COLORS.get(role, "#64748b")

    st.sidebar.markdown(f"""
    <div style="background:{color}18; border:1px solid {color}44; border-radius:12px;
                padding:14px 16px; margin-bottom:6px;">
        <div style="font-size:1rem; margin-bottom:4px;">
            {icon} <b style="color:#f1f5f9;">{full_name}</b>
        </div>
        <div style="font-size:0.72rem; color:{color}; font-weight:600;">{role}</div>
        <div style="font-size:0.68rem; color:#475569; margin-top:2px;">@{username}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🚪 Sign Out", use_container_width=True, key="__logout__"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("app.py")

    st.sidebar.markdown("---")


# ── Shared dark-theme CSS ─────────────────────────────────────────────────────
DARK_CSS = """
<style>
/* ── global reset ────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: #0f1117 !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] {
    background-color: #1a1f2e !important;
    border-right: 1px solid #2d3561;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── typography ──────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }
p, li, span, label       { color: #cbd5e1 !important; }
.stMarkdown              { color: #cbd5e1; }
code, pre                { background: #1e293b !important; color: #7dd3fc !important; }

/* ── metric cards ────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"]  { color: #94a3b8 !important; font-size: 0.78rem; }
[data-testid="stMetricValue"]  { color: #f1f5f9 !important; font-weight: 700; }

/* ── inputs ──────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
.stDateInput > div > div > input {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
}

/* ── buttons ─────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── approve/reject action buttons ──────────────────────────── */
.btn-approve > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
}
.btn-reject > button {
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
}

/* ── expanders ───────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #1a1f2e !important;
    border: 1px solid #2d3561 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
}
.streamlit-expanderContent {
    background: #0f1117 !important;
    border: 1px solid #2d3561 !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── data frames ─────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    border: 1px solid #2d3561;
}
.dataframe thead th {
    background: #1a1f2e !important;
    color: #94a3b8 !important;
}
.dataframe tbody td {
    background: #0f1117 !important;
    color: #e2e8f0 !important;
    border-color: #1e293b !important;
}

/* ── tabs ────────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    color: #94a3b8 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
}

/* ── divider ─────────────────────────────────────────────────── */
hr { border-color: #1e293b !important; }

/* ── info / warning / error boxes ───────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    background: #1a1f2e !important;
    border: 1px solid #2d3561 !important;
}

/* ── upload widget ───────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #1a1f2e !important;
    border: 1px dashed #334155 !important;
    border-radius: 8px !important;
}

/* ── toggle ──────────────────────────────────────────────────── */
[data-testid="stToggle"] label { color: #cbd5e1 !important; }

/* ── scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar               { width: 6px; height: 6px; }
::-webkit-scrollbar-track         { background: #0f1117; }
::-webkit-scrollbar-thumb         { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover   { background: #475569; }
</style>
"""
