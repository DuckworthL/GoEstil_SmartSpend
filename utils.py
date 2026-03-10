"""
utils.py — SmartSpend AI shared constants, helpers, and CSS
Imported by all pages.
"""

import math
import joblib
import pandas as pd

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "xgb_model.pkl"

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
    raw_ml_score = score          # preserve original ML probability for threshold check
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

    # 8. ML anomaly bridge — fires only when raw ML probability ≥ 40%
    if raw_ml_score >= ML_ANOMALY_THRESHOLD:
        score += ML_ANOMALY_BOOST
        boosts.append((f"High ML Anomaly Probability (Base Score ≥ {ML_ANOMALY_THRESHOLD}%)", int(ML_ANOMALY_BOOST)))

    return min(score, 100.0), boosts


def get_triage(score: float) -> tuple[str, str]:
    """Return (triage_label, hex_colour)."""
    if score >= 70:
        return "AUDIT FLAG", "#ef4444"
    if score >= 40:
        return "STANDARD REVIEW", "#f59e0b"
    return "LOW RISK", "#22c55e"


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
            "XGBoost model detected high-risk anomalous patterns in the submission data (Score >= 40%)."
        )

    return factors


def predict_risk(row: dict) -> tuple[float, float, list[tuple[str, int]]]:
    """
    Run model + business rules on a feature dict.
    Returns (raw_ml_score_pct, final_score_pct, boosts).

    The pipeline is: ColumnTransformer(RobustScaler on numeric, passthrough on
    binary/categorical) → XGBClassifier.  Engineered features must be computed
    here in exactly the same way as in train_model.py so inference is consistent
    with training.
    """
    model    = load_model()
    amount   = float(row["Exp_TotalAmount"])
    gap      = float(row["SubmissionGap"])
    is_aff   = float(row["IsAffidavit"])

    # ── Engineered features (must mirror train_model.py exactly) ──────────────
    # Log-transform: compresses ₱100–₱250k skew so RobustScaler centres well.
    log_amount = math.log1p(amount)

    # Gap ratio: gap relative to the 60-day policy window (0 = same-day, 1 = at limit).
    gap_ratio = gap / 60.0

    # Interaction: large amount + no receipt — strongest single audit signal.
    high_amt_no_rcpt = float(amount > 10_000 and is_aff == 1)

    # Ordinal tier: lets the model learn tier-specific patterns instead of
    # discovering all breakpoints from the raw (or log) amount alone.
    amount_tier = float(
        4 if amount >= 100_000 else
        3 if amount >= 50_000  else
        2 if amount >= 5_000   else
        1 if amount >= 1_000   else 0
    )

    # Interaction: same-day submission + no receipt — backdating red flag.
    same_day_no_rcpt = float(gap == 0 and is_aff == 1)

    # ── Build inference DataFrame (column names must match ColumnTransformer) ─
    df = pd.DataFrame([{
        # Numeric — RobustScaler applied by pipeline
        "Log_Amount":        log_amount,
        "Submission_Gap":    gap,
        "Gap_Ratio":         gap_ratio,
        # Passthrough — no scaling
        "IsAffidavit":       is_aff,
        "ExpCat_ID":         float(row["ExpCat_ID"]),
        "Dept_ID":           float(row["Dept_ID"]),
        "Is_Weekend":        float(row.get("Is_Weekend", 0)),
        "Is_Round_Number":   float(row.get("Is_Round_Number", 0)),
        "HighAmt_NoReceipt": high_amt_no_rcpt,
        "Amount_Tier":       amount_tier,
        "SameDay_NoReceipt": same_day_no_rcpt,
    }])

    raw_prob = float(model.predict_proba(df)[0][1]) * 100.0
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


# ── Shared dark-theme CSS + JS ────────────────────────────────────────────────
DARK_CSS = """
<style>
/* ══ CSS VARIABLES ══════════════════════════════════════════════ */
:root {
    --bg-base:      #0a0d14;
    --bg-surface:   #111827;
    --bg-card:      #141a26;
    --bg-elevated:  #1a2235;
    --border:       #1e2d45;
    --border-glow:  #2d4a7a;
    --accent:       #3b82f6;
    --accent-dark:  #1d4ed8;
    --accent-glow:  rgba(59,130,246,0.18);
    --green:        #22c55e;
    --green-glow:   rgba(34,197,94,0.15);
    --amber:        #f59e0b;
    --amber-glow:   rgba(245,158,11,0.15);
    --red:          #ef4444;
    --red-glow:     rgba(239,68,68,0.15);
    --text-primary: #f1f5f9;
    --text-muted:   #64748b;
    --text-sub:     #94a3b8;
}

/* ══ KEYFRAME ANIMATIONS ════════════════════════════════════════ */
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0);    }
}
@keyframes fadeIn {
    from { opacity:0; }
    to   { opacity:1; }
}
@keyframes slideInLeft {
    from { opacity:0; transform:translateX(-20px); }
    to   { opacity:1; transform:translateX(0);     }
}
@keyframes pulse {
    0%,100% { opacity:1; }
    50%      { opacity:0.55; }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position:  400px 0; }
}
@keyframes glow-ring {
    0%,100% { box-shadow: 0 0 0 0   var(--accent-glow); }
    50%      { box-shadow: 0 0 16px 4px var(--accent-glow); }
}
@keyframes spin-gauge {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes countUp {
    from { transform:scale(0.7); opacity:0; }
    to   { transform:scale(1);   opacity:1; }
}

/* ══ GLOBAL RESET ═══════════════════════════════════════════════ */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background-color: var(--bg-base) !important;
    color: #e2e8f0 !important;
}
[data-testid="stMainBlockContainer"] {
    animation: fadeIn 0.45s ease;
}

/* ══ SIDEBAR ════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1523 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
[data-testid="stSidebar"] input {
    background: #0f1523 !important;
    border-color: var(--border) !important;
}

/* ══ TYPOGRAPHY ═════════════════════════════════════════════════ */
h1, h2, h3 { color: var(--text-primary) !important; letter-spacing:-0.02em; }
h4, h5, h6 { color: var(--text-primary) !important; }
p, li, span, label { color: #cbd5e1 !important; }
.stMarkdown { color: #cbd5e1; }
code, pre {
    background: #1a2235 !important;
    color: #7dd3fc !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ══ METRIC CARDS ════════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s !important;
    animation: fadeInUp 0.4s ease both;
}
[data-testid="metric-container"]:hover {
    border-color: var(--border-glow) !important;
    box-shadow: 0 4px 24px rgba(59,130,246,0.12) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 800 !important;
    font-size: 1.5rem !important;
    animation: countUp 0.35s cubic-bezier(.34,1.56,.64,1) both;
}

/* ══ INPUT FIELDS ════════════════════════════════════════════════ */
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea,
.stNumberInput > div > div > input {
    background-color: var(--bg-elevated) !important;
    color: #e2e8f0 !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea  > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background-color: var(--bg-elevated) !important;
    color: #e2e8f0 !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
.stDateInput > div > div > input {
    background-color: var(--bg-elevated) !important;
    color: #e2e8f0 !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
}

/* ══ BUTTONS ═════════════════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.2s cubic-bezier(.34,1.56,.64,1) !important;
    box-shadow: 0 2px 12px rgba(37,99,235,0.25) !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 6px 24px rgba(37,99,235,0.4) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ── Coloured action button overrides ─────────── */
.btn-approve button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    box-shadow: 0 2px 12px rgba(22,163,74,0.3) !important;
}
.btn-approve button:hover { box-shadow: 0 6px 24px rgba(22,163,74,0.45) !important; }
.btn-reject button {
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    box-shadow: 0 2px 12px rgba(220,38,38,0.3) !important;
}
.btn-reject button:hover { box-shadow: 0 6px 24px rgba(220,38,38,0.45) !important; }
.btn-pay button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    box-shadow: 0 2px 12px rgba(37,99,235,0.3) !important;
    font-size: 1rem !important;
}
.btn-pay button:hover { box-shadow: 0 6px 24px rgba(37,99,235,0.45) !important; }
.btn-edit button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    box-shadow: 0 2px 12px rgba(124,58,237,0.3) !important;
}
.btn-edit button:hover { box-shadow: 0 6px 24px rgba(124,58,237,0.45) !important; }
.btn-submit button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    box-shadow: 0 2px 12px rgba(22,163,74,0.3) !important;
    font-size: 1rem !important; font-weight: 700 !important;
}
.btn-submit button:hover { box-shadow: 0 6px 24px rgba(22,163,74,0.45) !important; }
.btn-delete button {
    background: linear-gradient(135deg, #dc2626, #991b1b) !important;
    box-shadow: 0 2px 12px rgba(220,38,38,0.3) !important;
    font-weight: 700 !important;
}
.btn-delete button:hover { box-shadow: 0 6px 24px rgba(220,38,38,0.45) !important; }

/* ══ EXPANDERS ═══════════════════════════════════════════════════ */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-sub) !important;
    transition: border-color 0.2s, background 0.2s !important;
}
.streamlit-expanderHeader:hover {
    border-color: var(--border-glow) !important;
    background: var(--bg-elevated) !important;
}
.streamlit-expanderContent {
    background: var(--bg-base) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ══ DATA FRAMES ═════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
}
.dataframe thead th {
    background: var(--bg-elevated) !important;
    color: var(--text-sub) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
.dataframe tbody td {
    background: var(--bg-base) !important;
    color: #e2e8f0 !important;
    border-color: var(--border) !important;
    transition: background 0.15s !important;
}
.dataframe tbody tr:hover td { background: var(--bg-card) !important; }

/* ══ TABS ════════════════════════════════════════════════════════ */
[data-testid="stTabs"] {
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stTabs"] button {
    color: var(--text-muted) !important;
    border-radius: 8px 8px 0 0 !important;
    transition: color 0.2s, background 0.2s !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] button:hover {
    color: var(--text-sub) !important;
    background: var(--bg-elevated) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: rgba(59,130,246,0.07) !important;
}

/* ══ ALERT / INFO BOXES ══════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    animation: fadeInUp 0.3s ease !important;
}

/* ══ DIVIDER ═════════════════════════════════════════════════════ */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

/* ══ UPLOAD WIDGET ═══════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-glow) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    background: var(--bg-elevated) !important;
    border-color: var(--accent) !important;
}

/* ══ TOGGLE ══════════════════════════════════════════════════════ */
[data-testid="stToggle"] label { color: #cbd5e1 !important; }

/* ══ PROGRESS BAR ════════════════════════════════════════════════ */
[data-testid="stProgress"] > div {
    background: var(--border) !important;
    border-radius: 99px !important;
}
[data-testid="stProgress"] > div > div {
    border-radius: 99px !important;
    background: linear-gradient(90deg, var(--accent), #818cf8) !important;
    transition: width 0.6s ease !important;
}

/* ══ SCROLLBAR ═══════════════════════════════════════════════════ */
::-webkit-scrollbar               { width: 5px; height: 5px; }
::-webkit-scrollbar-track         { background: var(--bg-base); }
::-webkit-scrollbar-thumb         { background: #2d3f5e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover   { background: #3d5580; }

/* ══ HINT TEXT CLASS ═════════════════════════════════════════════ */
.click-hint {
    font-size: 0.78rem; color: var(--text-muted);
    margin-bottom: 8px; font-style: italic;
}

/* ══ CARD GLASS CLASS ════════════════════════════════════════════ */
.glass-card {
    background: rgba(20,26,38,0.85) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(45,73,122,0.5) !important;
    border-radius: 16px !important;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s !important;
}
.glass-card:hover {
    border-color: rgba(59,130,246,0.4) !important;
    box-shadow: 0 8px 32px rgba(59,130,246,0.12) !important;
    transform: translateY(-3px) !important;
}

/* ══ STATUS BADGE CLASS ══════════════════════════════════════════ */
.status-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}

/* ══ ANIMATED PULSE DOT ══════════════════════════════════════════ */
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--amber);
    animation: pulse 1.8s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
}
.pulse-dot.green  { background: var(--green); }
.pulse-dot.red    { background: var(--red); }
.pulse-dot.blue   { background: var(--accent); }

/* ══ SHIMMER SKELETON LOADER ═════════════════════════════════════ */
.shimmer {
    background: linear-gradient(90deg,
        var(--bg-card) 25%, var(--bg-elevated) 50%, var(--bg-card) 75%);
    background-size: 400px 100%;
    animation: shimmer 1.4s infinite;
    border-radius: 8px;
}

/* ══ PAGE SECTION ANIMATION ══════════════════════════════════════ */
.section-enter {
    animation: fadeInUp 0.45s cubic-bezier(.22,1,.36,1) both;
}
.section-enter:nth-child(2) { animation-delay: 0.06s; }
.section-enter:nth-child(3) { animation-delay: 0.12s; }
.section-enter:nth-child(4) { animation-delay: 0.18s; }
</style>
<script>
/* ── Animated counter for metric values ───────────────────────── */
(function() {
  function animateCounter(el) {
    const raw = el.innerText.replace(/[^0-9.]/g,'');
    const target = parseFloat(raw);
    if (isNaN(target) || target === 0) return;
    const prefix = el.innerText.match(/^[^0-9]*/)?.[0] || '';
    const suffix = el.innerText.match(/[^0-9.]+$/)?.[0] || '';
    const decimals = (raw.split('.')[1] || '').length;
    const duration = 600;
    const start = performance.now();
    function tick(now) {
      const t = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - t, 3);
      el.innerText = prefix + (target * ease).toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }
  function runCounters() {
    document.querySelectorAll('[data-testid="stMetricValue"]').forEach(animateCounter);
  }
  /* run on every Streamlit re-render */
  const obs = new MutationObserver(() => runCounters());
  obs.observe(document.body, { childList: true, subtree: true });
  runCounters();
})();

/* ── Row hover glow on data tables ────────────────────────────── */
(function() {
  function addTableHovers() {
    document.querySelectorAll('[data-testid="stDataFrame"] tbody tr').forEach(tr => {
      if (tr.dataset.hovInit) return;
      tr.dataset.hovInit = '1';
      tr.addEventListener('mouseenter', () => {
        tr.style.background = 'rgba(59,130,246,0.07)';
        tr.style.transition  = 'background 0.15s';
      });
      tr.addEventListener('mouseleave', () => {
        tr.style.background = '';
      });
    });
  }
  const obs2 = new MutationObserver(addTableHovers);
  obs2.observe(document.body, { childList: true, subtree: true });
  addTableHovers();
})();

/* ── Smooth scroll-into-view for alerts ───────────────────────── */
(function() {
  function watchAlerts() {
    document.querySelectorAll('[data-testid="stAlert"]').forEach(el => {
      if (el.dataset.scrolled) return;
      el.dataset.scrolled = '1';
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  }
  const obs3 = new MutationObserver(watchAlerts);
  obs3.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""
