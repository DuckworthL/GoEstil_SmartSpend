"""
pages/2_Reports_Log.py — SmartSpend AI
Clickable dataframe → modal dialog with full lifecycle detail & role actions.

CRUD (Manager only):
  • Edit Details  — modify any report field; AI score is re-computed automatically
  • Delete        — multi-select rows in the table → Delete Selected button

Roles:
  Employee        — read-only, track own report status
  Manager         — approve / reject / edit / delete
  Finance Officer — reimburse approved reports
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta

import db
from utils import (
    DARK_CSS, DEPT_LABELS, CAT_LABELS, PAYMENT_METHODS, get_triage,
    STATUS_CONFIG, status_color, status_emoji, status_label,
    apply_business_rules, predict_risk,
    require_login, render_user_sidebar,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reports Log — SmartSpend AI",
    page_icon="📁",
    layout="wide",
)
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ── Auth guard ────────────────────────────────────────────────────────────────
require_login()
render_user_sidebar()

role = st.session_state["role"]

# ── Extra CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.btn-approve button {
    background: linear-gradient(135deg,#16a34a,#15803d) !important;
    color:#fff !important; border:none !important;
}
.btn-reject button {
    background: linear-gradient(135deg,#dc2626,#b91c1c) !important;
    color:#fff !important; border:none !important;
}
.btn-pay button {
    background: linear-gradient(135deg,#2563eb,#1d4ed8) !important;
    color:#fff !important; border:none !important; font-size:1rem !important;
}
.btn-edit button {
    background: linear-gradient(135deg,#7c3aed,#6d28d9) !important;
    color:#fff !important; border:none !important;
}
.btn-delete button {
    background: linear-gradient(135deg,#dc2626,#991b1b) !important;
    color:#fff !important; border:none !important; font-weight:700 !important;
}
.click-hint { font-size:0.78rem; color:#64748b; margin-bottom:8px; font-style:italic; }
</style>
""", unsafe_allow_html=True)

# ── Role icons / guide ────────────────────────────────────────────────────────
ROLE_ICONS = {"Employee": "👤", "Manager": "🗂️", "Finance Officer": "🏦"}

ROLE_GUIDE = {
    "Employee":        ("#1e3a5f", "#3b82f6",
                        "👤 <b>Employee</b> — Read-only. Track your report status."),
    "Manager":         ("#3b1f00", "#f59e0b",
                        "🗂️ <b>Manager</b> — Click a report to Approve / Reject / Edit. "
                        "Select multiple rows to delete."),
    "Finance Officer": ("#14532d", "#22c55e",
                        "🏦 <b>Finance Officer</b> — Click an approved report to reimburse."),
}
g_bg, g_border, g_text = ROLE_GUIDE[role]
st.sidebar.markdown(f"""
<div style="background:{g_bg}44; border:1px solid {g_border};
            border-radius:10px; padding:12px 14px; margin-top:8px; font-size:0.82rem;">
    <span style="color:{g_border};">{g_text}</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filters")

# ── DB check ──────────────────────────────────────────────────────────────────
if not db.is_db_available():
    st.error(
        "❌ Database not reachable. Run `db_setup.sql` against "
        "**(localdb)\\\\MSSQLLocalDB** and restart.",
        icon="❌",
    )
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; gap:14px; margin-bottom:6px;
            animation:fadeInUp 0.45s cubic-bezier(.22,1,.36,1) both;">
  <div style="width:48px; height:48px; border-radius:14px;
              background:linear-gradient(135deg,#1d4ed8,#3b82f6);
              display:flex; align-items:center; justify-content:center;
              font-size:1.5rem; box-shadow:0 4px 16px rgba(59,130,246,0.3);">📁</div>
  <div>
    <h2 style="margin:0; color:#f1f5f9; font-size:1.55rem; font-weight:800;
               letter-spacing:-0.02em;">Reports Log</h2>
    <span style="font-size:0.82rem; color:#64748b;">
      {ROLE_ICONS[role]} Viewing as <b style="color:#94a3b8;">{role}</b>
      &nbsp;·&nbsp; Click any row to open full detail
      {"&nbsp;·&nbsp; Select multiple rows to delete" if role == "Manager" else ""}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
# Employees only see reports they submitted; Managers and FOs see all.
@st.cache_data(ttl=10)
def load_all(uid: int | None = None) -> pd.DataFrame:
    try:
        return db.get_all_reports(user_id=uid)
    except Exception:
        return pd.DataFrame()

_scope_uid = st.session_state.get("user_id") if role == "Employee" else None
df = load_all(_scope_uid)
if df.empty:
    if role == "Employee":
        st.info("You have no submitted expense reports yet. Head to **Submit Expense** to create one.", icon="📭")
    else:
        st.info("No expense reports yet. Submit one from the **Submit Expense** page.", icon="📭")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
DEFAULT_STATUS = {
    "Manager":         "Pending Manager Review",
    "Finance Officer": "Approved by Manager",
    "Employee":        "All",
}
all_statuses = ["All"] + sorted(df["Exp_Status"].dropna().unique().tolist())
default_idx  = all_statuses.index(DEFAULT_STATUS[role]) if DEFAULT_STATUS[role] in all_statuses else 0

sel_status = st.sidebar.selectbox("Status", all_statuses, index=default_idx)
sel_triage = st.sidebar.selectbox(
    "Triage Level",
    ["All"] + sorted(df["Triage_Level"].dropna().unique().tolist()),
)
dept_name_map = {v: k for k, v in DEPT_LABELS.items()}
dept_options  = ["All"] + [DEPT_LABELS.get(d, str(d)) for d in sorted(df["Dept_ID"].dropna().unique())]
sel_dept = st.sidebar.selectbox("Department", dept_options)

# ── Apply filters ─────────────────────────────────────────────────────────────
view = df.copy()
if sel_status != "All":
    view = view[view["Exp_Status"] == sel_status]
if sel_triage != "All":
    view = view[view["Triage_Level"] == sel_triage]
if sel_dept != "All" and sel_dept in dept_name_map:
    view = view[view["Dept_ID"] == dept_name_map[sel_dept]]

# ── KPI ribbon ────────────────────────────────────────────────────────────────
if role == "Employee":
    # Employee sees only their own KPIs scoped to their reports
    pend  = (df["Exp_Status"] == "Pending Manager Review").sum()
    apprd = (df["Exp_Status"] == "Approved by Manager").sum()
    rejd  = (df["Exp_Status"] == "Rejected by Manager").sum()
    reimb = (df["Exp_Status"] == "Reimbursed").sum()
else:
    # Manager and FO see company-wide KPIs (unfiltered)
    _all  = db.get_all_reports()
    pend  = (_all["Exp_Status"] == "Pending Manager Review").sum()
    apprd = (_all["Exp_Status"] == "Approved by Manager").sum()
    rejd  = (_all["Exp_Status"] == "Rejected by Manager").sum()
    reimb = (_all["Exp_Status"] == "Reimbursed").sum()
total_val = view["Exp_TotalAmount"].sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("⏳ Pending Review",  pend,  help="Awaiting manager decision")
k2.metric("✅ Approved",        apprd, help="Approved — in FO reimbursement queue")
k3.metric("❌ Rejected",        rejd,  help="Rejected by manager")
k4.metric("💸 Reimbursed",      reimb, help="Fully processed and paid")
k5.metric("💰 Filtered Value",  f"₱{total_val:,.0f}", help="Total amount in current view")

# ── CSV Export ────────────────────────────────────────────────────────────────
if not view.empty:
    _csv_cols = [
        "Report_ID", "Report_Title", "Dept_Name", "Cat_Name",
        "Exp_TotalAmount", "AI_Risk_Score", "Triage_Level", "Exp_Status",
        "Purchase_Date", "Submission_Date", "Submission_Gap",
        "Manager_Name", "Manager_Decision_Date",
        "Reimbursed_By", "Reimbursed_Date",
    ]
    _export_cols = [c for c in _csv_cols if c in view.columns]
    _csv_df = view[_export_cols].copy()
    # Ensure date columns export as clean YYYY-MM-DD strings (not blank timestamps)
    for _dc in ["Purchase_Date", "Submission_Date", "Manager_Decision_Date", "Reimbursed_Date"]:
        if _dc in _csv_df.columns:
            _csv_df[_dc] = pd.to_datetime(_csv_df[_dc], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    _csv_bytes = _csv_df.to_csv(index=False).encode("utf-8")
    _dl_col, _ = st.columns([1, 5])
    _dl_col.download_button(
        label="⬇️ Export CSV",
        data=_csv_bytes,
        file_name=f"smartspend_reports_{date.today()}.csv",
        mime="text/csv",
        help="Download current filtered view as CSV",
    )

st.markdown("---")
# ═══════════════════════════════════════════════════════════════════════════════

def _boosts_from_db_row(row) -> list:
    """Re-derive which business rules fired from a stored DB row."""
    purchase_date_raw = row.get("Purchase_Date")
    purchase_is_weekend = 0
    if purchase_date_raw is not None:
        try:
            if hasattr(purchase_date_raw, "weekday"):
                purchase_is_weekend = int(purchase_date_raw.weekday() >= 5)
            else:
                pd_obj = datetime.strptime(str(purchase_date_raw)[:10], "%Y-%m-%d")
                purchase_is_weekend = int(pd_obj.weekday() >= 5)
        except Exception:
            purchase_is_weekend = 0

    rule_row = {
        "SubmissionGap":       int(row.get("Submission_Gap", 0)),
        "IsAffidavit":         int(row.get("IsAffidavit", 0)),
        "Purchase_Is_Weekend": purchase_is_weekend,
        "ExpCat_ID":           int(row.get("ExpCat_ID", 0)),
        "Exp_TotalAmount":     float(row.get("Exp_TotalAmount", 0)),
        "Is_Round_Number":     int(row.get("Is_Round_Number", 0)),
    }
    _, boosts = apply_business_rules(0.0, rule_row)
    return boosts


# ═══════════════════════════════════════════════════════════════════════════════
# MODAL DIALOG — full report detail + role action + CRUD
# ═══════════════════════════════════════════════════════════════════════════════
@st.dialog("📋 Expense Report Detail", width="large")
def show_report_modal(rid: int):
    try:
        row = db.get_report_by_id(rid)
    except Exception as exc:
        st.error(f"Failed to load report #{rid}: {exc}")
        return
    if row is None:
        st.error(f"Report #{rid} not found.")
        return

    exp_status    = row.get("Exp_Status", "")
    risk_score    = float(row.get("AI_Risk_Score") or 0)
    triage_lbl, triage_color = get_triage(risk_score)
    # Risk gauge colour — independent of triage threshold
    risk_color = "#22c55e" if risk_score < 40 else "#f59e0b" if risk_score < 70 else "#ef4444"
    s_color = status_color(exp_status)
    s_emoji = status_emoji(exp_status)
    s_lbl   = status_label(exp_status)

    mgr_name        = row.get("Manager_Name") or None
    mgr_date        = str(row.get("Manager_Decision_Date", ""))[:10] if row.get("Manager_Decision_Date") else None
    reimbursed_date = str(row.get("Reimbursed_Date", ""))[:10] if row.get("Reimbursed_Date") else None
    reimbursed_by   = row.get("Reimbursed_By") or None
    submitted_date  = str(row.get("Submission_Date", ""))[:10] or "—"
    is_rejected     = exp_status == "Rejected by Manager"

    # ── Header card ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#1a1f2e; border:1px solid #2d3561; border-radius:14px;
                padding:18px 22px; margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between;
                    align-items:flex-start; flex-wrap:wrap; gap:10px;">
            <div>
                <div style="font-size:0.68rem; color:#64748b; text-transform:uppercase;
                            letter-spacing:0.1em; font-weight:600;">📄 Report #{rid}</div>
                <div style="font-size:1.18rem; font-weight:800; color:#f1f5f9; margin-top:5px;
                            letter-spacing:-0.01em;">
                    {row["Report_Title"]}
                </div>
                <div style="color:#64748b; font-size:0.8rem; margin-top:6px;">
                    <span style="background:#1e2d45; padding:2px 8px; border-radius:5px;
                                 color:#94a3b8; font-size:0.75rem;">{row["Dept_Name"]}</span>
                    &nbsp;
                    <span style="background:#1e2d45; padding:2px 8px; border-radius:5px;
                                 color:#94a3b8; font-size:0.75rem;">{row["Cat_Name"]}</span>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.65rem; font-weight:900; color:#f1f5f9;
                            letter-spacing:-0.02em;">
                    ₱{float(row["Exp_TotalAmount"]):,.2f}
                </div>
                <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">
                    Purchased: {str(row["Purchase_Date"])[:10]}
                </div>
                <div style="margin-top:8px;">
                    <span style="background:{s_color}1a; color:{s_color};
                                 border:1px solid {s_color}55; border-radius:99px;
                                 padding:4px 14px; font-size:0.72rem; font-weight:700;
                                 letter-spacing:0.03em;">
                        {s_emoji} {s_lbl}
                    </span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── AI Risk Assessment ─────────────────────────────────────────────────────
    st.markdown("#### 🤖 AI Risk Assessment")
    r1, r2, r3 = st.columns([1, 1, 2])

    with r1:
        # Animated circular gauge
        _circ_bg  = f"conic-gradient({risk_color} {risk_score*3.6:.0f}deg, #1a2235 0deg)"
        st.markdown(f"""
        <div style="position:relative; width:112px; height:112px; margin:0 auto;
                    animation:fadeIn 0.5s ease 0.1s both;">
          <div style="position:absolute; inset:0; border-radius:50%;
                      background:{_circ_bg}; box-shadow:0 0 20px {risk_color}33;"></div>
          <div style="position:absolute; inset:8px; border-radius:50%;
                      background:#0a0d14; display:flex; flex-direction:column;
                      align-items:center; justify-content:center;">
            <div style="font-size:1.45rem; font-weight:900; color:{risk_color};
                        line-height:1; letter-spacing:-0.02em;">{risk_score:.0f}%</div>
            <div style="font-size:0.5rem; color:#64748b; letter-spacing:0.1em;
                        margin-top:3px; text-transform:uppercase;">AI RISK</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        t_icon = "🚨" if triage_lbl == "AUDIT FLAG" else "⚠️" if triage_lbl == "STANDARD REVIEW" else "✅"
        st.markdown(f"""
        <div style="background:{triage_color}18; border:1px solid {triage_color};
                    border-radius:10px; padding:12px 14px; text-align:center;">
            <div style="font-size:0.66rem; color:#94a3b8; text-transform:uppercase;
                        letter-spacing:0.08em;">Triage Level</div>
            <div style="font-size:0.95rem; font-weight:700; color:{triage_color}; margin-top:6px;">
                {t_icon} {triage_lbl}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        if role == "Manager":
            if risk_score >= 70:
                ib, ibr, it, im = "#7c2d1222","#dc2626","#fca5a5","🚨 High risk — scrutinise all documentation before approving."
            elif risk_score >= 40:
                ib, ibr, it, im = "#78350f22","#f59e0b","#fde68a","⚠️ Moderate risk — verify receipt and justification."
            else:
                ib, ibr, it, im = "#14532d22","#22c55e","#86efac","✅ Low risk — safe to approve if documents are complete."
            st.markdown(f"""
            <div style="background:{ib}; border:1px solid {ibr}; border-radius:10px;
                        padding:10px 14px; font-size:0.81rem; color:{it}; line-height:1.55;">
                {im}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#1a1f2e; border:1px solid #2d3561; border-radius:10px;
                        padding:10px 14px; font-size:0.81rem; color:#94a3b8; line-height:1.55;">
                <b style="color:#f1f5f9;">AI Score: {risk_score:.1f}%</b><br>
                Model probability + business-rule boosts combined.
            </div>
            """, unsafe_allow_html=True)

    # ── Business-rule boost breakdown ─────────────────────────────────────────
    boosts = _boosts_from_db_row(row)
    if boosts:
        ml_base = max(0.0, risk_score - sum(b for _, b in boosts))
        items_html = "".join(
            f'<div style="display:flex; justify-content:space-between; '
            f'padding:5px 0; border-bottom:1px solid #1e293b; font-size:0.79rem;">'
            f'<span style="color:#cbd5e1;">{label}</span>'
            f'<span style="color:#f59e0b; font-weight:700;">+{pts} pts</span></div>'
            for label, pts in boosts
        )
        st.markdown(f"""
        <div style="background:#0f1117; border:1px solid #2d3561; border-radius:10px;
                    padding:12px 16px; margin-top:10px;">
            <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:8px;">
                ⚡ Policy Rule Boosts  &nbsp;
                <span style="color:#94a3b8;">ML base: {ml_base:.0f}%</span>
            </div>
            {items_html}
            <div style="display:flex; justify-content:space-between; padding:6px 0;
                        font-size:0.82rem; font-weight:700;">
                <span style="color:#f1f5f9;">Total boost added</span>
                <span style="color:#ef4444;">+{sum(b for _, b in boosts)} pts</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        ml_base = risk_score
        st.markdown("""
        <div style="background:#0f1117; border:1px solid #2d3561; border-radius:10px;
                    padding:10px 14px; margin-top:10px; font-size:0.8rem; color:#64748b;">
            ✅ No policy rule violations detected — score reflects ML model probability only.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Lifecycle Timeline ─────────────────────────────────────────────────────
    st.markdown("#### 📅 Lifecycle Timeline")

    # Step 2 — Manager decision
    if mgr_date:
        s2_done  = True
        s2_color = "#22c55e" if not is_rejected else "#ef4444"
        s2_icon  = "✓" if not is_rejected else "✕"
        s2_label = "Manager Approved" if not is_rejected else "Manager Rejected"
        s2_sub   = mgr_date + (f"<br><span style='color:#94a3b8;'>{mgr_name}</span>" if mgr_name else "")
    else:
        s2_done, s2_color, s2_icon = False, "#f59e0b", "2"
        s2_label, s2_sub = "Manager Review", "<span style='color:#f59e0b;'>Awaiting decision</span>"

    # Step 3 — Reimbursement
    if is_rejected:
        s3_done, s3_color, s3_icon = False, "#475569", "3"
        s3_label, s3_sub = "Reimbursement", "<span style='color:#475569;'>Not applicable</span>"
    elif reimbursed_date:
        s3_done, s3_color, s3_icon = True, "#2563eb", "✓"
        s3_label = "Reimbursed"
        s3_sub   = reimbursed_date + (f"<br><span style='color:#94a3b8;'>{reimbursed_by}</span>" if reimbursed_by else "")
    elif exp_status == "Approved by Manager":
        s3_done, s3_color, s3_icon = False, "#f59e0b", "3"
        s3_label, s3_sub = "Reimbursement", "<span style='color:#f59e0b;'>In FO queue</span>"
    else:
        s3_done, s3_color, s3_icon = False, "#334155", "3"
        s3_label, s3_sub = "Reimbursement", "<span style='color:#334155;'>Pending approval</span>"

    line1_color = s2_color if s2_done else "#2d3561"
    line2_color = s3_color if s3_done else "#2d3561"

    def _step(num_or_tick, color, done, label, sub):
        bg  = color if done else "#1e293b"
        bdr = color
        txt = "#fff" if done else color
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;flex:1;min-width:0;">'
            f'<div style="width:40px;height:40px;border-radius:50%;background:{bg};border:2px solid {bdr};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:1.05rem;font-weight:800;color:{txt};box-shadow:0 0 0 4px {color}22;">'
            f'{num_or_tick}</div>'
            f'<div style="font-size:0.72rem;font-weight:700;color:{color};margin-top:8px;text-align:center;line-height:1.4;">{label}</div>'
            f'<div style="font-size:0.67rem;color:#64748b;margin-top:3px;text-align:center;line-height:1.5;">{sub}</div>'
            f'</div>'
        )

    s1_html = _step("✓", "#22c55e", True,  "Submitted",  submitted_date)
    s2_html = _step(s2_icon, s2_color, s2_done, s2_label, s2_sub)
    s3_html = _step(s3_icon, s3_color, s3_done, s3_label, s3_sub)

    st.markdown(f"""
    <div style="background:#0f1623; border:1px solid #1e293b; border-radius:14px;
                padding:24px 20px 20px; margin-bottom:16px;">
      <div style="font-size:0.65rem; color:#475569; text-transform:uppercase;
                  letter-spacing:0.1em; margin-bottom:18px;">Report lifecycle</div>
      <div style="display:flex; align-items:flex-start; gap:0;">
        {s1_html}
        <div style="flex:0 0 auto; width:60px; height:2px; margin-top:20px;
                    background:linear-gradient(90deg,#22c55e,{line1_color});
                    border-radius:2px;"></div>
        {s2_html}
        <div style="flex:0 0 auto; width:60px; height:2px; margin-top:20px;
                    background:linear-gradient(90deg,{line1_color},{line2_color});
                    border-radius:2px;"></div>
        {s3_html}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Expense details + documentation ───────────────────────────────────────
    det_col, doc_col = st.columns([3, 2])

    with det_col:
        st.markdown("**Expense Details**")
        st.markdown(f"""
        <div style="background:#0f1117; border:1px solid #1e293b; border-radius:10px;
                    padding:14px 18px; font-size:0.84rem; line-height:2.1;">
            <span style="color:#64748b;">Seller / Merchant</span><br>
            <span style="color:#e2e8f0;">{row.get("Seller") or "—"}</span><br>
            <span style="color:#64748b;">Payment Method</span><br>
            <span style="color:#e2e8f0;">{row.get("Payment_Method") or "—"}</span><br>
            <span style="color:#64748b;">Submission Gap</span><br>
            <span style="color:#e2e8f0;">{row.get("Submission_Gap", 0)} days after purchase</span>
        </div>
        """, unsafe_allow_html=True)
        if row.get("Description"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#0f1117; border:1px solid #1e293b; border-radius:10px;
                        padding:12px 16px; font-size:0.82rem; color:#94a3b8;">
                <b style="color:#64748b; font-size:0.74rem;">NOTES</b><br>
                <span style="color:#cbd5e1;">{row["Description"]}</span>
            </div>
            """, unsafe_allow_html=True)

    with doc_col:
        st.markdown("**Documentation**")
        if row.get("IsAffidavit"):
            st.markdown("""
            <div style="background:#7c2d1222; border:1px solid #dc2626; border-radius:10px;
                        padding:14px 16px; margin-bottom:12px;">
                <div style="color:#fca5a5; font-weight:700; font-size:0.86rem;">
                    📄 No Receipt — Affidavit Filed
                </div>
                <div style="color:#94a3b8; font-size:0.76rem; margin-top:6px; line-height:1.5;">
                    Employee declared receipt loss. Verify Affidavit with HR.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#14532d22; border:1px solid #16a34a; border-radius:10px;
                        padding:14px 16px; margin-bottom:12px;">
                <div style="color:#86efac; font-weight:700; font-size:0.86rem;">
                    🧾 Receipt on File
                </div>
                <div style="color:#94a3b8; font-size:0.76rem; margin-top:6px; line-height:1.5;">
                    Supporting receipt is attached to this report.
                </div>
            </div>
            """, unsafe_allow_html=True)

        try:
            dept_spent  = db.get_dept_spend(int(row["Dept_ID"]))
            dept_budget = db.get_dept_budget(int(row["Dept_ID"]))
            pct  = min(dept_spent / dept_budget * 100, 100) if dept_budget > 0 else 0
            bclr = "#ef4444" if pct >= 90 else "#f59e0b" if pct >= 70 else "#22c55e"
            st.markdown(f"""
            <div>
                <div style="display:flex; justify-content:space-between;
                            font-size:0.72rem; color:#94a3b8; margin-bottom:4px;">
                    <span>Dept Budget</span><span style="color:{bclr};">{pct:.1f}% used</span>
                </div>
                <div style="background:#1e293b; border-radius:6px; height:9px; overflow:hidden;">
                    <div style="background:{bclr}; width:{pct:.1f}%; height:100%;
                                border-radius:6px;"></div>
                </div>
                <div style="display:flex; justify-content:space-between;
                            font-size:0.7rem; color:#475569; margin-top:3px;">
                    <span>₱{dept_spent:,.0f}</span><span>₱{dept_budget:,.0f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

    # ── Rejection reason card ──────────────────────────────────────────────────
    if is_rejected and row.get("Rejection_Reason"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#7c2d1222; border:1px solid #dc2626; border-radius:12px;
                    padding:16px 20px;">
            <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:8px;">❌ Rejection Reason</div>
            <div style="color:#fca5a5; font-size:0.88rem; line-height:1.65;">
                {row["Rejection_Reason"]}
            </div>
            <div style="font-size:0.72rem; color:#64748b; margin-top:10px;">
                By {mgr_name or "—"} on {mgr_date or "—"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # ROLE ACTION PANELS
    # ═══════════════════════════════════════════════════════════════════════════

    # ── MANAGER ───────────────────────────────────────────────────────────────
    if role == "Manager":
        mgr_name_in = st.session_state.get("full_name", "")

        if exp_status == "Pending Manager Review":
            st.markdown("#### 🗂️ Manager Actions")
            st.info(f"Acting as: **{mgr_name_in}**", icon="👤")

            tab_approve, tab_reject, tab_edit = st.tabs([
                "✅  Approve", "❌  Reject", "✏️  Edit Details",
            ])

            # ── Approve tab ────────────────────────────────────────────────────
            with tab_approve:
                st.markdown("""
                <div style="background:#14532d22; border:1px solid #16a34a; border-radius:10px;
                            padding:14px 18px; font-size:0.83rem; color:#86efac;
                            margin-bottom:12px; line-height:1.65;">
                    <b>Approval checklist:</b><br>
                    ✔ Expense is business-related and properly documented<br>
                    ✔ Amount is reasonable for the category and budget<br>
                    ✔ Receipt or affidavit is on file<br>
                    ✔ Submission is within the 61-day policy window
                </div>
                """, unsafe_allow_html=True)
                # ── Dept budget alert ──────────────────────────
                try:
                    _d_spent  = db.get_dept_spend(int(row["Dept_ID"]))
                    _d_budget = db.get_dept_budget(int(row["Dept_ID"]))
                    _d_pct    = (_d_spent / _d_budget * 100) if _d_budget > 0 else 0
                    _after    = (_d_spent + float(row["Exp_TotalAmount"])) / _d_budget * 100 if _d_budget > 0 else 0
                    if _d_pct >= 90:
                        st.warning(
                            f"🚨 **Dept budget at {_d_pct:.1f}% — approving this would reach {min(_after,100):.1f}%.** "
                            "Consult Finance before approving.",
                            icon="🚨",
                        )
                    elif _d_pct >= 70:
                        st.warning(
                            f"Dept budget is {_d_pct:.1f}% used. Approval would bring it to {_after:.1f}%.",
                            icon="⚠️",
                        )
                except Exception:
                    pass
                st.markdown('<div class="btn-approve">', unsafe_allow_html=True)
                if st.button(
                    "✅ Approve & Forward to Finance",
                    key=f"approve_{rid}",
                    use_container_width=True,
                ):
                    if not mgr_name_in.strip():
                        st.error("⚠️ Session error — please log in again.")
                    else:
                        try:
                            db.manager_approve(rid, mgr_name_in.strip())
                            st.success(
                                f"Report **#{rid}** approved by **{mgr_name_in}** "
                                "and forwarded to Finance.",
                                icon="✅",
                            )
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Reject tab ─────────────────────────────────────────────────────
            with tab_reject:
                st.markdown("""
                <div style="background:#7c2d1222; border:1px solid #dc2626; border-radius:10px;
                            padding:14px 18px; font-size:0.83rem; color:#fca5a5;
                            margin-bottom:12px; line-height:1.65;">
                    Write a clear rejection reason so the employee can resolve and resubmit.
                </div>
                """, unsafe_allow_html=True)
                rejection_reason = st.text_area(
                    "Rejection Reason * (required)",
                    placeholder=(
                        "e.g. Receipt is illegible. Please resubmit with a clear, "
                        "itemised receipt or file an Affidavit of Loss with HR."
                    ),
                    height=100,
                    key=f"reject_reason_{rid}",
                )
                st.markdown('<div class="btn-reject">', unsafe_allow_html=True)
                if st.button("❌ Reject This Report", key=f"reject_{rid}", use_container_width=True):
                    if not mgr_name_in.strip():
                        st.error("⚠️ Session error — please log in again.")
                    elif not rejection_reason.strip():
                        st.error("⚠️ Rejection reason is required.")
                    else:
                        try:
                            db.manager_reject(rid, mgr_name_in.strip(), rejection_reason.strip())
                            st.warning(
                                f"Report **#{rid}** rejected by **{mgr_name_in}**.",
                                icon="⚠️",
                            )
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Edit Details tab ───────────────────────────────────────────────
            with tab_edit:
                st.markdown("""
                <div style="background:#3b1f0044; border:1px solid #7c3aed; border-radius:10px;
                            padding:12px 16px; font-size:0.82rem; color:#c4b5fd;
                            margin-bottom:14px; line-height:1.55;">
                    ✏️ Edit report details below. Changing <b>amount, category, department,
                    purchase date, or receipt status</b> will automatically re-run the AI
                    risk model and update the score.
                </div>
                """, unsafe_allow_html=True)

                # Parse current purchase date
                raw_pd = row.get("Purchase_Date")
                if raw_pd is not None and hasattr(raw_pd, "year"):
                    cur_purchase_date = raw_pd
                else:
                    try:
                        cur_purchase_date = datetime.strptime(
                            str(raw_pd)[:10], "%Y-%m-%d"
                        ).date()
                    except Exception:
                        cur_purchase_date = date.today() - timedelta(days=1)

                with st.form(f"edit_form_{rid}"):
                    e_title = st.text_input(
                        "Report Title *",
                        value=str(row.get("Report_Title", "")),
                        max_chars=200,
                    )
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        e_cat = st.selectbox(
                            "Expense Category",
                            options=list(CAT_LABELS.keys()),
                            format_func=lambda k: CAT_LABELS[k],
                            index=list(CAT_LABELS.keys()).index(int(row.get("ExpCat_ID", 1)))
                            if int(row.get("ExpCat_ID", 1)) in CAT_LABELS else 0,
                        )
                    with ec2:
                        e_dept = st.selectbox(
                            "Department",
                            options=list(DEPT_LABELS.keys()),
                            format_func=lambda k: DEPT_LABELS[k],
                            index=list(DEPT_LABELS.keys()).index(int(row.get("Dept_ID", 101)))
                            if int(row.get("Dept_ID", 101)) in DEPT_LABELS else 0,
                        )
                    ea1, ea2 = st.columns(2)
                    with ea1:
                        e_amount = st.number_input(
                            "Total Amount (₱)",
                            min_value=1.0,
                            max_value=10_000_000.0,
                            value=float(row.get("Exp_TotalAmount", 0)),
                            step=50.0,
                            format="%.2f",
                        )
                    with ea2:
                        e_purchase_date = st.date_input(
                            "Purchase Date",
                            value=cur_purchase_date,
                        )
                    es1, es2 = st.columns(2)
                    with es1:
                        cur_pm = str(row.get("Payment_Method") or PAYMENT_METHODS[0])
                        pm_idx = PAYMENT_METHODS.index(cur_pm) if cur_pm in PAYMENT_METHODS else 0
                        e_payment = st.selectbox("Payment Method", PAYMENT_METHODS, index=pm_idx)
                    with es2:
                        e_seller = st.text_input(
                            "Seller / Merchant",
                            value=str(row.get("Seller") or ""),
                            max_chars=200,
                        )
                    e_desc = st.text_area(
                        "Description / Notes",
                        value=str(row.get("Description") or ""),
                        height=80,
                        max_chars=2000,
                    )
                    e_receipt = st.toggle(
                        "Receipt Attached",
                        value=not bool(row.get("IsAffidavit", 0)),
                    )
                    save_btn = st.form_submit_button(
                        "💾 Save Changes & Re-compute AI",
                        use_container_width=True,
                    )

                if save_btn:
                    if not e_title.strip():
                        st.error("❌ Report Title is required.")
                    else:
                        try:
                            today_d          = date.today()
                            sub_gap          = (today_d - e_purchase_date).days
                            is_weekend_sub   = today_d.weekday() >= 5
                            pur_is_weekend   = e_purchase_date.weekday() >= 5
                            is_round         = (e_amount % 100 == 0)
                            is_affidavit     = int(not e_receipt)

                            ai_row = {
                                "Exp_TotalAmount":     e_amount,
                                "Dept_ID":             e_dept,
                                "ExpCat_ID":           e_cat,
                                "IsAffidavit":         is_affidavit,
                                "SubmissionGap":       sub_gap,
                                "Is_Weekend":          int(is_weekend_sub),
                                "Purchase_Is_Weekend": int(pur_is_weekend),
                                "Is_Round_Number":     int(is_round),
                            }
                            _, new_score, _ = predict_risk(ai_row)
                            new_triage, _   = get_triage(new_score)

                            db.update_report(rid, {
                                "Report_Title":   e_title.strip(),
                                "Dept_ID":        e_dept,
                                "ExpCat_ID":      e_cat,
                                "Exp_TotalAmount": e_amount,
                                "Purchase_Date":  e_purchase_date,
                                "Submission_Gap": sub_gap,
                                "IsAffidavit":    is_affidavit,
                                "HasReceipt":     int(e_receipt),
                                "Is_Weekend":     int(is_weekend_sub),
                                "Is_Round_Number": int(is_round),
                                "Seller":         e_seller.strip() or None,
                                "Payment_Method": e_payment,
                                "Description":    e_desc.strip() or None,
                                "AI_Risk_Score":  round(new_score, 2),
                                "Triage_Level":   new_triage,
                            })
                            st.success(
                                f"Report **#{rid}** updated. "
                                f"New AI Risk Score: **{new_score:.1f}%** ({new_triage})",
                                icon="✅",
                            )
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Save failed: {exc}")

        elif exp_status in ("Approved by Manager", "Reimbursed"):
            st.success(
                f"**{s_lbl}** — Decision by: **{mgr_name or '—'}** on {mgr_date or '—'}",
                icon=s_emoji,
            )
        elif is_rejected:
            st.error(
                f"Rejected by **{mgr_name or '—'}** on {mgr_date or '—'}. "
                "Use the Edit tab to correct data if needed.",
                icon="❌",
            )

    # ── FINANCE OFFICER ────────────────────────────────────────────────────────
    elif role == "Finance Officer":
        if exp_status == "Approved by Manager":
            st.markdown("#### 🏦 Finance Officer — Reimbursement")
            fo_r1, fo_r2 = st.columns(2)
            fo_r1.metric("Amount to Reimburse", f"₱{float(row['Exp_TotalAmount']):,.2f}")
            fo_r2.metric("Approved by", mgr_name or "—")

            st.markdown("""
            <div style="background:#1e3a5f44; border:1px solid #2563eb; border-radius:10px;
                        padding:14px 18px; font-size:0.83rem; color:#93c5fd;
                        margin:12px 0; line-height:1.65;">
                <b>Before processing:</b><br>
                ✔ Confirm employee bank / GCash disbursement method<br>
                ✔ Issue payment for the full requested amount<br>
                ✔ Record the transaction reference
            </div>
            """, unsafe_allow_html=True)

            fo_name_in  = st.session_state.get("full_name", "")
            payment_ref = st.text_input(
                "Payment Reference / Transaction ID (optional)",
                placeholder="e.g. TXN-20260227-0042",
                key=f"fo_ref_{rid}",
            )
            st.info(f"Acting as: **{fo_name_in}**", icon="👤")

            st.markdown('<div class="btn-pay">', unsafe_allow_html=True)
            if st.button(
                f"💸  Process Full Reimbursement  ·  ₱{float(row['Exp_TotalAmount']):,.2f}",
                key=f"reimburse_{rid}",
                use_container_width=True,
            ):
                if not fo_name_in.strip():
                    st.error("⚠️ Session error — please log in again.")
                else:
                    try:
                        db.reimburse_report(rid, fo_name_in.strip())
                        st.success(
                            f"Reimbursed ₱{float(row['Exp_TotalAmount']):,.2f} "
                            f"by **{fo_name_in.strip()}**.",
                            icon="💸",
                        )
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)

        elif exp_status == "Reimbursed":
            st.success(
                f"**Fully Reimbursed** — by **{reimbursed_by or '—'}** on {reimbursed_date or '—'}",
                icon="💸",
            )
        elif exp_status == "Pending Manager Review":
            st.info("Awaiting manager decision. Report will appear in your queue once approved.", icon="⏳")
        elif is_rejected:
            st.warning("Rejected by Manager. No reimbursement required.", icon="⚠️")

    # ── EMPLOYEE ───────────────────────────────────────────────────────────────
    else:
        STATUS_MSG = {
            "Pending Manager Review": (
                "info",
                "⏳ **Awaiting Manager Review** — Your report is in the queue.",
            ),
            "Approved by Manager": (
                "success",
                f"✅ **Approved by {mgr_name or 'Manager'}** on {mgr_date or '—'} — "
                "Forwarded to Finance for reimbursement.",
            ),
            "Rejected by Manager": (
                "error",
                f"❌ **Rejected by {mgr_name or 'Manager'}** on {mgr_date or '—'} — "
                "See rejection reason above.",
            ),
            "Reimbursed": (
                "success",
                f"💸 **Reimbursed** — ₱{float(row['Exp_TotalAmount']):,.2f} "
                f"processed on {reimbursed_date or '—'}.",
            ),
        }
        msg_type, msg_text = STATUS_MSG.get(exp_status, ("info", f"Status: {exp_status}"))
        getattr(st, msg_type)(msg_text)
        # ── Resubmit button for rejected reports ──────────────────────────────
        if is_rejected:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="btn-edit">', unsafe_allow_html=True)
            if st.button("🔄 Resubmit This Report", key=f"resubmit_{rid}", use_container_width=True):
                try:
                    db.resubmit_report(rid)
                    st.success(
                        f"Report **#{rid}** resubmitted — now **Pending Manager Review** again.",
                        icon="✅",
                    )
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Resubmit failed: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)
        st.caption("All decisions are made by authorised Managers and Finance Officers.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABLE — clickable rows + multi-select delete for Manager
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("#### 📋 Expense Reports")

if view.empty:
    st.info("No reports match the current filters. Adjust the sidebar filters.")
else:
    # Build display DataFrame
    disp = view[["Report_ID", "Report_Title", "Dept_Name",
                 "Cat_Name", "Exp_TotalAmount", "AI_Risk_Score",
                 "Triage_Level", "Exp_Status", "Submission_Date"]].copy()
    disp["Amount (₱)"] = disp["Exp_TotalAmount"].apply(lambda x: f"₱{x:,.0f}")
    disp["Risk %"]     = disp["AI_Risk_Score"].apply(lambda x: f"{x:.0f}%")
    disp["Submitted"]  = pd.to_datetime(disp["Submission_Date"]).dt.strftime("%Y-%m-%d")

    table = (
        disp.rename(columns={
            "Report_ID":   "ID",
            "Report_Title":"Title",
            "Dept_Name":   "Dept",
            "Cat_Name":    "Category",
            "Triage_Level":"Triage",
            "Exp_Status":  "Status",
        })[["ID", "Title", "Dept", "Category", "Amount (₱)", "Risk %", "Triage", "Status", "Submitted"]]
        .set_index("ID")
    )

    if role == "Manager":
        st.markdown(
            '<p class="click-hint">'
            '👆 Click <b>1 row</b> to open detail (Approve / Reject / Edit) &nbsp;·&nbsp; '
            'Select <b>multiple rows</b> to enable batch delete.'
            '</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="click-hint">👆 Click any row to open full report detail.</p>',
            unsafe_allow_html=True,
        )

    sel_mode = "multi-row" if role == "Manager" else "single-row"

    event = st.dataframe(
        table,
        use_container_width=True,
        height=420,
        on_select="rerun",
        selection_mode=sel_mode,
        key="reports_table",
    )

    selected_rows = event.selection.rows

    # ── Manager: multi-delete when ≥2 rows OR explicit delete of 1 row ────────
    if role == "Manager" and len(selected_rows) >= 2:
        selected_ids = [int(view.iloc[i]["Report_ID"]) for i in selected_rows]
        st.markdown("---")
        st.markdown(f"""
        <div style="background:#7c2d1222; border:1px solid #dc2626; border-radius:10px;
                    padding:12px 16px; margin-bottom:10px;">
            <span style="color:#fca5a5; font-weight:700; font-size:0.9rem;">
                🗑️ {len(selected_ids)} reports selected for deletion:
            </span>
            <span style="color:#94a3b8; font-size:0.8rem; margin-left:8px;">
                IDs {", ".join(f"#{i}" for i in selected_ids)}
            </span>
            <br><span style="color:#ef4444; font-size:0.75rem;">
                ⚠️ This action is permanent and cannot be undone.
            </span>
        </div>
        """, unsafe_allow_html=True)

        col_del, col_cancel = st.columns([1, 3])
        with col_del:
            st.markdown('<div class="btn-delete">', unsafe_allow_html=True)
            if st.button(
                f"🗑️ Delete {len(selected_ids)} Reports",
                key="batch_delete_btn",
                use_container_width=True,
            ):
                try:
                    deleted = db.delete_reports(selected_ids)
                    st.success(
                        f"**{deleted} report(s) permanently deleted** "
                        f"(IDs: {', '.join(f'#{i}' for i in selected_ids)}).",
                        icon="🗑️",
                    )
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)
        with col_cancel:
            st.info("Deselect rows or click elsewhere to cancel.", icon="ℹ️")

    # ── Single-row → open detail modal ────────────────────────────────────────
    elif len(selected_rows) == 1:
        selected_index = selected_rows[0]
        rid = int(view.iloc[selected_index]["Report_ID"])

        # Manager: quick-delete shortcut above modal
        if role == "Manager":
            del_col, _ = st.columns([1, 5])
            with del_col:
                st.markdown('<div class="btn-delete">', unsafe_allow_html=True)
                if st.button(f"🗑️ Delete Report #{rid}", key=f"quick_del_{rid}"):
                    try:
                        db.delete_reports([rid])
                        st.success(f"Report **#{rid}** permanently deleted.", icon="🗑️")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")
                st.markdown("</div>", unsafe_allow_html=True)

        show_report_modal(rid)
