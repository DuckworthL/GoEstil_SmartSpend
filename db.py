"""
db.py — SmartSpend AI database layer
Connects to SQLite: smartspend.db
"""

import sqlite3
import pandas as pd
import hashlib
import os
from datetime import date
from init_sqlite import init_db

DB_PATH = 'smartspend.db'

# Auto-initialize SQLite DB if it doesn't exist
if not os.path.exists(DB_PATH):
    init_db()

def get_connection():
    """Return an open sqlite3 connection. check_same_thread=False allows Streamlit multi-threading."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def is_db_available() -> bool:
    """Quick health-check."""
    try:
        conn = get_connection()
        conn.close()
        return True
    except Exception:
        return False


# ── Write operations ──────────────────────────────────────────────────────────

def save_expense(
    report_title: str,
    dept_id: int,
    expcat_id: int,
    total_amount: float,
    purchase_date: date,
    submission_date: date,
    submission_gap: int,
    is_affidavit: bool,
    has_receipt: bool,
    is_weekend: bool,
    is_round_number: bool,
    seller: str,
    payment_method: str,
    description: str,
    ai_risk_score: float,
    triage_level: str,
    submitted_by_id: int | None = None,
) -> int:
    """
    Insert a new expense report.
    Returns the new Report_ID.
    """
    sql = """
        INSERT INTO ExpenseReports (
            Report_Title, Dept_ID, ExpCat_ID, Exp_TotalAmount,
            Purchase_Date, Submission_Date, Submission_Gap,
            IsAffidavit, HasReceipt, Is_Weekend, Is_Round_Number,
            Seller, Payment_Method, Description,
            AI_Risk_Score, Triage_Level, Exp_Status, Submitted_By
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Pending Manager Review', ?)
    """
    params = (
        report_title, dept_id, expcat_id, total_amount,
        purchase_date, submission_date, submission_gap,
        int(is_affidavit), int(has_receipt), int(is_weekend), int(is_round_number),
        seller, payment_method, description,
        round(ai_risk_score, 2), triage_level, submitted_by_id,
    )
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        report_id = cursor.lastrowid
        conn.commit()
    return report_id if report_id else -1


def update_report_status(report_id: int, status: str, reviewed_by: str) -> None:
    """Legacy generic status update (kept for backwards compat)."""
    sql = """
        UPDATE ExpenseReports
        SET Exp_Status  = ?,
            Reviewed_By = ?,
            Review_Date = date('now')
        WHERE Report_ID = ?
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (status, reviewed_by, report_id))
        conn.commit()


def manager_approve(report_id: int, manager_name: str) -> None:
    """Manager approves — moves report to Finance Officer reimbursement queue."""
    sql = """
        UPDATE ExpenseReports
        SET Exp_Status           = 'Approved by Manager',
            Manager_Name         = ?,
            Manager_Decision_Date = date('now')
        WHERE Report_ID = ?
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (manager_name, report_id))
        conn.commit()


def manager_reject(report_id: int, manager_name: str, rejection_reason: str) -> None:
    """Manager rejects — records reason, closes the report."""
    sql = """
        UPDATE ExpenseReports
        SET Exp_Status            = 'Rejected by Manager',
            Manager_Name          = ?,
            Manager_Decision_Date = date('now'),
            Rejection_Reason      = ?
        WHERE Report_ID = ?
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (manager_name, rejection_reason, report_id))
        conn.commit()


def reimburse_report(report_id: int, fo_name: str) -> None:
    """Finance Officer marks report as fully reimbursed."""
    sql = """
        UPDATE ExpenseReports
        SET Exp_Status     = 'Reimbursed',
            Reimbursed_By  = ?,
            Reimbursed_Date = date('now')
        WHERE Report_ID = ?
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (fo_name, report_id))
        conn.commit()


# ── Read operations ───────────────────────────────────────────────────────────

def get_all_reports(user_id: int | None = None) -> pd.DataFrame:
    """
    Return expense reports joined with Dept and Category names.
    If user_id is given, returns only reports submitted by that user.
    """
    if user_id is not None:
        sql = "SELECT * FROM vw_ExpenseReports WHERE Submitted_By = ? ORDER BY Created_At DESC"
        with get_connection() as conn:
            return pd.read_sql(sql, conn, params=[user_id])
    sql = "SELECT * FROM vw_ExpenseReports ORDER BY Created_At DESC"
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def get_report_by_id(report_id: int) -> pd.Series | None:
    """Return a single report row as a Series, or None if not found."""
    sql = "SELECT * FROM vw_ExpenseReports WHERE Report_ID = ?"
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=[report_id])
    return df.iloc[0] if not df.empty else None


def get_dept_spend(dept_id: int) -> float:
    """Return total approved + pending spend for a department."""
    sql = """
        SELECT IFNULL(SUM(Exp_TotalAmount), 0)
        FROM ExpenseReports
        WHERE Dept_ID = ?
          AND Exp_Status IN ('Pending Review', 'Pending Manager Review', 'Approved by Manager', 'Approved')
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (dept_id,))
        row = cursor.fetchone()
    return float(row[0]) if row else 0.0


def get_dept_budget(dept_id: int) -> float:
    """Return the budget for a department."""
    sql = "SELECT Budget FROM Departments WHERE Dept_ID = ?"
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (dept_id,))
        row = cursor.fetchone()
    return float(row[0]) if row else 0.0


def get_manager_queue() -> pd.DataFrame:
    """Reports awaiting manager decision, sorted by risk (highest first)."""
    sql = """
        SELECT * FROM vw_ExpenseReports
        WHERE Exp_Status = 'Pending Manager Review'
        ORDER BY AI_Risk_Score DESC
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def get_fo_queue() -> pd.DataFrame:
    """Reports approved by manager — ready for Finance Officer reimbursement."""
    sql = """
        SELECT * FROM vw_ExpenseReports
        WHERE Exp_Status = 'Approved by Manager'
        ORDER BY Manager_Decision_Date ASC
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def get_pending_reports() -> pd.DataFrame:
    """Backwards-compat alias for manager queue."""
    return get_manager_queue()


def resubmit_report(report_id: int) -> None:
    """
    Reset a rejected report back to 'Pending Manager Review'.
    """
    sql = """
        UPDATE ExpenseReports
        SET Exp_Status            = 'Pending Manager Review',
            Manager_Name          = NULL,
            Manager_Decision_Date = NULL,
            Rejection_Reason      = NULL
        WHERE Report_ID = ?
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (report_id,))
        conn.commit()


def get_submission_trend() -> pd.DataFrame:
    """
    Return weekly submission counts and average AI risk score
    for up to the past 90 days — used for trend charts.
    """
    sql = """
        SELECT
            date(Submission_Date, 'weekday 1', '-7 days') AS Week_Start,
            COUNT(*)               AS Submissions,
            AVG(IFNULL(AI_Risk_Score, 0)) AS Avg_Risk,
            SUM(Exp_TotalAmount)   AS Total_Amount,
            SUM(CASE WHEN Triage_Level = 'Audit Flag' OR Triage_Level = 'AUDIT FLAG' THEN 1 ELSE 0 END) AS Audit_Flags,
            SUM(CASE WHEN Triage_Level = 'Standard Review' OR Triage_Level = 'STANDARD REVIEW' THEN 1 ELSE 0 END) AS Standard
        FROM ExpenseReports
        WHERE Submission_Date >= date('now', '-90 days')
        GROUP BY date(Submission_Date, 'weekday 1', '-7 days')
        ORDER BY Week_Start
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def get_dept_spend_summary() -> pd.DataFrame:
    """
    Return all departments with budget, current spend, and usage percentage.
    """
    sql = """
        SELECT
            d.Dept_ID,
            d.Dept_Name,
            d.Budget,
            IFNULL(SUM(e.Exp_TotalAmount), 0)  AS Current_Spend,
            IFNULL(COUNT(e.Report_ID), 0)      AS Report_Count,
            IFNULL(AVG(e.AI_Risk_Score), 0)    AS Avg_Risk
        FROM Departments d
        LEFT JOIN ExpenseReports e
            ON d.Dept_ID = e.Dept_ID
           AND e.Exp_Status NOT IN ('Rejected by Manager')
        GROUP BY d.Dept_ID, d.Dept_Name, d.Budget
        ORDER BY d.Dept_ID
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def get_status_breakdown() -> pd.DataFrame:
    """Return count of reports per status."""
    sql = """
        SELECT Exp_Status, COUNT(*) AS Count
        FROM ExpenseReports
        GROUP BY Exp_Status
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


# ── User authentication ────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256."""
    salt = os.urandom(16)
    key  = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 200_000)
    return salt.hex() + ':' + key.hex()


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password."""
    try:
        salt_hex, key_hex = stored_hash.split(':', 1)
        salt = bytes.fromhex(salt_hex)
        key  = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 200_000)
        return key.hex() == key_hex
    except Exception:
        return False


def get_user(username: str) -> dict | None:
    """Return user record dict."""
    sql = """
        SELECT User_ID, Username, Password_Hash, Full_Name, Role, Is_Active
        FROM Users
        WHERE Username = ?
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (username,))
            row = cursor.fetchone()
            if row:
                return {
                    "User_ID":       row[0],
                    "Username":      row[1],
                    "Password_Hash": row[2],
                    "Full_Name":     row[3],
                    "Role":          row[4],
                    "Is_Active":     bool(row[5]),
                }
    except Exception:
        pass
    return None


def update_report(report_id: int, updates: dict) -> None:
    """Update editable fields of an expense report."""
    ALLOWED = {
        "Report_Title", "Seller", "Payment_Method", "Description",
        "Dept_ID", "ExpCat_ID", "Exp_TotalAmount", "Purchase_Date",
        "Submission_Gap", "IsAffidavit", "HasReceipt", "Is_Weekend",
        "Is_Round_Number", "AI_Risk_Score", "Triage_Level",
    }
    cols = {k: v for k, v in updates.items() if k in ALLOWED}
    if not cols:
        return
    set_clause = ", ".join(f"{k} = ?" for k in cols)
    values = list(cols.values()) + [report_id]
    sql = f"UPDATE ExpenseReports SET {set_clause} WHERE Report_ID = ?"
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, values)
        conn.commit()


def delete_reports(report_ids: list) -> int:
    """Hard-delete expense reports."""
    if not report_ids:
        return 0
    placeholders = ", ".join("?" * len(report_ids))
    sql = f"DELETE FROM ExpenseReports WHERE Report_ID IN ({placeholders})"
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, list(report_ids))
        count = cursor.rowcount
        conn.commit()
    return count


def seed_demo_users() -> None:
    """Insert demo accounts if Users table is empty."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM Users")
            if cursor.fetchone()[0] > 0:
                return

            demo = [
                ("employee1", "Employee@123", "Juan dela Cruz",  "Employee"),
                ("manager1",  "Manager@123",  "Jose Reyes",       "Manager"),
                ("fo1",       "Finance@123",  "Anna Cruz",        "Finance Officer"),
            ]
            insert_sql = """
                INSERT INTO Users (Username, Password_Hash, Full_Name, Role)
                VALUES (?, ?, ?, ?)
            """
            for username, password, full_name, role in demo:
                cursor.execute(insert_sql, (username, hash_password(password), full_name, role))
            conn.commit()
    except Exception:
        pass
