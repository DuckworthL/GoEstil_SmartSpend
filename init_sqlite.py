import sqlite3
import os

DB_PATH = 'smartspend.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Departments
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Departments (
        Dept_ID   INTEGER PRIMARY KEY,
        Dept_Name TEXT NOT NULL,
        Budget    REAL NOT NULL DEFAULT 0.00
    )
    ''')

    # Insert default departments
    cursor.execute("SELECT COUNT(*) FROM Departments")
    if cursor.fetchone()[0] == 0:
        cursor.executemany('''
        INSERT INTO Departments (Dept_ID, Dept_Name, Budget) VALUES (?, ?, ?)
        ''', [
            (101, 'Sales', 500000.00),
            (102, 'Engineering', 750000.00),
            (103, 'Marketing', 400000.00),
            (104, 'Finance', 300000.00),
            (105, 'HR/Operations', 350000.00)
        ])

    # Expense Categories
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ExpenseCategories (
        Cat_ID   INTEGER PRIMARY KEY,
        Cat_Name TEXT NOT NULL
    )
    ''')
    cursor.execute("SELECT COUNT(*) FROM ExpenseCategories")
    if cursor.fetchone()[0] == 0:
        cursor.executemany('''
        INSERT INTO ExpenseCategories (Cat_ID, Cat_Name) VALUES (?, ?)
        ''', [
            (1, 'Meals & Entertainment'),
            (2, 'Transportation'),
            (3, 'Accommodation'),
            (4, 'Office Supplies'),
            (5, 'Software/Subscriptions'),
            (6, 'Training & Development'),
            (7, 'Medical/Wellness'),
            (8, 'Miscellaneous')
        ])

    # Users
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Users (
        User_ID       INTEGER PRIMARY KEY AUTOINCREMENT,
        Username      TEXT    NOT NULL UNIQUE,
        Password_Hash TEXT    NOT NULL,
        Full_Name     TEXT    NOT NULL,
        Role          TEXT    NOT NULL CHECK (Role IN ('Employee', 'Manager', 'Finance Officer')),
        Dept_ID       INTEGER REFERENCES Departments(Dept_ID),
        Is_Active     INTEGER NOT NULL DEFAULT 1,
        Created_At    DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Expense Reports
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ExpenseReports (
        Report_ID        INTEGER        PRIMARY KEY AUTOINCREMENT,
        Report_Title     TEXT           NOT NULL,
        Dept_ID          INTEGER        NOT NULL REFERENCES Departments(Dept_ID),
        ExpCat_ID        INTEGER        NOT NULL REFERENCES ExpenseCategories(Cat_ID),
        Exp_TotalAmount  REAL           NOT NULL,
        Purchase_Date    DATE           NOT NULL,
        Submission_Date  DATE           NOT NULL DEFAULT (DATE('now')),
        Submission_Gap   INTEGER        NOT NULL DEFAULT 0,
        IsAffidavit      INTEGER        NOT NULL DEFAULT 0,
        HasReceipt       INTEGER        NOT NULL DEFAULT 1,
        Is_Weekend       INTEGER        NOT NULL DEFAULT 0,
        Is_Round_Number  INTEGER        NOT NULL DEFAULT 0,
        Seller           TEXT           NULL,
        Payment_Method   TEXT           NULL,
        Description      TEXT           NULL,
        AI_Risk_Score    REAL           NULL,
        Triage_Level     TEXT           NULL,
        Exp_Status       TEXT           NOT NULL DEFAULT 'Pending Review',
        Reviewed_By      TEXT           NULL,
        Review_Date      DATE           NULL,
        Manager_Name          TEXT           NULL,
        Manager_Decision_Date DATE           NULL,
        Rejection_Reason      TEXT           NULL,
        Reimbursed_By         TEXT           NULL,
        Reimbursed_Date       DATE           NULL,
        Submitted_By          INTEGER        NULL REFERENCES Users(User_ID),
        Created_At       DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # View: vw_ExpenseReports
    cursor.execute('''
    DROP VIEW IF EXISTS vw_ExpenseReports
    ''')
    cursor.execute('''
    CREATE VIEW vw_ExpenseReports AS
    SELECT
        er.Report_ID,
        er.Report_Title,
        er.Dept_ID,
        d.Dept_Name,
        er.ExpCat_ID,
        ec.Cat_Name,
        er.Exp_TotalAmount,
        er.Purchase_Date,
        er.Submission_Date,
        er.Submission_Gap,
        er.IsAffidavit,
        er.HasReceipt,
        er.Is_Weekend,
        er.Is_Round_Number,
        er.Seller,
        er.Payment_Method,
        er.Description,
        er.AI_Risk_Score,
        er.Triage_Level,
        er.Exp_Status,
        er.Reviewed_By,
        er.Review_Date,
        er.Manager_Name,
        er.Manager_Decision_Date,
        er.Rejection_Reason,
        er.Reimbursed_By,
        er.Reimbursed_Date,
        er.Submitted_By,
        u.Full_Name   AS Submitted_By_Name,
        er.Created_At
    FROM ExpenseReports er
    JOIN Departments       d  ON er.Dept_ID  = d.Dept_ID
    JOIN ExpenseCategories ec ON er.ExpCat_ID = ec.Cat_ID
    LEFT JOIN Users        u  ON er.Submitted_By = u.User_ID
    ''')

    conn.commit()
    conn.close()
    print(f"✅ SQLite database initialized at {DB_PATH}")

if __name__ == "__main__":
    init_db()
