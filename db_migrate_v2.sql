-- ============================================================
-- SmartSpend AI — DB Migration v2
-- Adds: Submitted_By FK, rebuilds view, adds trend query support
-- Safe to re-run (all changes are IF NOT EXISTS guarded)
-- ============================================================

USE SmartSpendDB;
GO

-- ── Add Submitted_By column (FK to dbo.Users) ────────────────────────────────
IF NOT EXISTS (SELECT 1 FROM sys.columns
               WHERE object_id = OBJECT_ID('dbo.ExpenseReports') AND name = 'Submitted_By')
    ALTER TABLE dbo.ExpenseReports ADD Submitted_By INT NULL;
GO

-- Add the FK constraint separately (safe to skip if already exists)
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_ExpenseReports_Submitted_By'
)
BEGIN
    -- Only add FK if Users table exists
    IF OBJECT_ID('dbo.Users', 'U') IS NOT NULL
        ALTER TABLE dbo.ExpenseReports
            ADD CONSTRAINT FK_ExpenseReports_Submitted_By
            FOREIGN KEY (Submitted_By) REFERENCES dbo.Users(User_ID);
END
GO

-- ── Rebuild view to include Submitted_By + submitter name ─────────────────────
IF OBJECT_ID('dbo.vw_ExpenseReports', 'V') IS NOT NULL
    DROP VIEW dbo.vw_ExpenseReports;
GO

CREATE VIEW dbo.vw_ExpenseReports AS
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
    -- Legacy columns
    er.Reviewed_By,
    er.Review_Date,
    -- Manager lifecycle
    er.Manager_Name,
    er.Manager_Decision_Date,
    er.Rejection_Reason,
    -- Finance Officer lifecycle
    er.Reimbursed_By,
    er.Reimbursed_Date,
    -- Submitter info
    er.Submitted_By,
    u.Full_Name   AS Submitted_By_Name,
    er.Created_At
FROM dbo.ExpenseReports er
JOIN dbo.Departments       d  ON er.Dept_ID  = d.Dept_ID
JOIN dbo.ExpenseCategories ec ON er.ExpCat_ID = ec.Cat_ID
LEFT JOIN dbo.Users         u  ON er.Submitted_By = u.User_ID;
GO

PRINT 'Migration v2 complete — Submitted_By column + updated view.';
