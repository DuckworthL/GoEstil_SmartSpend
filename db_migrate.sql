-- ============================================================
-- SmartSpend AI — DB Migration: 3-role lifecycle columns
-- Run against (localdb)\MSSQLLocalDB → SmartSpendDB
-- Safe to re-run (all changes are IF NOT EXISTS guarded)
-- ============================================================

USE SmartSpendDB;
GO

-- ── Add new lifecycle columns ─────────────────────────────────────────────────
IF NOT EXISTS (SELECT 1 FROM sys.columns
               WHERE object_id = OBJECT_ID('dbo.ExpenseReports') AND name = 'Manager_Name')
    ALTER TABLE dbo.ExpenseReports ADD Manager_Name NVARCHAR(100) NULL;

IF NOT EXISTS (SELECT 1 FROM sys.columns
               WHERE object_id = OBJECT_ID('dbo.ExpenseReports') AND name = 'Manager_Decision_Date')
    ALTER TABLE dbo.ExpenseReports ADD Manager_Decision_Date DATE NULL;

IF NOT EXISTS (SELECT 1 FROM sys.columns
               WHERE object_id = OBJECT_ID('dbo.ExpenseReports') AND name = 'Rejection_Reason')
    ALTER TABLE dbo.ExpenseReports ADD Rejection_Reason NVARCHAR(MAX) NULL;

IF NOT EXISTS (SELECT 1 FROM sys.columns
               WHERE object_id = OBJECT_ID('dbo.ExpenseReports') AND name = 'Reimbursed_By')
    ALTER TABLE dbo.ExpenseReports ADD Reimbursed_By NVARCHAR(100) NULL;

IF NOT EXISTS (SELECT 1 FROM sys.columns
               WHERE object_id = OBJECT_ID('dbo.ExpenseReports') AND name = 'Reimbursed_Date')
    ALTER TABLE dbo.ExpenseReports ADD Reimbursed_Date DATE NULL;

GO

-- ── Migrate old status values to new naming ───────────────────────────────────
UPDATE dbo.ExpenseReports
SET Exp_Status = 'Pending Manager Review'
WHERE Exp_Status IN ('Pending Review', 'Pending');

UPDATE dbo.ExpenseReports
SET Exp_Status = 'Approved by Manager'
WHERE Exp_Status = 'Approved';

UPDATE dbo.ExpenseReports
SET Exp_Status = 'Rejected by Manager'
WHERE Exp_Status = 'Rejected';

GO

-- ── Rebuild view with new columns ─────────────────────────────────────────────
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
    -- Legacy columns (kept for backwards compat)
    er.Reviewed_By,
    er.Review_Date,
    -- Manager lifecycle
    er.Manager_Name,
    er.Manager_Decision_Date,
    er.Rejection_Reason,
    -- Finance Officer lifecycle
    er.Reimbursed_By,
    er.Reimbursed_Date,
    er.Created_At
FROM dbo.ExpenseReports er
JOIN dbo.Departments       d  ON er.Dept_ID  = d.Dept_ID
JOIN dbo.ExpenseCategories ec ON er.ExpCat_ID = ec.Cat_ID;
GO

PRINT 'Migration complete — 3-role lifecycle columns added.';
