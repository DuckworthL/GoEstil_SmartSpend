-- ============================================================
-- SmartSpend AI — SQL Server LocalDB Schema Setup
-- Run once against (localdb)\MSSQLLocalDB
-- ============================================================

-- Create database
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'SmartSpendDB')
BEGIN
    CREATE DATABASE SmartSpendDB;
END
GO

USE SmartSpendDB;
GO

-- ── Departments ──────────────────────────────────────────────
IF OBJECT_ID('dbo.Departments', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.Departments (
        Dept_ID   INT           PRIMARY KEY,
        Dept_Name NVARCHAR(100) NOT NULL,
        Budget    DECIMAL(18,2) NOT NULL DEFAULT 0.00
    );

    INSERT INTO dbo.Departments (Dept_ID, Dept_Name, Budget) VALUES
        (101, 'Sales',           500000.00),
        (102, 'Engineering',     750000.00),
        (103, 'Marketing',       400000.00),
        (104, 'Finance',         300000.00),
        (105, 'HR/Operations',   350000.00);
END
GO

-- ── Expense Categories ───────────────────────────────────────
IF OBJECT_ID('dbo.ExpenseCategories', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ExpenseCategories (
        Cat_ID   INT           PRIMARY KEY,
        Cat_Name NVARCHAR(100) NOT NULL
    );

    INSERT INTO dbo.ExpenseCategories (Cat_ID, Cat_Name) VALUES
        (1, 'Meals & Entertainment'),
        (2, 'Transportation'),
        (3, 'Accommodation'),
        (4, 'Office Supplies'),
        (5, 'Software/Subscriptions'),
        (6, 'Training & Development'),
        (7, 'Medical/Wellness'),
        (8, 'Miscellaneous');
END
GO

-- ── Expense Reports ──────────────────────────────────────────
IF OBJECT_ID('dbo.ExpenseReports', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.ExpenseReports (
        Report_ID        INT            IDENTITY(1,1) PRIMARY KEY,
        Report_Title     NVARCHAR(200)  NOT NULL,
        Dept_ID          INT            NOT NULL REFERENCES dbo.Departments(Dept_ID),
        ExpCat_ID        INT            NOT NULL REFERENCES dbo.ExpenseCategories(Cat_ID),
        Exp_TotalAmount  DECIMAL(18,2)  NOT NULL,
        Purchase_Date    DATE           NOT NULL,
        Submission_Date  DATE           NOT NULL DEFAULT CAST(GETDATE() AS DATE),
        Submission_Gap   INT            NOT NULL DEFAULT 0,
        IsAffidavit      BIT            NOT NULL DEFAULT 0,
        HasReceipt       BIT            NOT NULL DEFAULT 1,
        Is_Weekend       BIT            NOT NULL DEFAULT 0,
        Is_Round_Number  BIT            NOT NULL DEFAULT 0,
        Seller           NVARCHAR(200)  NULL,
        Payment_Method   NVARCHAR(100)  NULL,
        Description      NVARCHAR(MAX)  NULL,
        AI_Risk_Score    DECIMAL(5,2)   NULL,
        Triage_Level     NVARCHAR(50)   NULL,
        Exp_Status       NVARCHAR(50)   NOT NULL DEFAULT 'Pending Review',
        Reviewed_By      NVARCHAR(100)  NULL,
        Review_Date      DATE           NULL,
        Created_At       DATETIME       NOT NULL DEFAULT GETDATE()
    );
END
GO

-- ── View: full report with lookup names ──────────────────────
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
    er.Reviewed_By,
    er.Review_Date,
    er.Created_At
FROM dbo.ExpenseReports er
JOIN dbo.Departments       d  ON er.Dept_ID  = d.Dept_ID
JOIN dbo.ExpenseCategories ec ON er.ExpCat_ID = ec.Cat_ID;
GO

PRINT 'SmartSpendDB setup complete.';
