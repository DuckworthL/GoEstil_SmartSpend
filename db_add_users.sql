-- ─────────────────────────────────────────────────────────────────────────────
-- db_add_users.sql — SmartSpend AI
-- Adds the Users table for role-based authentication.
-- Run once against (localdb)\MSSQLLocalDB → SmartSpendDB
-- ─────────────────────────────────────────────────────────────────────────────

USE SmartSpendDB;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.tables WHERE name = 'Users' AND schema_id = SCHEMA_ID('dbo')
)
BEGIN
    CREATE TABLE dbo.Users (
        User_ID       INT IDENTITY(1,1) PRIMARY KEY,
        Username      NVARCHAR(50)  NOT NULL UNIQUE,
        Password_Hash NVARCHAR(255) NOT NULL,          -- pbkdf2 + salt hex string
        Full_Name     NVARCHAR(100) NOT NULL,
        Role          NVARCHAR(30)  NOT NULL
                        CHECK (Role IN ('Employee', 'Manager', 'Finance Officer')),
        Dept_ID       INT NULL REFERENCES dbo.Departments(Dept_ID),
        Is_Active     BIT          NOT NULL DEFAULT 1,
        Created_At    DATETIME2        DEFAULT GETDATE()
    );
    PRINT 'Users table created.';
END
ELSE
    PRINT 'Users table already exists — skipped.';
GO

PRINT 'db_add_users.sql complete.';
GO
