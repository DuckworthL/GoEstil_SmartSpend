import pyodbc
import os
import sys

# ── Connection string ─────────────────────────────────────────────────────────
# Connecting to master first to create the database if it doesn't exist
_MASTER_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=(localdb)\\MSSQLLocalDB;"
    "DATABASE=master;"
    "Trusted_Connection=yes;"
)

_MASTER_CONN_STR_FALLBACK = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=(localdb)\\MSSQLLocalDB;"
    "DATABASE=master;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

def get_master_connection():
    """Return an open pyodbc connection to master with autocommit enabled."""
    try:
        return pyodbc.connect(_MASTER_CONN_STR, timeout=5, autocommit=True)
    except pyodbc.Error:
        return pyodbc.connect(_MASTER_CONN_STR_FALLBACK, timeout=5, autocommit=True)

def execute_sql_file(cursor, filename):
    print(f"Executing {filename} ...")
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return False
        
    with open(filename, 'r', encoding='utf-8') as f:
        sql_script = f.read()

    # Split the script by GO and execute each batch
    batches = sql_script.split('GO\n')
    for batch in batches:
        # Also split by GO at the end of file / lines
        sub_batches = batch.split('\nGO')
        for sub_batch in sub_batches:
            if sub_batch.strip():
                try:
                    cursor.execute(sub_batch)
                except pyodbc.ProgrammingError as e:
                    # Ignore harmless "already exists" errors if possible, 
                    # but our scripts already use IF NOT EXISTS guards.
                    print(f"  Warning executing batch in {filename}: {e}")
                except Exception as e:
                    print(f"  Error executing batch in {filename}: {e}")
                    print(f"  Failing batch:\n{sub_batch.strip()[:200]}...")
                    return False
    print(f"[SUCCESS] Successfully executed {filename}")
    return True

def main():
    print("Initiating SmartSpend AI Database Setup...")
    try:
        conn = get_master_connection()
    except Exception as e:
        print(f"Failed to connect to localdb: {e}")
        print("Please ensure SQL Server LocalDB is installed and running.")
        sys.exit(1)

    cursor = conn.cursor()

    scripts = [
        "db_setup.sql",
        "db_add_users.sql",
        "db_migrate.sql",
        "db_migrate_v2.sql"
    ]

    for script in scripts:
        success = execute_sql_file(cursor, script)
        if not success:
            print("Aborting database initialization due to errors.")
            conn.close()
            sys.exit(1)

    print("\n[SUCCESS] Database initialization complete! You can now run the app.")
    conn.close()

if __name__ == "__main__":
    main()
