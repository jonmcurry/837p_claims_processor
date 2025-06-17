import sqlalchemy
import urllib

# --- Database Configuration ---
# IMPORTANT: Replace these placeholder values with your actual SQL Server details.
# It is highly recommended to use environment variables or a secure configuration manager
# for sensitive information like passwords, rather than hardcoding them.
DB_CONFIG = {
    'server_name': 'localhost',  # e.g., 'localhost\SQLEXPRESS' or your server's DNS name
    'database_name': 'smart_pro_claims', # As per your schema files
    'username': 'sa',          # Your SQL Server username
    'password': 'ClearToFly1',          # Your SQL Server password
    'driver': 'ODBC Driver 17 for SQL Server' # Ensure this driver is installed
}

def get_sqlserver_engine():
    """
    Creates and returns a SQLAlchemy engine for SQL Server.

    Uses pyodbc as the DBAPI.
    Connection parameters are taken from the DB_CONFIG dictionary.

    Returns:
        sqlalchemy.engine.Engine: The SQLAlchemy engine instance, or None if an error occurs.
    """
    try:
        # For Windows Authentication, you might use: integrated_security = 'SSPI'
        # For SQL Server Authentication, provide username and password.
        params = urllib.parse.quote_plus(
            f"DRIVER={{{DB_CONFIG['driver']}}};"
            f"SERVER={DB_CONFIG['server_name']};"
            f"DATABASE={DB_CONFIG['database_name']};"
            f"UID={DB_CONFIG['username']};"
            f"PWD={DB_CONFIG['password']};"
            f"TrustServerCertificate=yes;" # Add if using self-signed cert or for local dev
        )

        # Connection URL format for mssql+pyodbc
        # Docs: https://docs.sqlalchemy.org/en/14/dialects/mssql.html#module-sqlalchemy.dialects.mssql.pyodbc
        conn_str = f"mssql+pyodbc:///?odbc_connect={params}"

        engine = sqlalchemy.create_engine(conn_str)
        # print(f"Successfully created SQLAlchemy engine for SQL Server: {DB_CONFIG['server_name']}")
        return engine

    except Exception as e:
        print(f"Error creating SQLAlchemy engine for SQL Server: {e}")
        print("Please check:")
        print(f"1. SQL Server is running and accessible from: {DB_CONFIG['server_name']}")
        print(f"2. Database '{DB_CONFIG['database_name']}' exists.")
        print(f"3. Credentials (username/password) are correct.")
        print(f"4. ODBC Driver '{DB_CONFIG['driver']}' is installed and correctly named.")
        print(f"5. Firewall rules allow connection to the SQL Server port (default 1433).")
        return None

# --- Test Connection (optional, run directly to verify) ---
if __name__ == '__main__':
    print("Attempting to connect to SQL Server using configuration in db_connector.py...")
    engine = get_sqlserver_engine()

    if engine:
        try:
            with engine.connect() as connection:
                print("Successfully connected to the SQL Server database!")
                print("SQLAlchemy Engine Details:", engine)
                # You can run a simple query here for further testing if needed, e.g.:
                # result = connection.execute(sqlalchemy.text("SELECT @@version;"))
                # for row in result:
                #     print("SQL Server Version:", row[0])
            print("Connection test successful and connection closed.")
        except sqlalchemy.exc.OperationalError as op_err:
            print(f"OperationalError during connection test: {op_err}")
            print("This often means the server was found, but there was an issue with database name, login credentials, or permissions.")
        except Exception as e:
            print(f"An unexpected error occurred during the connection test: {e}")
    else:
        print("Failed to create the SQLAlchemy engine. Connection test aborted.")