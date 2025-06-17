# Handles database connections
import sqlalchemy

# Configuration (Ideally, use environment variables or a config file for these)
DB_CONFIG = {
    "server_name": "localhost",  # Replace with your SQL Server instance name
    "database_name": "smart_pro_claims", # Replace with your database name
    "username": "sa",        # Replace with your SQL Server username
    "password": "ClearToFly1",        # Replace with your SQL Server password
    "driver": "ODBC Driver 17 for SQL Server" # Ensure this driver is installed
}

def get_sqlserver_engine():
    """
    Creates and returns a SQLAlchemy engine for SQL Server.

    Constructs the connection string using predefined configuration.
    Includes basic error handling for connection issues.
    """
    try:
        # Construct the connection string
        # For security, connection details (server, db, user, pass) should ideally be
        # stored in environment variables or a secure configuration management system,
        # not hardcoded in the script.
        conn_str = (
            f"mssql+pyodbc://{DB_CONFIG['username']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['server_name']}/{DB_CONFIG['database_name']}"
            f"?driver={DB_CONFIG['driver'].replace(' ', '+')}"
        )

        # Create the SQLAlchemy engine
        engine = sqlalchemy.create_engine(conn_str)

        print(f"SQLAlchemy engine created for {DB_CONFIG['database_name']} on {DB_CONFIG['server_name']}")
        return engine

    except Exception as e:
        print(f"Error creating SQL Server engine: {e}")
        # In a real application, you might want to log this error or raise it
        return None

if __name__ == '__main__':
    print("Attempting to create SQL Server engine (using placeholder credentials)...")
    engine = get_sqlserver_engine()

    if engine:
        print("Engine creation attempted.")
        try:
            # Try to establish a connection
            with engine.connect() as connection:
                print("Successfully connected to the database!")
                # You could run a simple query here if needed, e.g.:
                # result = connection.execute(sqlalchemy.text("SELECT 1"))
                # print(f"Test query result: {result.scalar_one()}")
        except sqlalchemy.exc.OperationalError as e:
            print(f"Connection failed (OperationalError): {e}")
            print("Please ensure:")
            print(f"1. SQL Server '{DB_CONFIG['server_name']}' is running and accessible.")
            print(f"2. Database '{DB_CONFIG['database_name']}' exists.")
            print(f"3. Credentials for user '{DB_CONFIG['username']}' are correct.")
            print(f"4. The ODBC driver '{DB_CONFIG['driver']}' is installed and configured correctly.")
            print("5. Network connectivity to the server is not blocked by a firewall.")
        except Exception as e:
            print(f"An unexpected error occurred during connection test: {e}")
    else:
        print("Engine creation failed. See error message above.")
        print("Please update the placeholder credentials in DB_CONFIG within this script.")
