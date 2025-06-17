import pandas as pd
from claims_ui.app.db_connector import get_sqlserver_engine

# --- Query Functions ---

def get_distinct_facilities():
    """Fetches distinct facilities for filter dropdowns."""
    query = "SELECT facility_id, facility_name FROM dbo.facilities WHERE active = 1 ORDER BY facility_name;"
    engine = get_sqlserver_engine()
    if not engine:
        return []
    try:
        df = pd.read_sql_query(query, engine)
        return [{'label': row['facility_name'], 'value': row['facility_id']} for index, row in df.iterrows()]
    except Exception as e:
        print(f"Error fetching distinct facilities: {e}")
        return []

def get_distinct_payers():
    """Fetches distinct payers for filter dropdowns."""
    query = "SELECT payer_id, payer_name FROM dbo.core_standard_payers WHERE active = 1 ORDER BY payer_name;"
    engine = get_sqlserver_engine()
    if not engine:
        return []
    try:
        df = pd.read_sql_query(query, engine)
        return [{'label': row['payer_name'], 'value': row['payer_id']} for index, row in df.iterrows()]
    except Exception as e:
        print(f"Error fetching distinct payers: {e}")
        return []

def get_distinct_failure_categories():
    """Fetches distinct failure categories for filter dropdowns."""
    query = "SELECT DISTINCT failure_category FROM dbo.failed_claims WHERE failure_category IS NOT NULL ORDER BY failure_category;"
    engine = get_sqlserver_engine()
    if not engine:
        return []
    try:
        df = pd.read_sql_query(query, engine)
        return [{'label': cat, 'value': cat} for cat in df['failure_category'].unique() if cat]
    except Exception as e:
        print(f"Error fetching distinct failure categories: {e}")
        return []

def get_processed_claims(facility_id=None, payer_id=None, start_date=None, end_date=None, limit=100):
    """Fetches processed claims data with filters."""
    engine = get_sqlserver_engine()
    if not engine:
        return pd.DataFrame()

    # Ensure limit is an integer and positive, default to a high number if not for TOP clause
    effective_limit = limit if isinstance(limit, int) and limit > 0 else 100000

    base_query = f"""
    SELECT TOP ({effective_limit})
        c.facility_id,
        c.patient_account_number, c.medical_record_number, c.patient_name, 
        c.date_of_birth, c.gender, c.created_at AS claim_date,
        cli.line_number, cli.procedure_code, cli.modifier1, cli.units, 
        cli.charge_amount, cli.service_from_date, cli.rvu_value, cli.reimbursement_amount,
        cd.diagnosis_code AS primary_diagnosis_code, 
        cd.diagnosis_description AS primary_diagnosis_description,
        p.first_name AS provider_first_name, p.last_name AS provider_last_name,
        csp.payer_name
    FROM dbo.claims c
    JOIN dbo.claims_line_items cli ON c.facility_id = cli.facility_id AND c.patient_account_number = cli.patient_account_number
    LEFT JOIN dbo.claims_diagnosis cd ON c.facility_id = cd.facility_id AND c.patient_account_number = cd.patient_account_number AND cd.diagnosis_sequence = 1
    LEFT JOIN dbo.physicians p ON cli.rendering_provider_id = p.rendering_provider_id
    JOIN dbo.facility_financial_classes ffc ON c.facility_id = ffc.facility_id AND c.financial_class_id = ffc.financial_class_id
    JOIN dbo.core_standard_payers csp ON ffc.payer_id = csp.payer_id
    WHERE 1=1
    """
    
    filters = []
    params = [] # Parameters for pyodbc

    if facility_id:
        if isinstance(facility_id, list):
            if facility_id: 
                placeholders = ','.join(['?'] * len(facility_id))
                filters.append(f"c.facility_id IN ({placeholders})")
                params.extend(facility_id)
        else: 
            filters.append("c.facility_id = ?")
            params.append(facility_id)
            
    if payer_id:
        if isinstance(payer_id, list):
            if payer_id:
                placeholders = ','.join(['?'] * len(payer_id))
                filters.append(f"ffc.payer_id IN ({placeholders})")
                params.extend(payer_id)
        else:
            filters.append("ffc.payer_id = ?")
            params.append(payer_id)

    if start_date:
        filters.append("c.created_at >= ?")
        params.append(start_date)
    if end_date:
        filters.append("c.created_at <= ?")
        params.append(end_date)

    if filters:
        base_query += " AND " + " AND ".join(filters)
    
    base_query += " ORDER BY c.created_at DESC, c.patient_account_number, cli.line_number;"

    try:
        df = pd.read_sql_query(base_query, engine, params=params)
        return df
    except Exception as e:
        print(f"Error fetching processed claims: {e}")
        return pd.DataFrame()

def get_failed_claims(facility_id=None, failure_category=None, start_date=None, end_date=None, limit=100):
    """Fetches failed claims data with filters."""
    engine = get_sqlserver_engine()
    if not engine:
        return pd.DataFrame()

    effective_limit = limit if isinstance(limit, int) and limit > 0 else 100000

    base_query = f"""
    SELECT TOP ({effective_limit})
        fc.claim_id, fc.facility_id, f.facility_name, fc.patient_account_number, 
        fc.failure_reason, fc.failure_category, fc.processing_stage, 
        fc.failed_at, fc.resolution_status, fc.potential_revenue_loss, fc.coder_id
    FROM dbo.failed_claims fc
    LEFT JOIN dbo.facilities f ON fc.facility_id = f.facility_id
    WHERE 1=1
    """
    filters = []
    params = []

    if facility_id:
        if isinstance(facility_id, list):
            if facility_id:
                placeholders = ','.join(['?'] * len(facility_id))
                filters.append(f"fc.facility_id IN ({placeholders})")
                params.extend(facility_id)
        else:
            filters.append("fc.facility_id = ?")
            params.append(facility_id)

    if failure_category:
        if isinstance(failure_category, list):
            if failure_category:
                placeholders = ','.join(['?'] * len(failure_category))
                filters.append(f"fc.failure_category IN ({placeholders})")
                params.extend(failure_category)
        else:
            filters.append("fc.failure_category = ?")
            params.append(failure_category)
            
    if start_date:
        filters.append("fc.failed_at >= ?")
        params.append(start_date)
    if end_date:
        filters.append("fc.failed_at <= ?")
        params.append(end_date)

    if filters:
        base_query += " AND " + " AND ".join(filters)
        
    base_query += " ORDER BY fc.failed_at DESC;"

    try:
        df = pd.read_sql_query(base_query, engine, params=params)
        return df
    except Exception as e:
        print(f"Error fetching failed claims: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    print("Testing claims_queries.py (ensure db_connector.py is configured and SQL Server is accessible).")
    # Example test calls (uncomment and run if your DB is configured):
    # facilities = get_distinct_facilities()
    # print(f"Found {len(facilities)} facilities. First few: {facilities[:3]}")
    # payers = get_distinct_payers()
    # print(f"Found {len(payers)} payers. First few: {payers[:3]}")
    # failure_cats = get_distinct_failure_categories()
    # print(f"Found {len(failure_cats)} failure categories. First few: {failure_cats[:3]}")
    # processed = get_processed_claims(limit=2)
    # print(f"Processed claims (limit 2):\n{processed}")
    # failed = get_failed_claims(limit=2)
    # print(f"Failed claims (limit 2):\n{failed}")