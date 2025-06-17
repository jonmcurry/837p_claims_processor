import pandas as pd
from claims_ui.app.db_connector import get_sqlserver_engine

# --- Distinct values for filters ---
def get_distinct_orgs():
    """Fetches distinct organizations for filter dropdowns."""
    query = "SELECT org_id, org_name FROM dbo.facility_organization WHERE active = 1 ORDER BY org_name;"
    engine = get_sqlserver_engine()
    if not engine: return []
    try:
        df = pd.read_sql_query(query, engine)
        return [{'label': row['org_name'], 'value': row['org_id']} for _, row in df.iterrows()]
    except Exception as e:
        print(f"Error fetching distinct organizations: {e}")
        return []

def get_distinct_regions():
    """Fetches distinct regions for filter dropdowns."""
    query = "SELECT region_id, region_name FROM dbo.facility_region WHERE active = 1 ORDER BY region_name;"
    engine = get_sqlserver_engine()
    if not engine: return []
    try:
        df = pd.read_sql_query(query, engine)
        return [{'label': row['region_name'], 'value': row['region_id']} for _, row in df.iterrows()]
    except Exception as e:
        print(f"Error fetching distinct regions: {e}")
        return []

# --- Metrics Query Functions ---

def get_daily_processing_summary_metrics(facility_id=None, start_date=None, end_date=None):
    """Fetches daily processing summary metrics with filters."""
    engine = get_sqlserver_engine()
    if not engine: return pd.DataFrame()
    
    query = "SELECT * FROM dbo.daily_processing_summary WHERE 1=1"
    filters = []
    params = []

    if facility_id:
        if isinstance(facility_id, list):
            if facility_id:
                placeholders = ','.join(['?'] * len(facility_id))
                filters.append(f"facility_id IN ({placeholders})")
                params.extend(facility_id)
        else:
            filters.append("facility_id = ?")
            params.append(facility_id)
    if start_date:
        filters.append("summary_date >= ?")
        params.append(start_date)
    if end_date:
        filters.append("summary_date <= ?")
        params.append(end_date)

    if filters:
        query += " AND " + " AND ".join(filters)
    query += " ORDER BY summary_date, facility_id;"
    
    try:
        df = pd.read_sql_query(query, engine, params=params)
        return df
    except Exception as e:
        print(f"Error fetching daily processing summary: {e}")
        return pd.DataFrame()

def get_org_level_processing_metrics(org_id=None, region_id=None, start_date=None, end_date=None):
    """Fetches and aggregates processing metrics at org/region level."""
    engine = get_sqlserver_engine()
    if not engine: return pd.DataFrame()

    query = """
    SELECT 
        fo.org_id, fo.org_name,
        fr.region_id, fr.region_name,
        SUM(dps.total_claims_processed) as total_claims_processed,
        SUM(dps.total_claims_failed) as total_claims_failed,
        SUM(dps.total_charge_amount) as total_charge_amount,
        SUM(dps.total_reimbursement_amount) as total_reimbursement_amount
    FROM dbo.daily_processing_summary dps
    JOIN dbo.facilities f ON dps.facility_id = f.facility_id
    JOIN dbo.facility_organization fo ON f.org_id = fo.org_id
    LEFT JOIN dbo.facility_region fr ON f.region_id = fr.region_id
    WHERE 1=1
    """
    filters = []
    params = []

    if org_id:
        filters.append("fo.org_id = ?")
        params.append(org_id)
    if region_id:
        filters.append("fr.region_id = ?")
        params.append(region_id)
    if start_date:
        filters.append("dps.summary_date >= ?")
        params.append(start_date)
    if end_date:
        filters.append("dps.summary_date <= ?")
        params.append(end_date)

    if filters:
        query += " AND " + " AND ".join(filters)
    
    query += " GROUP BY fo.org_id, fo.org_name, fr.region_id, fr.region_name ORDER BY fo.org_name, fr.region_name;"
    
    try:
        df = pd.read_sql_query(query, engine, params=params)
        if not df.empty:
            # Calculate overall error rate for the group
            df['overall_error_rate'] = (df['total_claims_failed'] / df['total_claims_processed']).fillna(0) * 100
        return df
    except Exception as e:
        print(f"Error fetching org level processing metrics: {e}")
        return pd.DataFrame()

def get_cpt_code_analytics(facility_id=None, start_date=None, end_date=None, top_n=10):
    """Fetches CPT code analytics."""
    engine = get_sqlserver_engine()
    if not engine: return pd.DataFrame()
    
    effective_top_n = top_n if isinstance(top_n, int) and top_n > 0 else 10

    query = f"""
    SELECT TOP ({effective_top_n})
        cli.procedure_code,
        COUNT(*) as cpt_volume,
        SUM(cli.charge_amount) as total_charges,
        SUM(cli.units) as total_units,
        SUM(cli.rvu_value) as total_rvus
    FROM dbo.claims_line_items cli
    JOIN dbo.claims c ON cli.facility_id = c.facility_id AND cli.patient_account_number = c.patient_account_number
    WHERE 1=1
    """
    filters = []
    params = []

    if facility_id:
        if isinstance(facility_id, list):
            if facility_id:
                placeholders = ','.join(['?'] * len(facility_id))
                filters.append(f"cli.facility_id IN ({placeholders})")
                params.extend(facility_id)
        else:
            filters.append("cli.facility_id = ?")
            params.append(facility_id)
    if start_date:
        filters.append("cli.service_from_date >= ?") # Using service_from_date for line items
        params.append(start_date)
    if end_date:
        filters.append("cli.service_from_date <= ?")
        params.append(end_date)

    if filters:
        query += " AND " + " AND ".join(filters)
        
    query += " GROUP BY cli.procedure_code ORDER BY cpt_volume DESC, total_charges DESC;"
    
    try:
        df = pd.read_sql_query(query, engine, params=params)
        return df
    except Exception as e:
        print(f"Error fetching CPT code analytics: {e}")
        return pd.DataFrame()

def get_diagnosis_code_analytics(facility_id=None, start_date=None, end_date=None, top_n=10):
    """Fetches diagnosis code analytics."""
    engine = get_sqlserver_engine()
    if not engine: return pd.DataFrame()

    effective_top_n = top_n if isinstance(top_n, int) and top_n > 0 else 10

    query = f"""
    SELECT TOP ({effective_top_n})
        cd.diagnosis_code, 
        cd.diagnosis_description,
        COUNT(*) as diagnosis_volume
    FROM dbo.claims_diagnosis cd
    JOIN dbo.claims c ON cd.facility_id = c.facility_id AND cd.patient_account_number = c.patient_account_number
    WHERE 1=1
    """
    filters = []
    params = []

    if facility_id:
        if isinstance(facility_id, list):
            if facility_id:
                placeholders = ','.join(['?'] * len(facility_id))
                filters.append(f"cd.facility_id IN ({placeholders})")
                params.extend(facility_id)
        else:
            filters.append("cd.facility_id = ?")
            params.append(facility_id)
    if start_date:
        filters.append("c.created_at >= ?") # Using claim's created_at for diagnosis context
        params.append(start_date)
    if end_date:
        filters.append("c.created_at <= ?")
        params.append(end_date)

    if filters:
        query += " AND " + " AND ".join(filters)
        
    query += " GROUP BY cd.diagnosis_code, cd.diagnosis_description ORDER BY diagnosis_volume DESC;"

    try:
        df = pd.read_sql_query(query, engine, params=params)
        return df
    except Exception as e:
        print(f"Error fetching diagnosis code analytics: {e}")
        return pd.DataFrame()

def get_payer_analytics(facility_id=None, start_date=None, end_date=None):
    """Fetches payer analytics."""
    engine = get_sqlserver_engine()
    if not engine: return pd.DataFrame()

    query = """
    SELECT 
        csp.payer_id, csp.payer_name,
        COUNT(DISTINCT c.patient_account_number) as distinct_claims_for_payer,
        SUM(cli.charge_amount) as total_charges_for_payer 
    FROM dbo.claims c
    JOIN dbo.facility_financial_classes ffc ON c.facility_id = ffc.facility_id AND c.financial_class_id = ffc.financial_class_id
    JOIN dbo.core_standard_payers csp ON ffc.payer_id = csp.payer_id
    JOIN dbo.claims_line_items cli ON c.facility_id = cli.facility_id AND c.patient_account_number = cli.patient_account_number
    WHERE 1=1
    """
    filters = []
    params = []

    if facility_id:
        if isinstance(facility_id, list):
            if facility_id:
                placeholders = ','.join(['?'] * len(facility_id))
                filters.append(f"c.facility_id IN ({placeholders})")
                params.extend(facility_id)
        else:
            filters.append("c.facility_id = ?")
            params.append(facility_id)
    if start_date:
        filters.append("c.created_at >= ?")
        params.append(start_date)
    if end_date:
        filters.append("c.created_at <= ?")
        params.append(end_date)

    if filters:
        query += " AND " + " AND ".join(filters)
        
    query += " GROUP BY csp.payer_id, csp.payer_name ORDER BY distinct_claims_for_payer DESC;"
    
    try:
        df = pd.read_sql_query(query, engine, params=params)
        return df
    except Exception as e:
        print(f"Error fetching payer analytics: {e}")
        return pd.DataFrame()

def get_patient_demographics(facility_id=None, start_date=None, end_date=None):
    """Fetches patient demographics data (gender, age)."""
    engine = get_sqlserver_engine()
    if not engine: return pd.DataFrame()

    query = """
    SELECT 
        gender, 
        date_of_birth,
        DATEDIFF(year, date_of_birth, GETDATE()) AS age
    FROM dbo.claims c
    WHERE 1=1
    """
    filters = []
    params = []

    if facility_id:
        if isinstance(facility_id, list):
            if facility_id:
                placeholders = ','.join(['?'] * len(facility_id))
                filters.append(f"c.facility_id IN ({placeholders})")
                params.extend(facility_id)
        else:
            filters.append("c.facility_id = ?")
            params.append(facility_id)
    if start_date:
        filters.append("c.created_at >= ?")
        params.append(start_date)
    if end_date:
        filters.append("c.created_at <= ?")
        params.append(end_date)

    if filters:
        query += " AND " + " AND ".join(filters)
    query += ";"

    try:
        df = pd.read_sql_query(query, engine, params=params)
        return df
    except Exception as e:
        print(f"Error fetching patient demographics: {e}")
        return pd.DataFrame()

def get_provider_summary_analytics(facility_id=None, start_date=None, end_date=None, top_n=10):
    """Fetches provider summary analytics."""
    engine = get_sqlserver_engine()
    if not engine: return pd.DataFrame()

    effective_top_n = top_n if isinstance(top_n, int) and top_n > 0 else 10

    query = f"""
    SELECT TOP ({effective_top_n})
        p.rendering_provider_id,
        p.first_name AS provider_first_name,
        p.last_name AS provider_last_name,
        COUNT(DISTINCT cli.patient_account_number) as unique_patients_count,
        SUM(cli.units) as total_units,
        SUM(cli.charge_amount) as total_provider_charges,
        SUM(cli.rvu_value) as total_provider_rvus
    FROM dbo.claims_line_items cli
    JOIN dbo.physicians p ON cli.rendering_provider_id = p.rendering_provider_id
    JOIN dbo.claims c ON cli.facility_id = c.facility_id AND cli.patient_account_number = c.patient_account_number
    WHERE cli.rendering_provider_id IS NOT NULL
    """
    filters = []
    params = []

    if facility_id:
        if isinstance(facility_id, list):
            if facility_id:
                placeholders = ','.join(['?'] * len(facility_id))
                filters.append(f"cli.facility_id IN ({placeholders})")
                params.extend(facility_id)
        else:
            filters.append("cli.facility_id = ?")
            params.append(facility_id)
    if start_date:
        filters.append("cli.service_from_date >= ?")
        params.append(start_date)
    if end_date:
        filters.append("cli.service_from_date <= ?")
        params.append(end_date)

    if filters:
        query += " AND " + " AND ".join(filters)

    query += " GROUP BY p.rendering_provider_id, p.first_name, p.last_name ORDER BY total_provider_charges DESC;"

    try:
        df = pd.read_sql_query(query, engine, params=params)
        return df
    except Exception as e:
        print(f"Error fetching provider summary analytics: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    print("Testing metrics_queries.py (ensure db_connector.py is configured and SQL Server is accessible).")
    # Example test calls:
    # orgs = get_distinct_orgs()
    # print(f"Found {len(orgs)} orgs. First: {orgs[:1]}")
    # daily_summary = get_daily_processing_summary_metrics(start_date='2024-01-01', end_date='2024-01-31')
    # print(f"Daily Summary:\n{daily_summary}")
    # cpt_analytics = get_cpt_code_analytics(top_n=3)
    # print(f"CPT Analytics (top 3):\n{cpt_analytics}")
    # provider_summary = get_provider_summary_analytics(top_n=3)
    # print(f"Provider Summary (top 3):\n{provider_summary}")