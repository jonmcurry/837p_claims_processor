# Contains SQL queries related to metrics and healthcare analytics
import pandas as pd
from claims_ui.app.db_connector import get_sqlserver_engine

# --- Processing Metrics ---

def get_daily_processing_summary_metrics(facility_id=None, start_date=None, end_date=None):
    """
    Retrieves daily processing summary metrics.
    Filters by facility_id (optional) and summary_date range (optional).
    """
    engine = get_sqlserver_engine()
    if engine is None:
        print("Database engine not available. Cannot fetch daily processing summary.")
        return pd.DataFrame()

    query_parts = [
        "SELECT summary_date, facility_id, total_claims_processed, total_claims_failed,",
        "total_charge_amount, total_reimbursement_amount, average_processing_time_seconds, error_rate_percentage",
        "FROM dbo.daily_processing_summary",
        "WHERE 1=1"
    ]
    params = {}

    if facility_id: # Assumes single facility_id for this specific query
        query_parts.append("AND facility_id = %(facility_id)s")
        params['facility_id'] = facility_id
    if start_date:
        query_parts.append("AND summary_date >= %(start_date)s")
        params['start_date'] = start_date
    if end_date:
        query_parts.append("AND summary_date <= %(end_date)s")
        params['end_date'] = end_date

    query_parts.append("ORDER BY summary_date DESC")
    query = " ".join(query_parts)

    try:
        print(f"Executing get_daily_processing_summary_metrics query with params: {params}")
        df = pd.read_sql_query(query, engine, params=params)
        print(f"Successfully fetched {len(df)} daily processing summary records.")
        return df
    except Exception as e:
        print(f"Error fetching daily processing summary metrics: {e}")
        return pd.DataFrame()

def get_org_level_processing_metrics(org_id=None, region_id=None, start_date=None, end_date=None):
    """
    Retrieves aggregated processing metrics at the organization/region level.
    Filters by org_id, region_id, and summary_date range.
    """
    engine = get_sqlserver_engine()
    if engine is None:
        print("Database engine not available. Cannot fetch org level metrics.")
        return pd.DataFrame()

    select_columns = [
        "fo.org_id, fo.org_name",
        "SUM(dps.total_claims_processed) AS total_claims_processed",
        "SUM(dps.total_claims_failed) AS total_claims_failed",
        "SUM(dps.total_charge_amount) AS total_charge_amount",
        "SUM(dps.total_reimbursement_amount) AS total_reimbursement_amount",
        """CASE
             WHEN SUM(dps.total_claims_processed) = 0 THEN 0
             ELSE (SUM(CAST(dps.total_claims_failed AS FLOAT)) * 100.0 / SUM(dps.total_claims_processed))
           END AS overall_error_rate_percentage"""
    ]
    group_by_columns = ["fo.org_id, fo.org_name"]
    query_from_join = [
        "FROM dbo.daily_processing_summary dps",
        "JOIN dbo.facilities f ON dps.facility_id = f.facility_id",
        "JOIN dbo.facility_organization fo ON f.org_id = fo.org_id"
    ]
    params = {}
    where_clauses = ["WHERE 1=1"]

    if region_id:
        query_from_join.append("JOIN dbo.facility_region fr ON f.region_id = fr.region_id")
        select_columns.insert(1, "fr.region_id, fr.region_name")
        group_by_columns.insert(1, "fr.region_id, fr.region_name")
        where_clauses.append("AND fr.region_id = %(region_id)s")
        params['region_id'] = region_id
    if org_id:
        where_clauses.append("AND fo.org_id = %(org_id)s")
        params['org_id'] = org_id
    if start_date:
        where_clauses.append("AND dps.summary_date >= %(start_date)s")
        params['start_date'] = start_date
    if end_date:
        where_clauses.append("AND dps.summary_date <= %(end_date)s")
        params['end_date'] = end_date

    query = (
        "SELECT " + ", ".join(select_columns) + " " +
        " ".join(query_from_join) + " " +
        " ".join(where_clauses) + " " +
        "GROUP BY " + ", ".join(group_by_columns) + " " +
        "ORDER BY fo.org_name" + (", fr.region_name" if region_id else "")
    )
    try:
        print(f"Executing get_org_level_processing_metrics query with params: {params}")
        df = pd.read_sql_query(query, engine, params=params)
        print(f"Successfully fetched {len(df)} org level processing metric records.")
        return df
    except Exception as e:
        print(f"Error fetching org level processing metrics: {e}")
        return pd.DataFrame()

# --- Healthcare Analytics ---

def get_cpt_code_analytics(facility_id=None, start_date=None, end_date=None, top_n=10):
    engine = get_sqlserver_engine()
    if engine is None: return pd.DataFrame()
    query_parts = [
        f"SELECT TOP {int(top_n)}" if top_n and isinstance(top_n, int) and top_n > 0 else "SELECT",
        "cli.procedure_code, COUNT(*) AS cpt_volume, SUM(cli.charge_amount) AS total_charges, SUM(cli.units) AS total_units, SUM(cli.rvu_value) AS total_rvus",
        "FROM dbo.claims_line_items cli JOIN dbo.claims c ON cli.facility_id = c.facility_id AND cli.patient_account_number = c.patient_account_number",
        "WHERE 1=1"
    ]
    params = {}
    if facility_id: # Assumes single facility_id or list handled by direct formatting if used elsewhere
        if isinstance(facility_id, list) and len(facility_id) > 0:
            in_clause = ", ".join([f"'{f}'" for f in facility_id])
            query_parts.append(f"AND c.facility_id IN ({in_clause})")
        elif isinstance(facility_id, str) and facility_id:
            query_parts.append("AND c.facility_id = %(facility_id_val)s")
            params['facility_id_val'] = facility_id
    if start_date:
        query_parts.append("AND cli.service_from_date >= %(start_date)s"); params['start_date'] = start_date
    if end_date:
        query_parts.append("AND cli.service_from_date <= %(end_date)s"); params['end_date'] = end_date
    query_parts.append("GROUP BY cli.procedure_code ORDER BY cpt_volume DESC, total_charges DESC")
    query = " ".join(query_parts)
    try:
        df = pd.read_sql_query(query, engine, params=params if params else None); return df
    except Exception as e: print(f"Error: {e}"); return pd.DataFrame()

def get_diagnosis_code_analytics(facility_id=None, start_date=None, end_date=None, top_n=10):
    engine = get_sqlserver_engine()
    if engine is None: return pd.DataFrame()
    query_parts = [
        f"SELECT TOP {int(top_n)}" if top_n and isinstance(top_n, int) and top_n > 0 else "SELECT",
        "cd.diagnosis_code, cd.diagnosis_description, COUNT(*) AS diagnosis_volume",
        "FROM dbo.claims_diagnosis cd JOIN dbo.claims c ON cd.facility_id = c.facility_id AND cd.patient_account_number = c.patient_account_number",
        "WHERE 1=1"
    ]
    params = {}
    if facility_id: # Assumes single facility_id or list handled by direct formatting
        if isinstance(facility_id, list) and len(facility_id) > 0:
            in_clause = ", ".join([f"'{f}'" for f in facility_id])
            query_parts.append(f"AND c.facility_id IN ({in_clause})")
        elif isinstance(facility_id, str) and facility_id:
            query_parts.append("AND c.facility_id = %(facility_id_val)s")
            params['facility_id_val'] = facility_id
    if start_date:
        query_parts.append("AND c.created_at >= %(start_date)s"); params['start_date'] = start_date
    if end_date:
        query_parts.append("AND c.created_at <= %(end_date)s"); params['end_date'] = end_date
    query_parts.append("GROUP BY cd.diagnosis_code, cd.diagnosis_description ORDER BY diagnosis_volume DESC")
    query = " ".join(query_parts)
    try:
        df = pd.read_sql_query(query, engine, params=params if params else None); return df
    except Exception as e: print(f"Error: {e}"); return pd.DataFrame()

def get_payer_analytics(facility_id=None, start_date=None, end_date=None):
    engine = get_sqlserver_engine()
    if engine is None: return pd.DataFrame()
    query_parts = [
        "SELECT csp.payer_id, csp.payer_name,",
        "COUNT(DISTINCT c.patient_account_number) AS distinct_claims_for_payer,",
        "SUM(cli.charge_amount) AS total_charges_for_payer",
        "FROM dbo.claims c",
        "JOIN dbo.facility_financial_classes ffc ON c.facility_id = ffc.facility_id AND c.financial_class_id = ffc.financial_class_id",
        "JOIN dbo.core_standard_payers csp ON ffc.payer_id = csp.payer_id",
        "LEFT JOIN dbo.claims_line_items cli ON c.facility_id = cli.facility_id AND c.patient_account_number = cli.patient_account_number",
        "WHERE 1=1"
    ]
    params = {}
    if facility_id:
        if isinstance(facility_id, list) and len(facility_id) > 0:
            in_clause = ", ".join([f"'{f}'" for f in facility_id])
            query_parts.append(f"AND c.facility_id IN ({in_clause})")
        elif isinstance(facility_id, str) and facility_id:
            query_parts.append("AND c.facility_id = %(facility_id_val)s")
            params['facility_id_val'] = facility_id
    if start_date:
        query_parts.append("AND c.created_at >= %(start_date)s"); params['start_date'] = start_date
    if end_date:
        query_parts.append("AND c.created_at <= %(end_date)s"); params['end_date'] = end_date
    query_parts.append("GROUP BY csp.payer_id, csp.payer_name ORDER BY distinct_claims_for_payer DESC, total_charges_for_payer DESC")
    query = " ".join(query_parts)
    try:
        df = pd.read_sql_query(query, engine, params=params if params else None); return df
    except Exception as e: print(f"Error: {e}"); return pd.DataFrame()

def get_patient_demographics(facility_id=None, start_date=None, end_date=None):
    engine = get_sqlserver_engine()
    if engine is None: return pd.DataFrame()
    query_parts = [
        "SELECT c.gender, c.date_of_birth, DATEDIFF(year, c.date_of_birth, GETDATE()) AS age",
        "FROM dbo.claims c WHERE 1=1"
    ]
    params = {}
    if facility_id:
        if isinstance(facility_id, list) and len(facility_id) > 0:
            in_clause = ", ".join([f"'{f}'" for f in facility_id])
            query_parts.append(f"AND c.facility_id IN ({in_clause})")
        elif isinstance(facility_id, str) and facility_id:
            query_parts.append("AND c.facility_id = %(facility_id_val)s")
            params['facility_id_val'] = facility_id
    if start_date:
        query_parts.append("AND c.created_at >= %(start_date)s"); params['start_date'] = start_date
    if end_date:
        query_parts.append("AND c.created_at <= %(end_date)s"); params['end_date'] = end_date
    query = " ".join(query_parts)
    try:
        df = pd.read_sql_query(query, engine, params=params if params else None); return df
    except Exception as e: print(f"Error: {e}"); return pd.DataFrame()

def get_provider_summary_analytics(facility_id=None, start_date=None, end_date=None, top_n=10):
    engine = get_sqlserver_engine()
    if engine is None: return pd.DataFrame()

    top_clause = f"TOP {int(top_n)}" if top_n and isinstance(top_n, int) and top_n > 0 else ""

    query_parts = [
        f"SELECT {top_clause}",
        "p.rendering_provider_id, p.first_name AS provider_first_name, p.last_name AS provider_last_name,",
        "COUNT(DISTINCT cli.patient_account_number) AS unique_patients_count,",
        "SUM(cli.units) AS total_units,",
        "SUM(cli.charge_amount) AS total_provider_charges,",
        "SUM(cli.rvu_value) AS total_provider_rvus",
        "FROM dbo.claims_line_items cli",
        "JOIN dbo.physicians p ON cli.rendering_provider_id = p.rendering_provider_id",
        "JOIN dbo.claims c ON cli.facility_id = c.facility_id AND cli.patient_account_number = c.patient_account_number",
        "WHERE 1=1"
    ]
    # Clean up query_parts in case top_clause is empty
    query_parts[0] = query_parts[0].replace("SELECT ", "SELECT").strip()
    if query_parts[0] == "SELECT": query_parts[0] = "SELECT " # ensure space if only SELECT

    params = {}
    if facility_id:
        if isinstance(facility_id, list) and len(facility_id) > 0:
            in_clause = ", ".join([f"'{f}'" for f in facility_id])
            query_parts.append(f"AND c.facility_id IN ({in_clause})")
        elif isinstance(facility_id, str) and facility_id:
            query_parts.append("AND c.facility_id = %(facility_id_val)s")
            params['facility_id_val'] = facility_id

    if start_date:
        query_parts.append("AND cli.service_from_date >= %(start_date)s"); params['start_date'] = start_date
    if end_date:
        query_parts.append("AND cli.service_from_date <= %(end_date)s"); params['end_date'] = end_date

    query_parts.append("GROUP BY p.rendering_provider_id, p.first_name, p.last_name")
    query_parts.append("ORDER BY total_provider_charges DESC")
    query = " ".join(query_parts)
    try:
        print(f"Executing get_provider_summary_analytics: {query[:400]}... with params: {params}")
        df = pd.read_sql_query(query, engine, params=params if params else None); return df
    except Exception as e: print(f"Error: {e}"); return pd.DataFrame()

# --- Functions to fetch distinct values for filters ---

def get_distinct_orgs():
    engine = get_sqlserver_engine()
    if engine is None: return []
    query = "SELECT DISTINCT org_id, org_name FROM dbo.facility_organization ORDER BY org_name;"
    try:
        df = pd.read_sql_query(query, engine)
        return [{'label': row['org_name'], 'value': row['org_id']} for _, row in df.iterrows()]
    except Exception as e: print(f"Error: {e}"); return []

def get_distinct_regions():
    engine = get_sqlserver_engine()
    if engine is None: return []
    query = "SELECT DISTINCT region_id, region_name FROM dbo.facility_region ORDER BY region_name;"
    try:
        df = pd.read_sql_query(query, engine)
        return [{'label': row['region_name'], 'value': row['region_id']} for _, row in df.iterrows()]
    except Exception as e: print(f"Error: {e}"); return []


if __name__ == '__main__':
    print("Testing metrics_queries.py...")

    print("\n--- Testing get_provider_summary_analytics ---")
    provider_summary_df = get_provider_summary_analytics(top_n=3, facility_id='1') # Example with single facility
    if not provider_summary_df.empty:
        print("Sample of provider summary analytics data:")
        print(provider_summary_df.head())
    else:
        print("No provider summary analytics data returned (or an error occurred).")

    print("\n--- Testing get_distinct_orgs ---")
    orgs = get_distinct_orgs()
    if orgs: print(f"Found {len(orgs)} orgs. First 3: {orgs[:3]}")
    else: print("No orgs found or error occurred.")

    print("\n--- Testing get_distinct_regions ---")
    regions = get_distinct_regions()
    if regions: print(f"Found {len(regions)} regions. First 3: {regions[:3]}")
    else: print("No regions found or error occurred.")

    # ... (keep other existing test calls) ...

    print("\n--- Testing get_daily_processing_summary_metrics ---")
    daily_summary_df = get_daily_processing_summary_metrics(start_date='2023-01-01', end_date='2023-01-31', facility_id='1')
    if not daily_summary_df.empty: print(daily_summary_df.head())

    print("\n--- Testing get_org_level_processing_metrics ---")
    org_metrics_df = get_org_level_processing_metrics(start_date='2023-01-01', end_date='2023-03-31')
    if not org_metrics_df.empty: print(org_metrics_df.head())

    print("\n--- Testing get_cpt_code_analytics ---")
    cpt_analytics_df = get_cpt_code_analytics(top_n=5, facility_id='1')
    if not cpt_analytics_df.empty: print(cpt_analytics_df.head())

    print("\n--- Testing get_diagnosis_code_analytics ---")
    diag_analytics_df = get_diagnosis_code_analytics(top_n=5, facility_id=['1','2']) # Example with list
    if not diag_analytics_df.empty: print(diag_analytics_df.head())

    print("\n--- Testing get_payer_analytics ---")
    payer_analytics_df = get_payer_analytics(facility_id='1')
    if not payer_analytics_df.empty: print(payer_analytics_df.head())

    print("\n--- Testing get_patient_demographics ---")
    demographics_df = get_patient_demographics(facility_id='1')
    if not demographics_df.empty: print(demographics_df.head())

    print("\nMetrics queries tests finished.")
