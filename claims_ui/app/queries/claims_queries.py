# Contains SQL queries related to claims data
import pandas as pd
from claims_ui.app.db_connector import get_sqlserver_engine # Assuming db_connector is in the same app directory

def get_processed_claims(facility_id=None, payer_id=None, start_date=None, end_date=None, limit=100):
    """
    Retrieves processed claims data by joining various related tables.
    Filters can be applied for facility, payer, date range, and a limit can be set.
    Pulls the primary diagnosis based on diagnosis_sequence = 1.
    """
    engine = get_sqlserver_engine()
    if engine is None:
        print("Database engine not available. Cannot fetch processed claims.")
        return pd.DataFrame()

    query_parts = [
        "SELECT",
        # Add TOP clause if limit is specified and is a positive integer
        f"TOP {int(limit)}" if limit and isinstance(limit, (int, float)) and limit > 0 else "",
        """
        c.patient_account_number, c.medical_record_number, c.patient_name, c.date_of_birth, c.gender, c.created_at AS claim_date,
        cli.line_number, cli.procedure_code, cli.modifier1, cli.units, cli.charge_amount, cli.service_from_date, cli.rvu_value, cli.reimbursement_amount,
        cd.diagnosis_code AS primary_diagnosis_code, cd.diagnosis_description AS primary_diagnosis_description,
        p.first_name AS provider_first_name, p.last_name AS provider_last_name,
        csp.payer_name
        FROM dbo.claims c
        JOIN dbo.claims_line_items cli ON c.facility_id = cli.facility_id AND c.patient_account_number = cli.patient_account_number
        -- Join for primary diagnosis (diagnosis_sequence = 1)
        LEFT JOIN dbo.claims_diagnosis cd ON c.facility_id = cd.facility_id AND c.patient_account_number = cd.patient_account_number AND cd.diagnosis_sequence = 1
        LEFT JOIN dbo.physicians p ON cli.rendering_provider_id = p.rendering_provider_id
        JOIN dbo.facility_financial_classes ffc ON c.facility_id = ffc.facility_id AND c.financial_class_id = ffc.financial_class_id
        JOIN dbo.core_standard_payers csp ON ffc.payer_id = csp.payer_id
        WHERE 1=1
        """
    ]

    # Remove empty strings from query_parts (e.g. if TOP was not added)
    query_parts = [part for part in query_parts if part]

    params = {}

    if facility_id:
        query_parts.append("AND c.facility_id = %(facility_id)s")
        params['facility_id'] = facility_id
    if payer_id:
        # Assuming csp.payer_id is the correct field for filtering by payer_id from parameters
        query_parts.append("AND csp.payer_id = %(payer_id)s")
        params['payer_id'] = payer_id
    if start_date:
        query_parts.append("AND c.created_at >= %(start_date)s")
        params['start_date'] = start_date
    if end_date:
        query_parts.append("AND c.created_at <= %(end_date)s")
        params['end_date'] = end_date

    query = " ".join(query_parts)

    # Add ORDER BY clause, e.g. by claim_date descending, if no other ordering is implied by TOP
    # For SQL Server, TOP without ORDER BY is non-deterministic for which rows are returned
    # if not (limit and isinstance(limit, (int, float)) and limit > 0): # Or if you always want an order
    query += " ORDER BY c.created_at DESC"


    try:
        print(f"Executing get_processed_claims query: {query[:500]}... with params: {params}") # Log truncated query
        df = pd.read_sql_query(query, engine, params=params)
        print(f"Successfully fetched {len(df)} processed claims.")
        return df
    except Exception as e:
        print(f"Error fetching processed claims: {e}")
        return pd.DataFrame()

def get_failed_claims(facility_id=None, failure_category=None, start_date=None, end_date=None, limit=100):
    """
    Retrieves failed claims data, optionally filtered by facility, failure category, date range, and limit.
    """
    engine = get_sqlserver_engine()
    if engine is None:
        print("Database engine not available. Cannot fetch failed claims.")
        return pd.DataFrame()

    query_parts = [
        "SELECT",
        f"TOP {int(limit)}" if limit and isinstance(limit, (int, float)) and limit > 0 else "",
        """
        fc.claim_id, fc.facility_id, f.facility_name, fc.patient_account_number,
        fc.failure_reason, fc.failure_category, fc.processing_stage,
        fc.failed_at, fc.resolution_status, fc.potential_revenue_loss, fc.coder_id
        FROM dbo.failed_claims fc
        LEFT JOIN dbo.facilities f ON fc.facility_id = f.facility_id
        WHERE 1=1
        """
    ]
    query_parts = [part for part in query_parts if part]
    params = {}

    if facility_id:
        query_parts.append("AND fc.facility_id = %(facility_id)s")
        params['facility_id'] = facility_id
    if failure_category:
        query_parts.append("AND fc.failure_category = %(failure_category)s")
        params['failure_category'] = failure_category
    if start_date:
        query_parts.append("AND fc.failed_at >= %(start_date)s")
        params['start_date'] = start_date
    if end_date:
        query_parts.append("AND fc.failed_at <= %(end_date)s")
        params['end_date'] = end_date

    query = " ".join(query_parts)
    query += " ORDER BY fc.failed_at DESC"


    try:
        print(f"Executing get_failed_claims query: {query[:500]}... with params: {params}") # Log truncated query
        df = pd.read_sql_query(query, engine, params=params)
        print(f"Successfully fetched {len(df)} failed claims.")
        return df
    except Exception as e:
        print(f"Error fetching failed claims: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    print("Testing claims_queries.py...")

    # Note: These tests will attempt to connect to the database.
    # Ensure your db_connector.py has valid (even if placeholder) credentials
    # or is mocked, otherwise these will likely fail if the DB is not accessible.

    print("\n--- Testing get_distinct_facilities ---")
    facilities = get_distinct_facilities()
    if facilities:
        print(f"Found {len(facilities)} facilities. First 3: {facilities[:3]}")
    else:
        print("No facilities found or error occurred.")

    print("\n--- Testing get_distinct_payers ---")
    payers = get_distinct_payers()
    if payers:
        print(f"Found {len(payers)} payers. First 3: {payers[:3]}")
    else:
        print("No payers found or error occurred.")

    print("\n--- Testing get_distinct_failure_categories ---")
    failure_categories = get_distinct_failure_categories()
    if failure_categories:
        print(f"Found {len(failure_categories)} failure categories. First 3: {failure_categories[:3]}")
    else:
        print("No failure categories found or error occurred.")

    print("\n--- Testing get_processed_claims ---")
    # Example: Get top 5 processed claims (no filters)
    processed_claims_df = get_processed_claims(limit=5)
    if not processed_claims_df.empty:
        print("Sample of processed claims data:")
        print(processed_claims_df.head())
        print(f"Shape of processed_claims_df: {processed_claims_df.shape}")
    else:
        print("No processed claims data returned (or an error occurred).")

    print("\n--- Testing get_failed_claims ---")
    # Example: Get top 5 failed claims for a specific category (if you have one)
    # failed_claims_df = get_failed_claims(failure_category="Eligibility", limit=5)
    failed_claims_df = get_failed_claims(limit=5) # Test without specific category for now
    if not failed_claims_df.empty:
        print("Sample of failed claims data:")
        print(failed_claims_df.head())
        print(f"Shape of failed_claims_df: {failed_claims_df.shape}")
    else:
        print("No failed claims data returned (or an error occurred).")

    print("\nClaims queries tests finished.")
