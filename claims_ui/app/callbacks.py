from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px

# --- Layout Imports ---
from .layouts.processed_claims_layout import create_processed_claims_layout
from .layouts.failed_claims_layout import create_failed_claims_layout
from .layouts.processing_metrics_layout import create_processing_metrics_layout
from .layouts.healthcare_analytics_layout import create_healthcare_analytics_layout

# --- Data Query Imports ---
# Make sure these paths are correct for your project structure
from .queries.claims_queries import (
    get_processed_claims, get_failed_claims, # 'get_claim_details' REMOVED as it's not needed
    get_payer_options, get_facility_options, get_failure_category_options
)
from .queries.metrics_queries import (
    get_kpi_data, get_daily_trends, get_facility_comparison, 
    get_payer_distribution, get_claim_type_distribution
)
from .queries.analytics_queries import (
    get_top_cpt_codes, get_top_dx_codes, get_payer_analysis,
    get_patient_demographics, get_provider_metrics
)


def register_callbacks(app):
    """Registers all callbacks for the application."""

    # =========================================================================
    # Page Routing and Navigation
    # =========================================================================

    @app.callback(
        [Output('page-content', 'children'),
         Output('content-title', 'children')],
        [Input('url', 'pathname')]
    )
    def display_page(pathname):
        """Renders the correct page layout based on the URL."""
        if pathname == '/processed':
            return create_processed_claims_layout(), "Processed Claims"
        elif pathname == '/failed':
            return create_failed_claims_layout(), "Failed Claims"
        elif pathname == '/analytics':
            return create_healthcare_analytics_layout(), "Healthcare Analytics"
        # Default to processing metrics for the root URL
        return create_processing_metrics_layout(), "Processing Metrics"

    @app.callback(
        [Output(f"nav-link-{page}", "className") for page in ["home", "analytics", "processed", "failed"]],
        [Input("url", "pathname")]
    )
    def update_active_links(pathname):
        """Updates the 'active' className for the current page's sidebar link."""
        links_mapping = {
            "/": "home",
            "/analytics": "analytics",
            "/processed": "processed",
            "/failed": "failed"
        }
        class_names = {page: "nav-link" for page in links_mapping.values()}
        current_page = links_mapping.get(pathname)
        if current_page:
            class_names[current_page] = "nav-link active"
        
        # Ensure the order matches the list of Outputs
        return [class_names["home"], class_names["analytics"], class_names["processed"], class_names["failed"]]


    # =========================================================================
    # Processing Metrics Page Callbacks
    # =========================================================================

    @app.callback(
        [Output('kpi-summary-pm', 'children'),
         Output('daily-trends-graph-pm', 'figure'),
         Output('facility-comparison-graph-pm', 'figure'),
         Output('payer-distribution-graph-pm', 'figure'),
         Output('claim-type-distribution-graph-pm', 'figure')],
        [Input('apply-filters-pm-button', 'n_clicks')],
        [State('date-picker-range-pm', 'start_date'),
         State('date-picker-range-pm', 'end_date'),
         State('facility-filter-pm', 'value')]
    )
    def update_processing_metrics(n_clicks, start_date, end_date, facility_ids):
        """Updates all components on the processing metrics tab."""
        if n_clicks is None:
            raise PreventUpdate # Don't run on initial load

        # 1. KPI Cards
        kpi_data = get_kpi_data(start_date, end_date, facility_ids)
        kpi_cards = [
            html.Div([
                html.H5(kpi['name']),
                html.P(f"{kpi.get('value', 0):,}")
            ], className='kpi-card') for kpi in kpi_data
        ]

        # 2. Daily Trends Graph
        daily_trends_df = get_daily_trends(start_date, end_date, facility_ids)
        daily_trends_fig = px.line(
            daily_trends_df, x='date', y='claims_processed',
            title='Daily Claims Processing Volume', template='plotly_white'
        )

        # 3. Facility Comparison Graph
        facility_comp_df = get_facility_comparison(start_date, end_date, facility_ids)
        facility_comp_fig = px.bar(
            facility_comp_df, x='facility_name', y='total_billed_amount',
            title='Billed Amount by Facility', template='plotly_white'
        )
        
        # 4. Payer Distribution Graph
        payer_dist_df = get_payer_distribution(start_date, end_date, facility_ids)
        payer_dist_fig = px.pie(
            payer_dist_df, names='payer_name', values='claim_count',
            title='Claim Distribution by Payer', hole=0.4, template='plotly_white'
        )

        # 5. Claim Type Distribution
        claim_type_df = get_claim_type_distribution(start_date, end_date, facility_ids)
        claim_type_fig = px.bar(
            claim_type_df, x='claim_type', y='count',
            title='Professional vs. Institutional Claims', template='plotly_white'
        )

        return kpi_cards, daily_trends_fig, facility_comp_fig, payer_dist_fig, claim_type_fig


    # =========================================================================
    # Processed Claims Page Callbacks
    # =========================================================================

    @app.callback(
        Output('processed-claims-table', 'data'),
        [Input('apply-filters-pc-button', 'n_clicks')],
        [State('facility-filter-pc', 'value'),
         State('payer-filter-pc', 'value'),
         State('date-picker-range-pc', 'start_date'),
         State('date-picker-range-pc', 'end_date')]
    )
    def update_processed_claims_table(n_clicks, facilities, payers, start_date, end_date):
        """Fetches and displays processed claims data in the table."""
        if n_clicks is None:
            raise PreventUpdate
        claims_data = get_processed_claims(facilities, payers, start_date, end_date)
        return claims_data
    
    @app.callback(
        Output('selected-claim-details-pc', 'children'),
        [Input('processed-claims-table', 'active_cell')],
        [State('processed-claims-table', 'data')]
    )
    def display_claim_details(active_cell, rows):
        """
        Displays details of the selected claim from the table's existing data,
        without requiring a new database query.
        """
        if not active_cell or not rows:
            return html.Div([html.H4("Claim Details"), html.P("Select a claim to see details.")])
        
        # Get the data for the selected row from the existing table data
        selected_row_data = rows[active_cell['row']]
        
        if not selected_row_data:
            return html.Div([html.H4("Claim Details"), html.P("Could not find details for the selected claim.")])
            
        # Build the details layout from the row data
        details_layout = [
            html.H4("Claim Details"),
            html.P(f"Claim ID: {selected_row_data.get('claim_id', 'N/A')}"),
            html.P(f"Patient Name: {selected_row_data.get('patient_name', 'N/A')}"),
            html.P(f"Billed Amount: ${selected_row_data.get('total_billed_amount', 0):,.2f}"),
            html.P(f"Paid Amount: ${selected_row_data.get('total_paid_amount', 0):,.2f}"),
            html.P(f"Payer: {selected_row_data.get('payer_name', 'N/A')}"),
            html.P(f"Status: {selected_row_data.get('claim_status', 'N/A')}"),
        ]
        return html.Div(details_layout)


    # =========================================================================
    # Failed Claims Page Callbacks
    # =================================e========================================
    
    # (Placeholder for failed claims callbacks - they would follow a similar pattern)
    # ...


    # =========================================================================
    # Healthcare Analytics Page Callbacks
    # =========================================================================

    @app.callback(
        Output('healthcare-analytics-content', 'children'),
        [Input('healthcare-analytics-subtabs', 'value'),
         Input('apply-filters-ha-button', 'n_clicks')],
        [State('date-picker-range-ha', 'start_date'),
         State('date-picker-range-ha', 'end_date'),
         State('facility-filter-ha', 'value'),
         State('top-n-filter-ha', 'value')]
    )
    def render_analytics_content(subtab, n_clicks, start_date, end_date, facility_ids, top_n):
        """Renders the content for the selected analytics sub-tab."""
        if n_clicks is None:
            return html.P("Apply filters to see analytics content.")

        if subtab == 'subtab-cpt':
            df = get_top_cpt_codes(start_date, end_date, facility_ids, top_n)
            fig = px.bar(df, x='cpt_code', y='count', title=f'Top {top_n} CPT Codes')
            return dcc.Graph(figure=fig)
        elif subtab == 'subtab-dx':
            df = get_top_dx_codes(start_date, end_date, facility_ids, top_n)
            fig = px.bar(df, x='dx_code', y='count', title=f'Top {top_n} Diagnosis Codes')
            return dcc.Graph(figure=fig)
        
        return html.P(f"Content for {subtab}")
        
    print("Callbacks registered successfully.")
