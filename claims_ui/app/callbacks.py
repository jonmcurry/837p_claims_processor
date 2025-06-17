from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
from datetime import date

# --- Layout Imports ---
from .layouts.processed_claims_layout import create_processed_claims_layout
from .layouts.failed_claims_layout import create_failed_claims_layout
from .layouts.processing_metrics_layout import create_processing_metrics_layout
from .layouts.healthcare_analytics_layout import create_healthcare_analytics_layout

# --- Data Query Imports ---
# CORRECTED: Only importing modules that exist to prevent import errors.
# We will use placeholder data in the callbacks instead of calling functions from here for now.
from .queries import claims_queries, metrics_queries

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
        return create_processing_metrics_layout(), "Processing Metrics"

    @app.callback(
        [Output(f"nav-link-{page}", "className") for page in ["home", "analytics", "processed", "failed"]],
        [Input("url", "pathname")]
    )
    def update_active_links(pathname):
        """Updates the 'active' className for the current page's sidebar link."""
        links_mapping = {"/": "home", "/analytics": "analytics", "/processed": "processed", "/failed": "failed"}
        class_names = {page: "nav-link" for page in links_mapping.values()}
        current_page = links_mapping.get(pathname)
        if current_page:
            class_names[current_page] = "nav-link active"
        return [class_names["home"], class_names["analytics"], class_names["processed"], class_names["failed"]]

    # =========================================================================
    # Dropdown Population Callbacks
    # =========================================================================
    
    @app.callback(
        [Output('facility-filter-pc', 'options'),
         Output('payer-filter-pc', 'options'),
         Output('facility-filter-pm', 'options'),
         Output('facility-filter-ha', 'options')],
        [Input('url', 'pathname')]
    )
    def populate_all_filters(pathname):
        """Populates all dropdowns with placeholder data when the app loads."""
        facility_options = [{'label': 'Main General Hospital', 'value': 'fac-1'}, {'label': 'Downtown Clinic', 'value': 'fac-2'}]
        payer_options = [{'label': 'Blue Shield', 'value': 'pay-1'}, {'label': 'United Health', 'value': 'pay-2'}]
        # This will populate dropdowns on all pages as they share some filter IDs
        return facility_options, payer_options, facility_options, facility_options

    # =========================================================================
    # Processing Metrics Page Callbacks (Main Dashboard)
    # =========================================================================

    @app.callback(
        [Output('kpi-summary-pm', 'children'),
         Output('daily-trends-graph-pm', 'figure'),
         Output('facility-comparison-graph-pm', 'figure'),
         Output('payer-distribution-graph-pm', 'figure'),
         Output('claim-type-distribution-graph-pm', 'figure')],
        [Input('url', 'pathname'), # *** ADDED THIS INPUT TO TRIGGER ON PAGE LOAD ***
         Input('apply-filters-pm-button', 'n_clicks')]
    )
    def update_processing_metrics(pathname, n_clicks):
        """Updates all components on the processing metrics tab with placeholder data."""
        # This will now run when the page loads to '/' or when the button is clicked.
        
        # 1. KPI Cards
        kpi_data = [
            {'name': 'Total Claims Processed', 'value': 12245},
            {'name': 'Total Billed Amount', 'value': '3.2M'},
            {'name': 'Acceptance Rate', 'value': '92%'},
            {'name': 'Avg. Processing Time', 'value': '1.8d'}
        ]
        kpi_cards = [
            html.Div([html.H5(kpi['name']), html.P(kpi['value'])], className='kpi-card') for kpi in kpi_data
        ]

        # 2. Daily Trends Graph
        trends_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-05-01', '2023-05-02', '2023-05-03', '2023-05-04', '2023-05-05', '2023-05-06', '2023-05-07']),
            'claims_processed': [250, 310, 280, 350, 400, 380, 410]
        })
        daily_trends_fig = px.area(trends_df, x='date', y='claims_processed', title='Daily Claims Volume').update_layout(template='plotly_white')

        # 3. Facility Comparison Graph
        facility_df = pd.DataFrame({'Facility': ['Main General', 'Downtown Clinic', 'Uptown Medical'], 'Billed Amount': [1800000, 1400000, 950000]})
        facility_comp_fig = px.bar(facility_df, x='Facility', y='Billed Amount', title='Billed Amount by Facility').update_layout(template='plotly_white')
        
        # 4. Payer Distribution Graph
        payer_df = pd.DataFrame({'Payer': ['Blue Shield', 'United Health', 'Aetna', 'Cigna'], 'Claims': [5000, 4500, 2745, 1900]})
        payer_dist_fig = px.pie(payer_df, names='Payer', values='Claims', title='Claims by Payer', hole=0.4).update_layout(template='plotly_white')

        # 5. Claim Type Distribution
        type_df = pd.DataFrame({'Claim Type': ['Professional', 'Institutional'], 'Count': [8500, 3745]})
        claim_type_fig = px.bar(type_df, x='Count', y='Claim Type', orientation='h', title='Claim Types').update_layout(template='plotly_white')

        return kpi_cards, daily_trends_fig, facility_comp_fig, payer_dist_fig, claim_type_fig

    # =========================================================================
    # Processed Claims Page Callbacks
    # =========================================================================

    @app.callback(
        Output('processed-claims-table', 'data'),
        [Input('apply-filters-pc-button', 'n_clicks')]
    )
    def update_processed_claims_table(n_clicks):
        if n_clicks is None:
            return [] # Return an empty list initially
        placeholder_data = [
            {'claim_id': 'C-001', 'patient_name': 'John Smith', 'total_billed_amount': 550.75, 'total_paid_amount': 450.00, 'payer_name': 'Blue Shield', 'claim_status': 'Paid'},
            {'claim_id': 'C-002', 'patient_name': 'Jane Doe', 'total_billed_amount': 1200.00, 'total_paid_amount': 950.00, 'payer_name': 'United Health', 'claim_status': 'Paid'},
        ]
        return placeholder_data
    
    @app.callback(
        Output('selected-claim-details-pc', 'children'),
        [Input('processed-claims-table', 'active_cell')],
        [State('processed-claims-table', 'data')]
    )
    def display_claim_details(active_cell, rows):
        if not active_cell or not rows:
            return html.Div([html.H4("Claim Details"), html.P("Select a claim to see details.")])
        selected_row_data = rows[active_cell['row']]
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
    # Healthcare Analytics Page Callbacks
    # =========================================================================

    @app.callback(
        Output('healthcare-analytics-content', 'children'),
        [Input('healthcare-analytics-subtabs', 'value'),
         Input('apply-filters-ha-button', 'n_clicks')]
    )
    def render_analytics_content(subtab, n_clicks):
        if n_clicks is None:
            return html.P("Apply filters to see analytics content.", style={'padding': '20px'})
            
        if subtab == 'subtab-cpt':
            df = pd.DataFrame({'cpt_code': ['99213', '99214', '99396', '99203', '99212'], 'count': [500, 420, 310, 250, 180]})
            fig = px.bar(df, x='cpt_code', y='count', title=f'Top 5 CPT Codes').update_layout(template='plotly_white')
            return dcc.Graph(figure=fig)
        elif subtab == 'subtab-dx':
            df = pd.DataFrame({'dx_code': ['I10', 'E11.9', 'Z00.00', 'K21.9', 'M54.5'], 'count': [610, 550, 480, 320, 290]})
            fig = px.bar(df, x='dx_code', y='count', title=f'Top 5 Diagnosis Codes').update_layout(template='plotly_white')
            return dcc.Graph(figure=fig)
        
        return html.P(f"Placeholder content for {subtab}", style={'padding': '20px'})
        
    print("Callbacks registered successfully.")
