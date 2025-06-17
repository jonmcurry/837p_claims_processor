from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Layout Imports ---
from .layouts.processed_claims_layout import create_processed_claims_layout
from .layouts.failed_claims_layout import create_failed_claims_layout
from .layouts.processing_metrics_layout import create_processing_metrics_layout
from .layouts.healthcare_analytics_layout import create_healthcare_analytics_layout

# --- Data Query Imports ---
from .queries import claims_queries, metrics_queries

# --- Helper Function for Empty Charts ---
def create_empty_figure(message="No data available for the selected filters."):
    """Creates a blank Plotly figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{
            "text": message, "xref": "paper", "yref": "paper",
            "showarrow": False, "font": {"size": 14, "color": "#6c757d"}
        }]
    )
    return fig

def register_callbacks(app):
    """Registers all callbacks for the application."""

    # =========================================================================
    # Page Routing and Navigation
    # =========================================================================
    @app.callback(
        [Output('page-content', 'children'), Output('content-title', 'children')],
        [Input('url', 'pathname')]
    )
    def display_page(pathname):
        """Renders the correct page layout based on the URL."""
        if pathname == '/processed': return create_processed_claims_layout(), "Processed Claims"
        if pathname == '/failed': return create_failed_claims_layout(), "Failed Claims Analysis"
        if pathname == '/analytics': return create_healthcare_analytics_layout(), "Healthcare Analytics"
        return create_processing_metrics_layout(), "Main Dashboard"

    @app.callback(
        [Output(f"nav-link-{page}", "className") for page in ["home", "analytics", "processed", "failed"]],
        [Input("url", "pathname")]
    )
    def update_active_links(pathname):
        """Updates the 'active' className for the current page's sidebar link."""
        links_mapping = {"/": "home", "/analytics": "analytics", "/processed": "processed", "/failed": "failed"}
        class_names = {page: "nav-link" for page in links_mapping.values()}
        current_page = links_mapping.get(pathname)
        if current_page: class_names[current_page] = "nav-link active"
        return [class_names["home"], class_names["analytics"], class_names["processed"], class_names["failed"]]

    # =========================================================================
    # Dropdown Population Callbacks (SEPARATED FOR EACH PAGE)
    # =========================================================================
    @app.callback(Output('facility-filter-pm', 'options'), [Input('url', 'pathname')])
    def populate_pm_filters(pathname):
        if pathname != '/': raise PreventUpdate
        return [{'label': 'Main General Hospital', 'value': 'fac-1'}, {'label': 'Downtown Clinic', 'value': 'fac-2'}]

    @app.callback([Output('facility-filter-pc', 'options'), Output('payer-filter-pc', 'options')], [Input('url', 'pathname')])
    def populate_pc_filters(pathname):
        if pathname != '/processed': raise PreventUpdate
        facility_options = [{'label': 'Main General Hospital', 'value': 'fac-1'}, {'label': 'Downtown Clinic', 'value': 'fac-2'}]
        payer_options = [{'label': 'Blue Shield', 'value': 'pay-1'}, {'label': 'United Health', 'value': 'pay-2'}]
        return facility_options, payer_options

    @app.callback([Output('facility-filter-fc', 'options'), Output('failure-category-filter-fc', 'options')], [Input('url', 'pathname')])
    def populate_fc_filters(pathname):
        if pathname != '/failed': raise PreventUpdate
        facility_options = [{'label': 'Main General Hospital', 'value': 'fac-1'}, {'label': 'Downtown Clinic', 'value': 'fac-2'}]
        failure_options = [{'label': 'Invalid Member ID', 'value': 'fail-1'}, {'label': 'Service Not Covered', 'value': 'fail-2'}]
        return facility_options, failure_options
    
    @app.callback(Output('facility-filter-ha', 'options'), [Input('url', 'pathname')])
    def populate_ha_filters(pathname):
        if pathname != '/analytics': raise PreventUpdate
        return [{'label': 'Main General Hospital', 'value': 'fac-1'}, {'label': 'Downtown Clinic', 'value': 'fac-2'}]

    # =========================================================================
    # Main Dashboard Page Callbacks
    # =========================================================================
    @app.callback(
        [Output('kpi-summary-pm', 'children'), Output('daily-trends-graph-pm', 'figure'),
         Output('facility-comparison-graph-pm', 'figure'), Output('payer-distribution-graph-pm', 'figure'),
         Output('claim-type-distribution-graph-pm', 'figure')],
        [Input('url', 'pathname'), Input('apply-filters-pm-button', 'n_clicks')]
    )
    def update_processing_metrics(pathname, n_clicks):
        if pathname != '/': raise PreventUpdate
        kpi_data = [{'name': 'Total Claims', 'value': 12245}, {'name': 'Billed Amount', 'value': '3.2M'}, {'name': 'Acceptance %', 'value': '92%'}, {'name': 'Avg. Time', 'value': '1.8d'}]
        kpi_cards = [html.Div([html.H5(kpi['name']), html.P(kpi['value'])], className='kpi-card') for kpi in kpi_data]
        trends_df = pd.DataFrame({'date': pd.to_datetime(['2023-05-01', '2023-05-02', '2023-05-03', '2023-05-04', '2023-05-05']),'claims_processed': [250, 310, 280, 350, 400]})
        daily_trends_fig = px.area(trends_df, x='date', y='claims_processed', title='Daily Claims Volume').update_layout(template='plotly_white')
        facility_df = pd.DataFrame({'Facility': ['Main General', 'Downtown Clinic', 'Uptown Medical'], 'Billed Amount': [1.8, 1.4, 0.95]})
        facility_comp_fig = px.bar(facility_df, x='Facility', y='Billed Amount', title='Billed Amount by Facility (Millions)', text_auto='.2s').update_layout(template='plotly_white')
        payer_df = pd.DataFrame({'Payer': ['Blue Shield', 'United Health', 'Aetna', 'Cigna'], 'Claims': [5000, 4500, 2745, 1900]})
        payer_dist_fig = px.pie(payer_df, names='Payer', values='Claims', title='Claims by Payer', hole=0.4).update_layout(template='plotly_white')
        type_df = pd.DataFrame({'Claim Type': ['Professional', 'Institutional'], 'Count': [8500, 3745]})
        claim_type_fig = px.bar(type_df, x='Count', y='Claim Type', orientation='h', title='Claim Types', text_auto=True).update_layout(template='plotly_white')
        return kpi_cards, daily_trends_fig, facility_comp_fig, payer_dist_fig, claim_type_fig

    # =========================================================================
    # Processed Claims Page Callbacks
    # =========================================================================
    @app.callback(
        [Output('processed-claims-table', 'data'), Output('processed-claims-table', 'columns')],
        [Input('url', 'pathname'), Input('apply-filters-pc-button', 'n_clicks')]
    )
    def update_processed_claims_table(pathname, n_clicks):
        if pathname != '/processed': raise PreventUpdate
        cols = ["Claim ID", "Patient Name", "Billed Amount", "Paid Amount", "Payer", "Status"]
        columns = [{"name": i, "id": i.lower().replace(" ", "_")} for i in cols]
        placeholder_data = [
            {'claim_id': 'C-001', 'patient_name': 'John Smith', 'billed_amount': 550.75, 'paid_amount': 450.00, 'payer': 'Blue Shield', 'status': 'Paid'},
            {'claim_id': 'C-002', 'patient_name': 'Jane Doe', 'billed_amount': 1200.00, 'paid_amount': 950.00, 'payer': 'United Health', 'status': 'Paid'},
        ]
        return placeholder_data, columns
    
    @app.callback(Output('selected-claim-details-pc', 'children'), [Input('processed-claims-table', 'active_cell')], [State('processed-claims-table', 'data')])
    def display_claim_details(active_cell, rows):
        if not active_cell or not rows: return html.Div([html.H4("Claim Details"), html.P("Select a claim to see details.")], className="no-data-placeholder")
        selected_row_data = rows[active_cell['row']]
        details_layout = [html.H4("Claim Details"), html.P(f"Claim ID: {selected_row_data.get('claim_id', 'N/A')}"), html.P(f"Patient Name: {selected_row_data.get('patient_name', 'N/A')}"), html.P(f"Billed Amount: ${selected_row_data.get('billed_amount', 0):,.2f}"), html.P(f"Paid Amount: ${selected_row_data.get('paid_amount', 0):,.2f}")]
        return html.Div(details_layout)

    # =========================================================================
    # Failed Claims Page Callbacks
    # =========================================================================
    @app.callback(
        [Output('failed-claims-table', 'data'), Output('failed-claims-table', 'columns'), Output('failure-reason-bargraph-fc', 'figure')],
        [Input('url', 'pathname'), Input('apply-filters-fc-button', 'n_clicks')]
    )
    def update_failed_claims_page(pathname, n_clicks):
        if pathname != '/failed': raise PreventUpdate
        cols = ["Claim ID", "Patient Name", "Failure Reason", "Facility"]
        columns = [{"name": i, "id": i.lower().replace(" ", "_")} for i in cols]
        placeholder_data = [{'claim_id': 'C-004', 'patient_name': 'Mary Major', 'failure_reason': 'Invalid Member ID', 'facility': 'Main General Hospital'}, {'claim_id': 'C-005', 'patient_name': 'David Copper', 'failure_reason': 'Service Not Covered', 'facility': 'Downtown Clinic'}]
        reason_df = pd.DataFrame([{'Reason': 'Invalid Member ID', 'Count': 42}, {'Reason': 'Service Not Covered', 'Count': 25}, {'Reason': 'Duplicate Claim', 'Count': 15}])
        reason_fig = px.bar(reason_df, y='Reason', x='Count', title='Top Failure Reasons', orientation='h', text_auto=True).update_layout(template='plotly_white')
        return placeholder_data, columns, reason_fig

    # =========================================================================
    # Healthcare Analytics Page Callbacks
    # =========================================================================
    @app.callback(
        Output('healthcare-analytics-content', 'children'),
        [Input('url', 'pathname'), Input('healthcare-analytics-subtabs', 'value'), Input('apply-filters-ha-button', 'n_clicks')]
    )
    def render_analytics_content(pathname, subtab, n_clicks):
        if pathname != '/analytics': raise PreventUpdate
        if subtab == 'subtab-cpt':
            df = pd.DataFrame({'cpt_code': ['99213', '99214', '99396', '99203', '99212'], 'count': [500, 420, 310, 250, 180]})
            fig = px.bar(df, x='cpt_code', y='count', title=f'Top 5 CPT Codes', text_auto=True).update_layout(template='plotly_white')
            return dcc.Graph(figure=fig)
        elif subtab == 'subtab-dx':
            df = pd.DataFrame({'dx_code': ['I10', 'E11.9', 'Z00.00', 'K21.9', 'M54.5'], 'count': [610, 550, 480, 320, 290]})
            fig = px.bar(df, x='dx_code', y='count', title=f'Top 5 Diagnosis Codes', text_auto=True).update_layout(template='plotly_white')
            return dcc.Graph(figure=fig)
        return html.P(f"Placeholder content for {subtab}", style={'padding': '20px'})
        
    print("Callbacks registered successfully.")
