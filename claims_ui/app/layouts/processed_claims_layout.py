from dash import dcc, html, dash_table

def create_processed_claims_layout():
    """Creates the layout for the Processed Claims Dashboard."""
    return html.Div([
        # Filters Card
        html.Div([
            html.H4("Processed Claims Filters"), # This H4 will be styled by .dashboard-card > h4
            html.Div([
                dcc.Dropdown(
                    id='facility-filter-pc',
                    placeholder='Select Facility(s)...',
                    multi=True,
                ),
            ], className='form-element-wrapper'),
            html.Div([
                dcc.Dropdown(
                    id='payer-filter-pc',
                    placeholder='Select Payer(s)...',
                    multi=True,
                ),
            ], className='form-element-wrapper'),
            html.Div([
                dcc.DatePickerRange(
                    id='date-picker-range-pc',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                    display_format='YYYY-MM-DD',
                ),
            ], className='form-element-wrapper'),
            html.Button('Apply Filters', id='apply-filters-pc-button', n_clicks=0)
        ], className='dashboard-card filter-section'),

        # Data Table Card
        html.Div([
            # Optional: html.H5("Claims Data", className="card-header-title"), # Example of a card header title
            dash_table.DataTable(
                id='processed-claims-table',
                columns=[],  # Populated by callback
                data=[],     # Populated by callback
                # COMMON_DATATABLE_PROPS are applied in callbacks.py
            )
        ], className='dashboard-card table-container'),

        # Selected Row Details Card
        html.Div(
            id='selected-claim-details-pc',
            className='dashboard-card details-container' # Apply card style
        )
    ])