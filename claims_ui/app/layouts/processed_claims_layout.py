from dash import dcc, html, dash_table

def create_processed_claims_layout():
    """Creates the layout for the Processed Claims Dashboard using card styling."""
    return html.Div([
        # Filters Card
        html.Div([
            html.Div(html.H4("Processed Claims Filters"), className="dashboard-card-header"),
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
        ], className='dashboard-card'),

        # This outer Div will hold the table and the details side-by-side
        html.Div([
            # Data Table Card (takes up more space)
            html.Div([
                dash_table.DataTable(
                    id='processed-claims-table',
                    columns=[],
                    data=[],
                    # COMMON_DATATABLE_PROPS are applied in callbacks.py
                )
            ], className='dashboard-card table-container', style={'flex': '3', 'marginRight': '24px'}),
            
            # Selected Row Details Card (takes up less space)
            html.Div(
                id='selected-claim-details-pc', 
                className='dashboard-card details-container',
                style={'flex': '1'}
            )
        ], style={'display': 'flex', 'flexDirection': 'row'})
    ])
