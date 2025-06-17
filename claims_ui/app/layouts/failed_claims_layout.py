from dash import dcc, html, dash_table

def create_failed_claims_layout():
    """Creates the layout for the Failed Claims Dashboard using card styling."""
    return html.Div([
        # Filters Card
        html.Div([
            html.Div(html.H4("Failed Claims Filters"), className="dashboard-card-header"),
            html.Div([
                dcc.Dropdown(id='facility-filter-fc', placeholder='Select Facility(s)...', multi=True),
            ], className='form-element-wrapper'),
            html.Div([
                dcc.Dropdown(id='failure-category-filter-fc', placeholder='Select Failure Category(s)...', multi=True),
            ], className='form-element-wrapper'),
            html.Div([
                 dcc.DatePickerRange(
                    id='date-picker-range-fc',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                    display_format='YYYY-MM-DD'
                ),
            ], className='form-element-wrapper'),
            html.Button('Apply Filters', id='apply-filters-fc-button', n_clicks=0)
        ], className='dashboard-card'),

        # Container for the two-column layout (Table/Details + Graph)
        html.Div([
             # Left Column: Table and Details
             html.Div([
                # Data Table Card
                html.Div([
                    dash_table.DataTable(
                        id='failed-claims-table',
                        columns=[],
                        data=[],
                    )
                ], className='dashboard-card table-container'),
                
                # Selected Row Details Card
                html.Div(
                    id='selected-failed-claim-details-fc', 
                    className='dashboard-card details-container'
                ),
             ], style={'flex': '2', 'marginRight': '24px'}),

            # Right Column: Graph
            html.Div([
                # Graph Card for Failure Analysis
                html.Div([
                    dcc.Graph(id='failure-reason-bargraph-fc')
                ], className='dashboard-card graph-container') 
            ], style={'flex': '1'})

        ], style={'display': 'flex', 'flexDirection': 'row'})
    ])
