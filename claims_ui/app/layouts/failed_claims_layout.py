from dash import dcc, html, dash_table

def create_failed_claims_layout():
    """Creates the layout for the Failed Claims Dashboard using card styling."""
    return html.Div([
        # This interval triggers the initial data load for this page.
        dcc.Interval(id='interval-fc', interval=500, max_intervals=1),
        
        # Filters Card
        html.Div([
            html.Div(html.H4("Failed Claims Filters"), className="dashboard-card-header"),
            html.Div([dcc.Dropdown(id='facility-filter-fc', placeholder='Select Facility(s)...', multi=True)], className='form-element-wrapper'),
            html.Div([dcc.Dropdown(id='failure-category-filter-fc', placeholder='Select Failure Category(s)...', multi=True)], className='form-element-wrapper'),
            html.Div([dcc.DatePickerRange(id='date-picker-range-fc', start_date_placeholder_text='Start Date', end_date_placeholder_text='End Date')], className='form-element-wrapper'),
            html.Button('Apply Filters', id='apply-filters-fc-button', n_clicks=0)
        ], className='dashboard-card'),

        # Table/Details and Graph side-by-side
        html.Div([
             html.Div([
                html.Div([dash_table.DataTable(id='failed-claims-table', columns=[], data=[])], className='dashboard-card table-container'),
                html.Div(id='selected-failed-claim-details-fc', className='dashboard-card details-container'),
             ], style={'flex': '2', 'marginRight': '24px'}),
            html.Div([
                html.Div([dcc.Graph(id='failure-reason-bargraph-fc')], className='dashboard-card graph-container') 
            ], style={'flex': '1'})
        ], style={'display': 'flex', 'flexDirection': 'row'})
    ])
