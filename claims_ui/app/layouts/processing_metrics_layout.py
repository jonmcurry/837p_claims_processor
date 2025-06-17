from dash import dcc, html

def create_processing_metrics_layout():
    """Creates the layout for the Processing Metrics Dashboard using card styling."""
    return html.Div([
        # This interval triggers the initial data load for this page.
        dcc.Interval(id='interval-pm', interval=500, max_intervals=1),

        # Filters Card
        html.Div([
            html.Div(html.H4("Processing Metrics Filters"), className="dashboard-card-header"),
            html.Div([
                dcc.Dropdown(id='facility-filter-pm', placeholder='Select Facility(s) (Optional)', multi=True),
            ], className='form-element-wrapper'),
            html.Div([
                dcc.DatePickerRange(
                    id='date-picker-range-pm',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                    display_format='YYYY-MM-DD'
                ),
            ], className='form-element-wrapper'),
            html.Button('Apply Filters', id='apply-filters-pm-button', n_clicks=0)
        ], className='dashboard-card'),

        # KPIs
        html.Div(id='kpi-summary-pm', className='kpi-grid-container'), 
        
        # Graphs in a two-column grid layout
        html.Div([
            html.Div([
                html.Div([dcc.Graph(id='daily-trends-graph-pm')], className='dashboard-card graph-container'),
                html.Div([dcc.Graph(id='payer-distribution-graph-pm')], className='dashboard-card graph-container'),
            ], style={'flex': 1, 'marginRight': '12px'}),
            html.Div([
                 html.Div([dcc.Graph(id='facility-comparison-graph-pm')], className='dashboard-card graph-container'),
                html.Div([dcc.Graph(id='claim-type-distribution-graph-pm')], className='dashboard-card graph-container'),
            ], style={'flex': 1, 'marginLeft': '12px'}),
        ], style={'display': 'flex', 'flexDirection': 'row'})
    ])
