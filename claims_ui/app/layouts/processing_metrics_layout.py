from dash import dcc, html

def create_processing_metrics_layout():
    """Creates the layout for the Processing Metrics Dashboard."""
    return html.Div([
        html.Div([
            html.H4("Processing Metrics Filters"),
            html.Div([
                dcc.Dropdown(id='org-filter-pm', placeholder='Select Organization', multi=False),
            ], style={'marginBottom': '10px'}),
            html.Div([
                dcc.Dropdown(id='region-filter-pm', placeholder='Select Region (after Org)', multi=False),
            ], style={'marginBottom': '10px'}),
            html.Div([
                dcc.Dropdown(id='facility-filter-pm', placeholder='Select Facility(s) (Optional)', multi=True),
            ], style={'marginBottom': '10px'}),
            html.Div([
                dcc.DatePickerRange(
                    id='date-picker-range-pm',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                    display_format='YYYY-MM-DD'
                ),
            ], style={'marginBottom': '15px'}),
            html.Button('Apply Filters', id='apply-filters-pm-button', n_clicks=0)
        ], className='filter-section'),

        html.Div(id='kpi-summary-pm', className='kpi-summary-pm'), # Styled via CSS
        html.Hr(),
        html.Div([
            #html.H5("Daily Processing Trends"), # Title is in graph now
            dcc.Graph(id='daily-trends-graph-pm')
        ], className='graph-container'),
        
        html.Div([
            #html.H5("Facility Comparison"), # Title is in graph now
            dcc.Graph(id='facility-comparison-graph-pm')
        ], className='graph-container'),
        
        html.Div([
            #html.H5("Organization/Region Summary"), # Title is in graph now
            dcc.Graph(id='org-region-summary-graph-pm')
        ], className='graph-container')
    ])