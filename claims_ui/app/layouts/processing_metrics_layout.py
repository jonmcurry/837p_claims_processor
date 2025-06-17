from dash import dcc, html

def create_processing_metrics_layout():
    """Creates the layout for the Processing Metrics Dashboard."""
    return html.Div([
        # Filters Card
        html.Div([
            html.H4("Processing Metrics Filters"),
            html.Div([
                dcc.Dropdown(id='org-filter-pm', placeholder='Select Organization', multi=False),
            ], className='form-element-wrapper'),
            html.Div([
                dcc.Dropdown(id='region-filter-pm', placeholder='Select Region (after Org)', multi=False),
            ], className='form-element-wrapper'),
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
        ], className='dashboard-card filter-section'),

        # KPIs - This div is styled by .kpi-summary-pm in CSS, items within it get .kpi-item (also .dashboard-card if needed)
        html.Div(id='kpi-summary-pm', className='kpi-summary-pm'), 
        
        # Graphs in individual cards
        html.Div([
            # Title is set in graph by callback
            dcc.Graph(id='daily-trends-graph-pm')
        ], className='dashboard-card graph-container'),
        
        html.Div([
            # Title is set in graph by callback
            dcc.Graph(id='facility-comparison-graph-pm')
        ], className='dashboard-card graph-container'),
        
        html.Div([
            # Title is set in graph by callback
            dcc.Graph(id='org-region-summary-graph-pm')
        ], className='dashboard-card graph-container')
    ])