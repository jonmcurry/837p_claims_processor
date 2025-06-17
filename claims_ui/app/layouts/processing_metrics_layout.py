from dash import dcc, html

def create_processing_metrics_layout():
    """Creates the layout for the Processing Metrics Dashboard using card styling."""
    return html.Div([
        # Filters Card
        html.Div([
            html.Div(html.H4("Processing Metrics Filters"), className="dashboard-card-header"),
            html.Div([ # Content of the filter card
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
            ], className="dashboard-card-content")
        ], className='dashboard-card'),

        html.Div(id='kpi-summary-pm', className='kpi-grid-container'),

        html.Div([
            dcc.Graph(id='daily-trends-graph-pm')
        ], className='dashboard-card graph-container'),

        html.Div([
            dcc.Graph(id='facility-comparison-graph-pm')
        ], className='dashboard-card graph-container'),

        html.Div([
            dcc.Graph(id='org-region-summary-graph-pm')
        ], className='dashboard-card graph-container')
    ])

if __name__ == '__main__':
    print("This is the layout definition for the Processing Metrics page (processing_metrics_layout.py).")