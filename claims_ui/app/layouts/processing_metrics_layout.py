import dash_core_components as dcc
import dash_html_components as html

def create_processing_metrics_layout():
    """
    Creates the layout for the Processing Metrics tab.
    Includes filters, KPI summaries, and graphs.
    Dropdown options will be populated by a callback.
    """
    return html.Div(id='processing-metrics-layout-container', children=[
        html.Div(className='filter-section', children=[
            html.H4("Processing Metrics Filters"),
            dcc.Dropdown(
                id='org-filter-pm',
                placeholder='Select Organization (All)',
                multi=False,
                options=[],
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            dcc.Dropdown(
                id='region-filter-pm',
                placeholder='Select Region (All)',
                multi=False,
                options=[],
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            dcc.Dropdown(
                id='facility-filter-pm',
                placeholder='Select Facility(s) (All)',
                multi=True,
                options=[],
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            dcc.DatePickerRange(
                id='date-picker-range-pm',
                start_date_placeholder_text='Start Date',
                end_date_placeholder_text='End Date',
                className='date-picker-full-width',
                style={'marginBottom': '10px'}
            ),
            html.Button('Apply Filters', id='apply-filters-pm-button', n_clicks=0, className='btn btn-primary')
        ]),

        # KPI container already has 'kpi-container' class from style.css for flexbox layout
        html.Div(id='kpi-summary-pm', className='kpi-container filter-section'), # Also adding filter-section for border/bg

        # Group graphs into rows for better layout control if needed, or apply graph-container individually
        html.Div(className='graph-row', children=[
            html.Div(className='graph-container graph-item-half-width', children=[
                dcc.Graph(id='daily-trends-graph-pm')
            ]),
            html.Div(className='graph-container graph-item-half-width', children=[
                dcc.Graph(id='facility-comparison-graph-pm')
            ])
        ]),

        html.Div(className='graph-container graph-item-full-width', children=[ # Full width for this one
             dcc.Graph(id='org-region-summary-graph-pm')
        ])
    ])

print("Processing metrics layout definition updated with CSS classes.")
