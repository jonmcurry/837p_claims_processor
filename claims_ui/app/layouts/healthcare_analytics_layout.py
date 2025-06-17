from dash import dcc, html

def create_healthcare_analytics_layout():
    """Creates the layout for the Healthcare Analytics Dashboard using card styling."""
    return html.Div([
        # This interval triggers the initial data load for this page.
        dcc.Interval(id='interval-ha', interval=500, max_intervals=1),

        # Filters Card
        html.Div([
            html.Div(html.H4("Healthcare Analytics Filters"), className="dashboard-card-header"),
            # ... (rest of the filter elements)
            html.Div([dcc.Dropdown(id='facility-filter-ha', placeholder='Select Facility(s) (Optional)', multi=True)], className='form-element-wrapper'),
            html.Div([dcc.DatePickerRange(id='date-picker-range-ha', start_date_placeholder_text='Start Date', end_date_placeholder_text='End Date')], className='form-element-wrapper'),
            html.Div([dcc.Input(id='top-n-filter-ha', type='number', value=10, placeholder='Top N items')], className='form-element-wrapper'),
            html.Button('Apply Filters', id='apply-filters-ha-button', n_clicks=0)
        ], className='dashboard-card'),

        # Card to contain the sub-tabs and their content
        html.Div([
            dcc.Tabs(id='healthcare-analytics-subtabs', value='subtab-cpt', className='custom-styled-tabs', children=[
                dcc.Tab(label='CPT Code Analysis', value='subtab-cpt'),
                dcc.Tab(label='Diagnosis Code Analysis', value='subtab-dx'),
            ]),
            html.Div(id='healthcare-analytics-content', style={'padding': '24px 0 0 0'})
        ], className='dashboard-card')
    ])
