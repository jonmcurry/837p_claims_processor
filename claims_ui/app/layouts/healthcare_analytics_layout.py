from dash import dcc, html

def create_healthcare_analytics_layout():
    """Creates the layout for the Healthcare Analytics Dashboard."""
    return html.Div([
        html.Div([
            html.H4("Healthcare Analytics Filters"),
            html.Div([
                dcc.Dropdown(id='facility-filter-ha', placeholder='Select Facility(s) (Optional)', multi=True),
            ], style={'marginBottom': '10px'}),
            html.Div([
                dcc.DatePickerRange(
                    id='date-picker-range-ha',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                    display_format='YYYY-MM-DD'
                ),
            ], style={'marginBottom': '10px'}),
            html.Div([
                dcc.Input(id='top-n-filter-ha', type='number', value=10, placeholder='Top N items', 
                          style={'marginRight': '10px', 'padding': '5px', 'borderRadius': '3px', 'border': '1px solid #ccc'}),
                html.Button('Apply Filters', id='apply-filters-ha-button', n_clicks=0)
            ], style={'marginBottom': '15px'}),
        ], className='filter-section'),
        
        html.Hr(),
        dcc.Tabs(id='healthcare-analytics-subtabs', value='subtab-cpt', children=[
            dcc.Tab(label='CPT Code Analysis', value='subtab-cpt'),
            dcc.Tab(label='Diagnosis Code Analysis', value='subtab-dx'),
            dcc.Tab(label='Payer Analysis', value='subtab-payer'),
            dcc.Tab(label='Patient Demographics', value='subtab-demographics'),
            dcc.Tab(label='Provider Metrics', value='subtab-provider'),
        ]),
        html.Div(id='healthcare-analytics-content', style={'marginTop': '20px'}) # Content rendered by callback
    ])