from dash import dcc, html

def create_healthcare_analytics_layout():
    """Creates the layout for the Healthcare Analytics Dashboard."""
    return html.Div([
        # Filters Card
        html.Div([
            html.H4("Healthcare Analytics Filters"),
            html.Div([
                dcc.Dropdown(id='facility-filter-ha', placeholder='Select Facility(s) (Optional)', multi=True),
            ], className='form-element-wrapper'),
            html.Div([
                dcc.DatePickerRange(
                    id='date-picker-range-ha',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                    display_format='YYYY-MM-DD'
                ),
            ], className='form-element-wrapper'),
            # Wrap Input and Button in a div for better layout control if needed
            html.Div([
                dcc.Input(
                    id='top-n-filter-ha',
                    type='number',
                    value=10,
                    placeholder='Top N items',
                    style={'marginRight': '10px'} # Keep style if specific spacing needed
                ),
                html.Button('Apply Filters', id='apply-filters-ha-button', n_clicks=0)
            ], className='form-element-wrapper', style={'display': 'flex', 'alignItems': 'center'}), # Example flex styling
        ], className='dashboard-card filter-section'),

        # Sub-tabs for different analytics views
        html.Div([ # Wrapper for sub-tabs to style them like the main tabs
            dcc.Tabs(
                id='healthcare-analytics-subtabs',
                value='subtab-cpt',
                className='custom-tabs-container', # Use same class as main tabs for styling
                children=[
                    dcc.Tab(label='CPT Code Analysis', value='subtab-cpt'),
                    dcc.Tab(label='Diagnosis Code Analysis', value='subtab-dx'),
                    dcc.Tab(label='Payer Analysis', value='subtab-payer'),
                    dcc.Tab(label='Patient Demographics', value='subtab-demographics'),
                    dcc.Tab(label='Provider Metrics', value='subtab-provider'),
                ]
            )
        ], className='tab-container-wrapper', style={'marginTop': '20px'}),

        html.Div(id='healthcare-analytics-content', style={'marginTop': '0px'})
    ])