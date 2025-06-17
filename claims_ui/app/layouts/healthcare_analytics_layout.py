from dash import dcc, html

def create_healthcare_analytics_layout():
    """Creates the layout for the Healthcare Analytics Dashboard using card styling."""
    return html.Div([
        # Filters Card
        html.Div([
            html.Div(html.H4("Healthcare Analytics Filters"), className="dashboard-card-header"),
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
            html.Div([
                dcc.Input(
                    id='top-n-filter-ha',
                    type='number',
                    value=10,
                    placeholder='Top N items',
                    style={'marginRight': '10px', 'width': '100%'}
                ),
            ], className='form-element-wrapper', style={'flex': 1, 'marginRight': '16px'}),
             html.Div([
                html.Button('Apply Filters', id='apply-filters-ha-button', n_clicks=0, style={'width': '100%'})
             ], className='form-element-wrapper', style={'flex': 1}),
        ], className='dashboard-card', style={'display': 'flex', 'alignItems': 'flex-end'}),

        # Card to contain the sub-tabs and their content
        html.Div([
            dcc.Tabs(
                id='healthcare-analytics-subtabs',
                value='subtab-cpt',
                className='custom-styled-tabs',
                children=[
                    dcc.Tab(label='CPT Code Analysis', value='subtab-cpt'),
                    dcc.Tab(label='Diagnosis Code Analysis', value='subtab-dx'),
                    dcc.Tab(label='Payer Analysis', value='subtab-payer'),
                    dcc.Tab(label='Patient Demographics', value='subtab-demographics'),
                    dcc.Tab(label='Provider Metrics', value='subtab-provider'),
                ],
                style={'padding': '0 24px', 'margin': '0 -24px'} # Adjust padding to align with card
            ),
            # Content for the subtabs will be rendered here, inside the card
            html.Div(
                id='healthcare-analytics-content',
                style={'padding': '24px 0 0 0'}
            )
        ], className='dashboard-card')
    ])
