from dash import dcc, html

def create_healthcare_analytics_layout():
    return html.Div([
        html.Div([
            html.Div(html.H4("Healthcare Analytics Filters"), className="dashboard-card-header"),
            html.Div([
                html.Div([dcc.Dropdown(id='facility-filter-ha', placeholder='Select Facility(s) (Optional)', multi=True)], className='form-element-wrapper'),
                html.Div([dcc.DatePickerRange(id='date-picker-range-ha', display_format='YYYY-MM-DD')], className='form-element-wrapper'),
                html.Div([ 
                    dcc.Input(id='top-n-filter-ha', type='number', value=10, placeholder='Top N items', style={'marginRight': '10px'}),
                    html.Button('Apply Filters', id='apply-filters-ha-button', n_clicks=0)
                ], className='form-element-wrapper', style={'display': 'flex', 'alignItems': 'center'}),
            ], className="dashboard-card-content")
        ], className='dashboard-card'),
        
        html.Div([
            dcc.Tabs(
                id='healthcare-analytics-subtabs', value='subtab-cpt', className='custom-styled-tabs',
                children=[
                    dcc.Tab(label='CPT Code Analysis', value='subtab-cpt'),
                    dcc.Tab(label='Diagnosis Code Analysis', value='subtab-dx'),
                    dcc.Tab(label='Payer Analysis', value='subtab-payer'),
                    dcc.Tab(label='Patient Demographics', value='subtab-demographics'),
                    dcc.Tab(label='Provider Metrics', value='subtab-provider'),
                ]
            )
        ], className='tab-wrapper-bar', style={'marginTop': '24px'}), 

        html.Div(id='healthcare-analytics-content', style={'paddingTop': '24px'}) 
    ])