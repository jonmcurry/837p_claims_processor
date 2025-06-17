from dash import dcc, html, dash_table

def create_processed_claims_layout():
    """Creates the layout for the Processed Claims Dashboard using card styling."""
    return html.Div([
        html.Div([
            html.Div(html.H4("Processed Claims Filters"), className="dashboard-card-header"),
            html.Div([
                html.Div([dcc.Dropdown(id='facility-filter-pc', placeholder='Select Facility(s)...', multi=True)], className='form-element-wrapper'),
                html.Div([dcc.Dropdown(id='payer-filter-pc', placeholder='Select Payer(s)...', multi=True)], className='form-element-wrapper'),
                html.Div([dcc.DatePickerRange(id='date-picker-range-pc', start_date_placeholder_text='Start Date', end_date_placeholder_text='End Date', display_format='YYYY-MM-DD')], className='form-element-wrapper'),
                html.Button('Apply Filters', id='apply-filters-pc-button', n_clicks=0)
            ], className="dashboard-card-content")
        ], className='dashboard-card'),

        html.Div([
            html.Div(html.H4("Processed Claims Data"), className="dashboard-card-header"),
            html.Div([dash_table.DataTable(id='processed-claims-table', columns=[], data=[])], className="dashboard-card-content")
        ], className='dashboard-card table-container'),
        
        html.Div(id='selected-claim-details-pc') 
    ])