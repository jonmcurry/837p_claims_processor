from dash import dcc, html, dash_table

def create_failed_claims_layout():
    """Creates the layout for the Failed Claims Dashboard."""
    return html.Div([
        # Filters Card
        html.Div([
            html.H4("Failed Claims Filters"),
            html.Div([
                dcc.Dropdown(id='facility-filter-fc', placeholder='Select Facility(s)...', multi=True),
            ], className='form-element-wrapper'),
            html.Div([
                dcc.Dropdown(id='failure-category-filter-fc', placeholder='Select Failure Category(s)...', multi=True),
            ], className='form-element-wrapper'),
            html.Div([
                 dcc.DatePickerRange(
                    id='date-picker-range-fc',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                    display_format='YYYY-MM-DD'
                ),
            ], className='form-element-wrapper'),
            html.Button('Apply Filters', id='apply-filters-fc-button', n_clicks=0)
        ], className='dashboard-card filter-section'),

        # Data Table Card
        html.Div([
            dash_table.DataTable(
                id='failed-claims-table',
                columns=[],
                data=[],
                # COMMON_DATATABLE_PROPS are applied in callbacks.py
            )
        ], className='dashboard-card table-container'),
        
        # Selected Row Details Card
        html.Div(
            id='selected-failed-claim-details-fc', 
            className='dashboard-card details-container' # Apply card style
        ),
        
        html.Hr(), # Keep hr for visual separation if desired, or remove if cards are enough

        # Graph Card
        html.Div([
            # html.H4("Failure Analysis", className="card-header-title"), # Title is set in graph by callback
            dcc.Graph(id='failure-reason-bargraph-fc')
        ], className='dashboard-card graph-container')
    ])