from dash import dcc, html, dash_table

def create_failed_claims_layout():
    """Creates the layout for the Failed Claims Dashboard."""
    return html.Div([
        html.Div([
            html.H4("Failed Claims Filters"),
            html.Div([
                dcc.Dropdown(id='facility-filter-fc', placeholder='Select Facility(s)...', multi=True),
            ], style={'marginBottom': '10px'}),
            html.Div([
                dcc.Dropdown(id='failure-category-filter-fc', placeholder='Select Failure Category(s)...', multi=True),
            ], style={'marginBottom': '10px'}),
            html.Div([
                 dcc.DatePickerRange(
                    id='date-picker-range-fc',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                    display_format='YYYY-MM-DD'
                ),
            ], style={'marginBottom': '15px'}),
            html.Button('Apply Filters', id='apply-filters-fc-button', n_clicks=0)
        ], className='filter-section'),

        html.Div([
            dash_table.DataTable(
                id='failed-claims-table',
                columns=[],
                data=[],
                page_size=15,
                sort_action='native',
                filter_action='native',
                row_selectable='single',
                style_cell={'textAlign': 'left', 'padding': '5px', 'fontFamily': 'Arial, sans-serif', 'fontSize': '0.9em'},
                style_header={'backgroundColor': '#ecf0f1', 'fontWeight': 'bold', 'borderBottom': '1px solid black'},
                style_data={'whiteSpace': 'normal', 'height': 'auto', 'borderBottom': '1px solid #eee'},
                export_format='csv',
                fill_width=False,
            )
        ], className='table-container'),
        
        html.Div(id='selected-failed-claim-details-fc', className='details-container', style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #eee'}),
        
        html.Hr(),
        html.Div([
            #html.H4("Failure Analysis", style={'textAlign': 'center', 'marginBottom': '10px'}), # Title is in graph now
            dcc.Graph(id='failure-reason-bargraph-fc')
        ], className='graph-container')
    ])