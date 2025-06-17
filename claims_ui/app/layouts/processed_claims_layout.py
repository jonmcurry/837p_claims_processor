from dash import dcc, html, dash_table

def create_processed_claims_layout():
    """Creates the layout for the Processed Claims Dashboard."""
    return html.Div([
        html.Div([
            html.H4("Processed Claims Filters"),
            html.Div([
                dcc.Dropdown(
                    id='facility-filter-pc', 
                    placeholder='Select Facility(s)...', 
                    multi=True,
                    className='filter-dropdown' # For potential specific styling
                ),
            ], style={'marginBottom': '10px'}),
            html.Div([
                dcc.Dropdown(
                    id='payer-filter-pc', 
                    placeholder='Select Payer(s)...', 
                    multi=True,
                    className='filter-dropdown'
                ),
            ], style={'marginBottom': '10px'}),
            html.Div([
                dcc.DatePickerRange(
                    id='date-picker-range-pc',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                    display_format='YYYY-MM-DD',
                    className='date-picker'
                ),
            ], style={'marginBottom': '15px'}),
            html.Button('Apply Filters', id='apply-filters-pc-button', n_clicks=0)
        ], className='filter-section'),

        html.Div([
            dash_table.DataTable(
                id='processed-claims-table',
                columns=[],  # Populated by callback
                data=[],     # Populated by callback
                page_size=15,
                sort_action='native',
                filter_action='native',
                row_selectable='single', # or 'multi'
                style_cell={'textAlign': 'left', 'padding': '5px', 'fontFamily': 'Arial, sans-serif', 'fontSize': '0.9em'},
                style_header={'backgroundColor': '#ecf0f1', 'fontWeight': 'bold', 'borderBottom': '1px solid black'},
                style_data={'whiteSpace': 'normal', 'height': 'auto', 'borderBottom': '1px solid #eee'},
                export_format='csv',
                fill_width=False,
            )
        ], className='table-container'),
        
        html.Div(id='selected-claim-details-pc', className='details-container', style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #eee'}) # For displaying selected row details
    ])