from dash import dcc
from dash import html
from dash import dash_table

def create_failed_claims_layout():
    """
    Creates the layout for the Failed Claims tab.
    Includes filters, a data table, and a bar graph for failure analysis.
    Dropdown options will be populated by a callback.
    """
    return html.Div(id='failed-claims-layout-container', children=[
        html.Div(className='filter-section', children=[
            html.H4("Failed Claims Filters"),
            dcc.Dropdown(
                id='facility-filter-fc',
                placeholder='Select Facility(s)...',
                multi=True,
                options=[],
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            dcc.Dropdown(
                id='failure-category-filter-fc',
                placeholder='Select Failure Category(s)...',
                multi=True,
                options=[],
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            dcc.DatePickerRange(
                id='date-picker-range-fc',
                start_date_placeholder_text='Start Date',
                end_date_placeholder_text='End Date',
                className='date-picker-full-width',
                style={'marginBottom': '10px'}
            ),
            html.Button('Apply Filters', id='apply-filters-fc-button', n_clicks=0, className='btn btn-primary')
        ]),

        html.Div(className='table-container', children=[
            dash_table.DataTable(
                id='failed-claims-table',
                columns=[],
                data=[],
                page_size=15,
                sort_action='native',
                filter_action='native',
                page_action='native',
                export_format='csv',
                fill_width=False,
                row_selectable='single',
                style_cell={'textAlign': 'left', 'padding': '5px', 'fontFamily': 'Arial, sans-serif'},
                style_header={'backgroundColor': '#ecf0f1', 'fontWeight': 'bold', 'borderBottom': '1px solid black'},
                style_data={'whiteSpace': 'normal', 'height': 'auto', 'borderBottom': '1px solid #eee'},
                style_table={'overflowX': 'auto'}
            )
        ]),

        html.Div(id='selected-failed-claim-details-fc', children=[
            # Content populated by callback, styled via ID in style.css
        ]),

        html.Div(className='graph-container', children=[
            html.H4("Failure Analysis"), # Moved H4 into graph container for better context
            dcc.Graph(id='failure-reason-bargraph-fc')
        ])
    ])

print("Failed claims layout definition updated with CSS classes and DataTable styles.")
