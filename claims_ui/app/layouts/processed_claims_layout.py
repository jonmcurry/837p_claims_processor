from dash import dcc
from dash import html
from dash import dash_table

def create_processed_claims_layout():
    """
    Creates the layout for the Processed Claims tab.
    Includes filters for facility, payer, date range, and a data table.
    Dropdown options will be populated by a callback.
    """
    return html.Div(id='processed-claims-layout-container', children=[
        html.Div(className='filter-section', children=[
            html.H4("Processed Claims Filters"),
            dcc.Dropdown(
                id='facility-filter-pc',
                placeholder='Select Facility(s)...',
                multi=True,
                options=[],
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            dcc.Dropdown(
                id='payer-filter-pc',
                placeholder='Select Payer(s)...',
                multi=True,
                options=[],
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            dcc.DatePickerRange(
                id='date-picker-range-pc',
                start_date_placeholder_text='Start Date',
                end_date_placeholder_text='End Date',
                className='date-picker-full-width', # Custom class for width if needed by CSS
                style={'marginBottom': '10px'}
            ),
            html.Button('Apply Filters', id='apply-filters-pc-button', n_clicks=0, className='btn btn-primary')
        ]),

        html.Div(className='table-container', children=[
            dash_table.DataTable(
                id='processed-claims-table',
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
                style_table={'overflowX': 'auto'} # Ensure table itself is scrollable horizontally
            )
        ]),

        html.Div(id='selected-claim-details-pc', children=[ # This one can keep its specific ID for now, or get a class
            # Content populated by callback, already has some styling via ID in style.css
        ])
    ])

print("Processed claims layout definition updated with CSS classes and DataTable styles.")
import dash_core_components as dcc
import dash_html_components as html
from dash import dash_table

def create_processed_claims_layout():
    """
    Creates the layout for the Processed Claims tab.
    Includes filters for facility, payer, date range, and a data table.
    Dropdown options will be populated by a callback.
    """
    return html.Div(id='processed-claims-layout-container', children=[
        html.Div(className='filter-section', children=[
            html.H4("Processed Claims Filters"),
            dcc.Dropdown(
                id='facility-filter-pc',
                placeholder='Select Facility(s)...',
                multi=True,
                options=[],
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            dcc.Dropdown(
                id='payer-filter-pc',
                placeholder='Select Payer(s)...',
                multi=True,
                options=[],
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            dcc.DatePickerRange(
                id='date-picker-range-pc',
                start_date_placeholder_text='Start Date',
                end_date_placeholder_text='End Date',
                className='date-picker-full-width', # Custom class for width if needed by CSS
                style={'marginBottom': '10px'}
            ),
            html.Button('Apply Filters', id='apply-filters-pc-button', n_clicks=0, className='btn btn-primary')
        ]),

        html.Div(className='table-container', children=[
            dash_table.DataTable(
                id='processed-claims-table',
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
                style_table={'overflowX': 'auto'} # Ensure table itself is scrollable horizontally
            )
        ]),

        html.Div(id='selected-claim-details-pc', children=[ # This one can keep its specific ID for now, or get a class
            # Content populated by callback, already has some styling via ID in style.css
        ])
    ])

print("Processed claims layout definition updated with CSS classes and DataTable styles.")

