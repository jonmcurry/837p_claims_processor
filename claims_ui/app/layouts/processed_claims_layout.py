from dash import dcc, html, dash_table

def create_processed_claims_layout():
    """Creates the layout for the Processed Claims Dashboard using card styling."""
    return html.Div([
        # Filters Card
        html.Div([
            html.H4("Processed Claims Filters"), # Styled by .dashboard-card > h4
            html.Div([
                dcc.Dropdown(
                    id='facility-filter-pc', 
                    placeholder='Select Facility(s)...', 
                    multi=True,
                ),
            ], className='form-element-wrapper'), # Consistent spacing for form elements
            html.Div([
                dcc.Dropdown(
                    id='payer-filter-pc', 
                    placeholder='Select Payer(s)...', 
                    multi=True,
                ),
            ], className='form-element-wrapper'),
            html.Div([
                dcc.DatePickerRange(
                    id='date-picker-range-pc',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                    display_format='YYYY-MM-DD',
                ),
            ], className='form-element-wrapper'),
            html.Button('Apply Filters', id='apply-filters-pc-button', n_clicks=0)
        ], className='dashboard-card'), # Applied .dashboard-card

        # Data Table Card
        html.Div([
            # Optional: Use a card header for the table title
            # html.Div(html.H4("Claims Data"), className="dashboard-card-header"),
            # html.Div([ # Content wrapper if header is used
            dash_table.DataTable(
                id='processed-claims-table',
                columns=[],
                data=[],
                # COMMON_DATATABLE_PROPS are applied in callbacks.py
            )
            # ], className="dashboard-card-content") # if header is used
        ], className='dashboard-card table-container'), # .table-container for specific table card needs
        
        # Selected Row Details Card
        html.Div(
            id='selected-claim-details-pc', 
             className='dashboard-card details-container' 
        ) 
    ])