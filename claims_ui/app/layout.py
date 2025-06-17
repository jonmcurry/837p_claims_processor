from dash import dcc, html

def create_main_layout():
    """
    Creates the main layout for the Dash application.
    Features a sticky header and a styled tab bar.
    """
    return html.Div([
        # Main application header
        html.Div([
            html.H1("Claims Analytics Dashboard")
        ], className='app-header'),

        # Wrapper for the tabs to style the bar they sit in
        html.Div([
            dcc.Tabs(
                id='main-tabs', 
                value='tab-processing-metrics', # Default to the most visual tab
                className='custom-styled-tabs',
                children=[
                    dcc.Tab(label='Processing Metrics', value='tab-processing-metrics'),
                    dcc.Tab(label='Healthcare Analytics', value='tab-healthcare-analytics'),
                    dcc.Tab(label='Processed Claims', value='tab-processed-claims'),
                    dcc.Tab(label='Failed Claims', value='tab-failed-claims'),
                ]
            )
        ], className='tab-wrapper-bar'),

        # Main content area where tab-specific layouts are rendered
        # This wrapper adds padding and context for the cards within it.
        html.Div(id='main-content-area', className='dashboard-main-content')
    ])

if __name__ == '__main__':
    print("This is the main layout definition (layout.py). It's imported by run.py.")

