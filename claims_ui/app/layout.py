from dash import dcc, html

def create_main_layout():
    """
    Creates the main layout for the Dash application.
    This version includes a styled header and tab navigation.
    """
    return html.Div([
        html.Div([
            html.H1("Claims Analytics Dashboard")
        ], className='app-header'),

        # Wrapper for the tabs themselves to control their background and padding
        html.Div([
            dcc.Tabs(
                id='main-tabs', 
                value='tab-processed-claims', 
                className='custom-tabs-container', # For specific styling of the tab bar
                children=[
                    dcc.Tab(label='Processed Claims', value='tab-processed-claims'),
                    dcc.Tab(label='Failed Claims', value='tab-failed-claims'),
                    dcc.Tab(label='Processing Metrics', value='tab-processing-metrics'),
                    dcc.Tab(label='Healthcare Analytics', value='tab-healthcare-analytics'),
                ]
            )
        ], className='tab-container-wrapper'), # This class styles the bar holding the tabs

        # Main content area where tab-specific layouts will be rendered
        # This area will have its own padding defined by #main-content-area in CSS
        html.Div(id='main-content-area') 
    ])

if __name__ == '__main__':
    print("This is the main layout definition (layout.py). It's imported by run.py.")