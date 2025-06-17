from dash import dcc, html

def create_main_layout():
    """
    Creates the main layout for the Dash application.
    Features a sticky header and a styled tab bar.
    """
    return html.Div([
        html.Div([
            html.H1("Claims Analytics Dashboard")
        ], className='app-header'), # Styled by CSS

        # Wrapper for the tabs to style the bar they sit in
        html.Div([
            dcc.Tabs(
                id='main-tabs', 
                value='tab-processed-claims', 
                className='custom-styled-tabs', # Targets dcc.Tabs and its children for styling
                children=[
                    dcc.Tab(label='Processed Claims', value='tab-processed-claims'),
                    dcc.Tab(label='Failed Claims', value='tab-failed-claims'),
                    dcc.Tab(label='Processing Metrics', value='tab-processing-metrics'),
                    dcc.Tab(label='Healthcare Analytics', value='tab-healthcare-analytics'),
                ]
            )
        ], className='tab-wrapper-bar'), # Styles the container of the dcc.Tabs

        # Main content area where tab-specific layouts are rendered
        html.Div(id='main-content-area') # Styled by CSS for padding etc.
    ])

if __name__ == '__main__':
    print("This is the main layout definition (layout.py). It's imported by run.py.")