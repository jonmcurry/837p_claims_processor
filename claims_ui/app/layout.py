from dash import dcc, html

# This module defines the main layout of the application.
# It includes the header, tabs for different dashboards, and a content area
# where the content of the selected tab will be rendered by callbacks.

def create_main_layout():
    """
    Creates the main layout for the Dash application.

    Returns:
        html.Div: The main layout component.
    """
    return html.Div([
        html.Div([
            html.H1("Claims Analytics Dashboard", style={'textAlign': 'center'})
        ], className='app-header'),

        html.Div([
            dcc.Tabs(id='main-tabs', value='tab-processed-claims', children=[
                dcc.Tab(label='Processed Claims', value='tab-processed-claims'),
                dcc.Tab(label='Failed Claims', value='tab-failed-claims'),
                dcc.Tab(label='Processing Metrics', value='tab-processing-metrics'),
                dcc.Tab(label='Healthcare Analytics', value='tab-healthcare-analytics'),
            ]),
            # This Div will be populated by callbacks based on the selected tab
            html.Div(id='main-content-area')
        ], className='tab-container')
    ])

if __name__ == '__main__':
    # This part is for testing the layout if you run this file directly.
    # In a real Dash app, this layout is usually assigned to app.layout in the main app file.
    # To test, you would need a dummy Dash app instance.
    # from dash import Dash
    # app = Dash(__name__)
    # app.layout = create_main_layout()
    # app.run_server(debug=True)
    print("This is the main layout definition. It's typically imported and used in run.py or app.py.")