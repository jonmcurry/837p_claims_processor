# Defines the layout of the Dash application
from dash import dcc
from dash import html
# from . import app # app is typically imported in callbacks, not directly needed for layout definition usually

def create_main_layout():
    """
    Creates the main layout structure for the Dash application.
    Includes a title and a tabbed interface for different sections.
    """
    return html.Div(id='app-root', children=[ # Changed main-container to app-root for clarity
        html.Div(className='app-header', children=[
            html.H1("Claims Analytics Dashboard") # Removed inline style, will be handled by CSS
        ]),
        
        html.Div(className='tab-container', children=[
            dcc.Tabs(id='main-tabs', value='tab-processed-claims', children=[
                dcc.Tab(label="Processed Claims", value="tab-processed-claims"),
                dcc.Tab(label="Failed Claims", value="tab-failed-claims"),
                dcc.Tab(label="Processing Metrics", value="tab-processing-metrics"),
                dcc.Tab(label="Healthcare Analytics", value="tab-healthcare-analytics"),
            ]),
            
            # This Div will be updated by callbacks based on the selected tab
            html.Div(id='main-content-area') # Removed inline style, padding handled by CSS or content specific styles
        ])
    ])

# If you were to assign it directly here, it might look like:
# layout = create_main_layout()
# However, it's common to set app.layout in run.py or the main app file.
print("Main layout definition created.")
