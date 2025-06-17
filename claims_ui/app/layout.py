from dash import dcc, html

def create_main_layout():
    """
    Creates the main layout with a sidebar and content area.
    Navigation is handled by dcc.Link and dcc.Location.
    """
    # This is the sidebar component
    sidebar = html.Div(
        [
            html.H2("Claims Processor", className="sidebar-title"),
            html.Hr(),
            html.Nav(
                [
                    dcc.Link("Processing Metrics", href="/", className="nav-link"),
                    dcc.Link("Healthcare Analytics", href="/analytics", className="nav-link"),
                    dcc.Link("Processed Claims", href="/processed", className="nav-link"),
                    dcc.Link("Failed Claims", href="/failed", className="nav-link"),
                ],
                className="nav-group"
            ),
        ],
        className="sidebar",
    )

    # This is the main content area
    content = html.Div(
        [
            # Header bar within the content area
            html.Div([
                html.H1(id="content-title"), # Title will be updated by callback
                # Placeholder for user icons/controls on the right
                html.Div(className="user-controls-placeholder") 
            ], className="content-header"),
            
            # The actual page content will be rendered here by a callback
            html.Div(id="page-content", className="page-content-wrapper"),
        ],
        className="main-content",
    )

    # The root layout component
    return html.Div([
        dcc.Location(id="url", refresh=False), # Essential for URL-based navigation
        sidebar,
        content
    ])

if __name__ == '__main__':
    print("This is the main layout definition (layout.py). It's imported by run.py.")

