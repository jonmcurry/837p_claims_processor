from dash import dcc, html

# Define navigation links data structure for reusability
nav_links_data = [
    {"href": "/", "text": "Processing Metrics", "id_suffix": "home"},
    {"href": "/analytics", "text": "Healthcare Analytics", "id_suffix": "analytics"},
    {"href": "/processed", "text": "Processed Claims", "id_suffix": "processed"},
    {"href": "/failed", "text": "Failed Claims", "id_suffix": "failed"},
]

def create_main_layout():
    """
    Creates the main application layout with a sticky header, a fixed sidebar,
    and a main content display area.
    """
    
    header = html.Div(
        [html.H1("Claims Analytics Dashboard")],
        className='app-header'
    )

    sidebar_nav_links = [
        dcc.Link(
            link_data["text"], 
            href=link_data["href"], 
            className="nav-link", 
            id=f"nav-link-{link_data['id_suffix']}"
        ) for link_data in nav_links_data
    ]
    
    sidebar = html.Div(
        [
            html.H2("Navigation", className="sidebar-title"),
            html.Hr(),
            html.Nav(sidebar_nav_links, className="nav-group"),
        ],
        className="sidebar",
    )

    main_content_area = html.Div(
        [
            html.Div(
                [
                    html.H1(id="content-title"),
                    html.Div(className="user-controls-placeholder")
                ], 
                className="content-header"
            ),
            html.Div(id="page-content", className="page-content-wrapper"),
        ],
        className="main-content",
    )

    return html.Div([
        dcc.Location(id="url", refresh=False),
        header,
        sidebar,
        main_content_area
    ])

if __name__ == '__main__':
    print("This is the main layout definition (claims_ui/app/layout.py).")