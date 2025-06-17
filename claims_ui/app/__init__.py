# Initializes the Dash app
import dash

# Create the Dash app instance
# suppress_callback_exceptions=True is helpful for multi-page/tabbed apps
# where callbacks might be defined in other files before their layout is added to the main app layout.
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Server attribute is needed for some deployment environments (e.g., Gunicorn)
server = app.server

print("Dash app initialized.")
