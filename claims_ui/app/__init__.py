import dash

# Initialize the Dash application
# suppress_callback_exceptions=True is useful for multi-page apps or apps where
# callbacks are defined in separate files and layout components are generated dynamically.
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Expose the Flask server instance for potential WSGI deployment
server = app.server

# You can configure Dash further here if needed, e.g., app.title
app.title = "Claims Analytics Dashboard"