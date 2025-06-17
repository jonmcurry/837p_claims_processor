# This will be the entry point to run the Dash app
from claims_ui.app import app  # Import the app instance from app/__init__.py
from claims_ui.app.layout import create_main_layout
# Import callbacks to ensure they are registered with the app
import claims_ui.app.callbacks # noqa: F401 (unused import is intentional here)


# Set the layout of the application
# The layout is defined in layout.py and returned by create_main_layout()
app.layout = create_main_layout()

if __name__ == '__main__':
    # Run the Dash app
    # debug=True enables hot-reloading and error messages in the browser
    # host='0.0.0.0' makes the app accessible on your local network
    # port=8050 is the default port for Dash apps
    print("Starting Dash server on http://0.0.0.0:8050/")
    app.run_server(debug=True, host='0.0.0.0', port=8050)
