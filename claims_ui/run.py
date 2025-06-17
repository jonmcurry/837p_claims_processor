import sys
import os

# Add the parent directory (project root) to sys.path to allow imports like 'from claims_ui.app import app'
# This is useful if you run 'python run.py' from within the 'claims_ui' directory directly.
# However, the recommended way to run is 'python -m claims_ui.run' from the project root directory (e.g., 837p_claims_processor).
# If using 'python -m ...', this explicit path manipulation might not be strictly necessary
# but doesn't hurt.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from claims_ui.app import app  # Import the app instance from app/__init__.py
from claims_ui.app.layout import create_main_layout
import claims_ui.app.callbacks  # Ensure callbacks are registered

app.layout = create_main_layout()

if __name__ == '__main__':
    # Set host to '0.0.0.0' to make it accessible on your network
    # For development, debug=True is helpful for live reloading and error messages
    app.run(debug=True, host='0.0.0.0', port=8050)