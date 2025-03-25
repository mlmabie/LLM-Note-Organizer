"""
API module for Note Organizer.
"""

# Import API components to make them available
from note_organizer.api.server import app, start

# Export components
__all__ = ["app", "start"]
