"""
API endpoints package.
"""

from .logs import router as logs_router

__all__ = [
    'logs_router'
]