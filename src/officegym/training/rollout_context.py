"""Task-local audit identity; never forwarded to the model generation API."""
from contextvars import ContextVar
ACTIVE_TRAJECTORY=ContextVar('officegym_active_trajectory',default=None)
