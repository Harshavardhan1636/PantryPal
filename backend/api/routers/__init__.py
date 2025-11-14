"""Router package initialization."""

from backend.api.routers import (
    auth,
    households,
    pantry,
    purchases,
    consumption,
    waste,
    predictions,
    recipes,
    notifications,
    admin,
    metrics,
)

__all__ = [
    "auth",
    "households",
    "pantry",
    "purchases",
    "consumption",
    "waste",
    "predictions",
    "recipes",
    "notifications",
    "admin",
    "metrics",
]
