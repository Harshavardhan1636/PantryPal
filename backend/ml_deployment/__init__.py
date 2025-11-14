"""
ML Deployment Package

Model registry, containerization, and deployment services.

Components:
- model_registry: MLflow-based model lifecycle management
- model_deployment: BentoML containerization and Kubernetes deployment
"""

from .model_registry import (
    ModelRegistry,
    ModelMetadata,
    ModelStage,
    ModelType,
    ModelValidationResult,
)

from .model_deployment import (
    WastePredictor,
    ModelDeploymentManager,
)

__all__ = [
    # Registry
    "ModelRegistry",
    "ModelMetadata",
    "ModelStage",
    "ModelType",
    "ModelValidationResult",
    
    # Deployment
    "WastePredictor",
    "ModelDeploymentManager",
]

__version__ = "1.0.0"
