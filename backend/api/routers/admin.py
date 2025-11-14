"""
Admin Router
Handles ML model deployment and system administration.
"""

import logging
from uuid import UUID
from typing import List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import require_admin_user, pagination_params
from backend.api.schemas import ModelDeployRequest, ModelResponse, MessageResponse
from backend.shared.database_v2 import get_session
from backend.shared.models_v2 import User

logger = logging.getLogger(__name__)

router = APIRouter()


# Placeholder model storage (in production, use proper model registry)
_model_registry: dict[UUID, dict] = {}


@router.post(
    "/models/deploy",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Deploy ML model",
    description="Deploy a new ML model version (admin only)",
)
async def deploy_model(
    request: ModelDeployRequest,
    admin_user: User = Depends(require_admin_user),
    session: AsyncSession = Depends(get_session),
) -> ModelResponse:
    """Deploy a new ML model."""
    from uuid import uuid4
    
    model_id = uuid4()
    
    model = {
        "id": model_id,
        "model_type": request.model_type,
        "version": request.version,
        "artifact_uri": request.model_artifact_uri,
        "metadata": request.metadata,
        "deployed_at": datetime.utcnow(),
        "deployed_by": admin_user.id,
        "is_active": True,
    }
    
    _model_registry[model_id] = model
    
    logger.info(
        f"Model deployed: {model_id} - {request.model_type} v{request.version} "
        f"by user {admin_user.email}"
    )
    
    return ModelResponse(
        id=model_id,
        model_type=request.model_type,
        version=request.version,
        artifact_uri=request.model_artifact_uri,
        metadata=request.metadata,
        deployed_at=model["deployed_at"],
        is_active=True,
    )


@router.get(
    "/models",
    response_model=List[ModelResponse],
    summary="List ML models",
    description="List all deployed ML models (admin only)",
)
async def list_models(
    model_type: str | None = None,
    admin_user: User = Depends(require_admin_user),
    pagination: dict = Depends(pagination_params),
) -> List[ModelResponse]:
    """List deployed ML models."""
    models = list(_model_registry.values())
    
    if model_type:
        models = [m for m in models if m["model_type"] == model_type]
    
    # Sort by deployment date descending
    models.sort(key=lambda m: m["deployed_at"], reverse=True)
    
    # Apply pagination
    start = pagination["skip"]
    end = start + pagination["limit"]
    models = models[start:end]
    
    return [
        ModelResponse(
            id=m["id"],
            model_type=m["model_type"],
            version=m["version"],
            artifact_uri=m["artifact_uri"],
            metadata=m["metadata"],
            deployed_at=m["deployed_at"],
            is_active=m["is_active"],
        )
        for m in models
    ]


@router.patch(
    "/models/{model_id}/activate",
    response_model=MessageResponse,
    summary="Activate model",
    description="Activate a deployed model (admin only)",
)
async def activate_model(
    model_id: UUID,
    admin_user: User = Depends(require_admin_user),
) -> MessageResponse:
    """Activate a model."""
    if model_id not in _model_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    
    # Deactivate other models of same type
    model_type = _model_registry[model_id]["model_type"]
    for mid, model in _model_registry.items():
        if model["model_type"] == model_type:
            model["is_active"] = (mid == model_id)
    
    logger.info(f"Model activated: {model_id} by user {admin_user.email}")
    
    return MessageResponse(
        message=f"Model {model_id} activated successfully",
        success=True,
    )


@router.delete(
    "/models/{model_id}",
    response_model=MessageResponse,
    summary="Delete model",
    description="Delete a deployed model (admin only)",
)
async def delete_model(
    model_id: UUID,
    admin_user: User = Depends(require_admin_user),
) -> MessageResponse:
    """Delete a model."""
    if model_id not in _model_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    
    del _model_registry[model_id]
    
    logger.info(f"Model deleted: {model_id} by user {admin_user.email}")
    
    return MessageResponse(
        message=f"Model {model_id} deleted successfully",
        success=True,
    )
