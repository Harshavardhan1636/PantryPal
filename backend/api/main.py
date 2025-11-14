"""
PantryPal FastAPI Application - Main Entry Point
Production-ready REST API with comprehensive endpoints for food waste reduction platform.

Features:
- JWT authentication with refresh tokens
- Multi-tenant household management
- Pantry inventory tracking with barcode scanning
- OCR receipt processing
- Consumption & waste logging
- ML-powered predictions and recommendations
- Recipe search and management
- Notifications system
- Admin model management
- Comprehensive metrics and analytics
- Auto-generated OpenAPI documentation

Author: Senior SDE-3
Date: 2025-11-12
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

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
from backend.shared.database_v2 import init_database, close_database, check_database_health
from backend.api.middleware.rate_limit import RateLimitMiddleware
from backend.api.middleware.request_id import RequestIDMiddleware
from backend.api.middleware.logging import LoggingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pantrypal_api.log")
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting PantryPal API...")
    try:
        await init_database()
        logger.info("✅ Database initialized successfully")
        
        # Check database health
        health = await check_database_health()
        logger.info(f"📊 Database health: {health}")
        
        logger.info("🎉 PantryPal API is ready!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down PantryPal API...")
    try:
        await close_database()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")


# Create FastAPI application
app = FastAPI(
    title="PantryPal API",
    description=(
        "Production-ready REST API for PantryPal food waste reduction platform.\n\n"
        "Features:\n"
        "- 🔐 JWT authentication with refresh tokens\n"
        "- 🏠 Multi-tenant household management\n"
        "- 📦 Pantry inventory tracking with barcode scanning\n"
        "- 📄 OCR receipt processing\n"
        "- 📊 Consumption & waste analytics\n"
        "- 🤖 ML-powered predictions and recommendations\n"
        "- 🍳 Recipe search and management\n"
        "- 🔔 Multi-channel notifications\n"
        "- ⚙️ Admin model deployment\n"
        "- 📈 Comprehensive metrics\n\n"
        "Built with FastAPI, PostgreSQL, SQLAlchemy, and modern ML frameworks."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "PantryPal Support",
        "email": "support@pantrypal.com",
        "url": "https://pantrypal.com/support",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# ============================================================================
# Middleware Configuration
# ============================================================================

# CORS - Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev
        "http://localhost:19006",  # React Native Expo
        "https://pantrypal.com",  # Production frontend
        "https://app.pantrypal.com",  # Production app subdomain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Total-Count"],
)

# GZip compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request ID tracking
app.add_middleware(RequestIDMiddleware)

# Logging middleware
app.add_middleware(LoggingMiddleware)

# Rate limiting (10000 requests per hour per IP)
app.add_middleware(RateLimitMiddleware, max_requests=10000, window_seconds=3600)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": errors,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """Handle database errors."""
    logger.error(f"Database error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "database_error",
            "message": "A database error occurred. Please try again later.",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred. Please contact support.",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ============================================================================
# Router Registration
# ============================================================================

# Authentication endpoints
app.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

# Household management
app.include_router(
    households.router,
    prefix="/households",
    tags=["Households"],
)

# Pantry inventory
app.include_router(
    pantry.router,
    prefix="/pantry",
    tags=["Pantry"],
)

# Purchase & receipt processing
app.include_router(
    purchases.router,
    prefix="/purchases",
    tags=["Purchases"],
)

# Consumption tracking
app.include_router(
    consumption.router,
    prefix="/consume",
    tags=["Consumption"],
)

# Waste logging
app.include_router(
    waste.router,
    prefix="/waste",
    tags=["Waste"],
)

# ML predictions & recommendations
app.include_router(
    predictions.router,
    prefix="/predictions",
    tags=["Predictions & Recommendations"],
)

# Recipe management
app.include_router(
    recipes.router,
    prefix="/recipes",
    tags=["Recipes"],
)

# Notifications
app.include_router(
    notifications.router,
    prefix="/notifications",
    tags=["Notifications"],
)

# Admin & model management
app.include_router(
    admin.router,
    prefix="/admin",
    tags=["Admin"],
)

# Metrics & analytics
app.include_router(
    metrics.router,
    prefix="/household",
    tags=["Metrics"],
)


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get(
    "/",
    summary="Root endpoint",
    description="Welcome message and API information",
    response_description="API welcome message",
)
async def root() -> dict:
    """Root endpoint - API information."""
    return {
        "service": "PantryPal API",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get(
    "/health",
    summary="Health check",
    description="Check API and database health status",
    response_description="Health status information",
    tags=["Health"],
)
async def health_check() -> dict:
    """Health check endpoint."""
    try:
        db_health = await check_database_health()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": db_health,
            "api": {
                "version": "2.0.0",
                "uptime": "operational",
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


@app.get(
    "/ready",
    summary="Readiness check",
    description="Check if API is ready to serve traffic",
    response_description="Readiness status",
    tags=["Health"],
)
async def readiness_check() -> dict:
    """Readiness check for Kubernetes."""
    try:
        db_health = await check_database_health()
        
        if db_health["active_connections"] >= 0:
            return {
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        workers=4,
    )
