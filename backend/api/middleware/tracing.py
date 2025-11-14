"""
OpenTelemetry Tracing Configuration for PantryPal

Instruments:
- FastAPI endpoints
- Database queries
- ML predictions
- Background jobs
- External API calls

Propagates trace IDs across async operations and worker queues.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from fastapi import FastAPI, Request
from typing import Dict, Any, Callable
import logging
import os
import json
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)

# Global tracer
tracer: trace.Tracer = None

# ============================================================================
# Tracer Setup
# ============================================================================

def setup_tracing(app: FastAPI, service_name: str = "pantrypal-backend") -> None:
    """
    Configure OpenTelemetry tracing for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        service_name: Name of the service for tracing
    """
    global tracer
    
    # Create resource with service information
    resource = Resource.create({
        SERVICE_NAME: service_name,
        "service.version": os.getenv("APP_VERSION", "1.0.0"),
        "deployment.environment": os.getenv("ENVIRONMENT", "production"),
    })
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Configure Cloud Trace exporter (GCP)
    if os.getenv("GOOGLE_CLOUD_PROJECT"):
        cloud_trace_exporter = CloudTraceSpanExporter(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        tracer_provider.add_span_processor(
            BatchSpanProcessor(cloud_trace_exporter)
        )
    
    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    # Get tracer
    tracer = trace.get_tracer(__name__)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    # Instrument SQLAlchemy
    SQLAlchemyInstrumentor().instrument()
    
    # Instrument Redis
    RedisInstrumentor().instrument()
    
    # Instrument HTTP requests
    RequestsInstrumentor().instrument()
    
    logger.info(f"✅ OpenTelemetry tracing configured for {service_name}")


# ============================================================================
# Trace Context Propagation
# ============================================================================

class TraceContextMiddleware:
    """
    Middleware to extract and inject trace context.
    """
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.propagator = TraceContextTextMapPropagator()
    
    async def __call__(self, request: Request, call_next: Callable):
        # Extract trace context from headers
        context = self.propagator.extract(carrier=dict(request.headers))
        
        # Attach trace context to request state
        request.state.trace_context = context
        
        # Process request with trace context
        with trace.use_span(trace.get_current_span(), end_on_exit=False):
            response = await call_next(request)
        
        # Inject trace context into response headers
        self.propagator.inject(response.headers)
        
        return response


def get_trace_id_from_request(request: Request) -> str:
    """
    Extract trace ID from request for logging correlation.
    
    Args:
        request: FastAPI request object
    
    Returns:
        Trace ID as hex string
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, '032x')
    return "no-trace-id"


def inject_trace_context_to_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Inject current trace context into headers for propagation.
    
    Args:
        headers: Existing headers dictionary
    
    Returns:
        Updated headers with trace context
    """
    propagator = TraceContextTextMapPropagator()
    propagator.inject(headers)
    return headers


def inject_trace_context_to_pubsub(message_attributes: Dict[str, str]) -> Dict[str, str]:
    """
    Inject trace context into Pub/Sub message attributes.
    
    Args:
        message_attributes: Pub/Sub message attributes
    
    Returns:
        Updated attributes with trace context
    """
    return inject_trace_context_to_headers(message_attributes)


def extract_trace_context_from_pubsub(message_attributes: Dict[str, str]) -> Any:
    """
    Extract trace context from Pub/Sub message attributes.
    
    Args:
        message_attributes: Pub/Sub message attributes
    
    Returns:
        Trace context
    """
    propagator = TraceContextTextMapPropagator()
    return propagator.extract(carrier=message_attributes)


# ============================================================================
# Tracing Decorators
# ============================================================================

def trace_function(operation_name: str = None, attributes: Dict[str, Any] = None):
    """
    Decorator to trace a function execution.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        attributes: Additional attributes to add to the span
    
    Example:
        @trace_function(operation_name="predict_waste", attributes={"model": "lightgbm"})
        def predict_waste(pantry_item_id: int):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            
            with tracer.start_as_current_span(op_name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                # Add function parameters as attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            
            with tracer.start_as_current_span(op_name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@contextmanager
def trace_span(operation_name: str, attributes: Dict[str, Any] = None):
    """
    Context manager to create a trace span.
    
    Args:
        operation_name: Name of the operation
        attributes: Additional attributes to add to the span
    
    Example:
        with trace_span("database_query", {"query_type": "SELECT"}):
            result = db.execute(query)
    """
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


# ============================================================================
# Database Query Tracing
# ============================================================================

def trace_db_query(query_type: str, table: str = None):
    """
    Decorator to trace database queries.
    
    Args:
        query_type: Type of query (SELECT, INSERT, UPDATE, DELETE)
        table: Table name (optional)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(f"db.{query_type.lower()}") as span:
                span.set_attribute("db.system", "postgresql")
                span.set_attribute("db.operation", query_type)
                
                if table:
                    span.set_attribute("db.table", table)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


# ============================================================================
# ML Prediction Tracing
# ============================================================================

def trace_ml_prediction(model_name: str, prediction_type: str):
    """
    Decorator to trace ML predictions.
    
    Args:
        model_name: Name of the ML model
        prediction_type: Type of prediction (on_demand, batch)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span("ml.predict") as span:
                span.set_attribute("ml.model.name", model_name)
                span.set_attribute("ml.prediction.type", prediction_type)
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Add result metadata
                    if isinstance(result, dict):
                        if "confidence" in result:
                            span.set_attribute("ml.prediction.confidence", result["confidence"])
                        if "risk_score" in result:
                            span.set_attribute("ml.prediction.risk_score", result["risk_score"])
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span("ml.predict") as span:
                span.set_attribute("ml.model.name", model_name)
                span.set_attribute("ml.prediction.type", prediction_type)
                
                try:
                    result = func(*args, **kwargs)
                    
                    if isinstance(result, dict):
                        if "confidence" in result:
                            span.set_attribute("ml.prediction.confidence", result["confidence"])
                        if "risk_score" in result:
                            span.set_attribute("ml.prediction.risk_score", result["risk_score"])
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# ============================================================================
# Background Job Tracing
# ============================================================================

def trace_background_job(job_name: str, job_id: str = None):
    """
    Decorator to trace background jobs.
    
    Args:
        job_name: Name of the background job
        job_id: Unique job ID (optional)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(f"job.{job_name}") as span:
                span.set_attribute("job.name", job_name)
                
                if job_id:
                    span.set_attribute("job.id", job_id)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


# ============================================================================
# External API Call Tracing
# ============================================================================

def trace_external_api_call(service_name: str, endpoint: str):
    """
    Decorator to trace external API calls.
    
    Args:
        service_name: Name of the external service
        endpoint: API endpoint being called
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(f"http.{service_name}") as span:
                span.set_attribute("http.service", service_name)
                span.set_attribute("http.endpoint", endpoint)
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Extract status code if response object
                    if hasattr(result, "status_code"):
                        span.set_attribute("http.status_code", result.status_code)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


# ============================================================================
# Utility Functions
# ============================================================================

def add_span_event(event_name: str, attributes: Dict[str, Any] = None) -> None:
    """
    Add an event to the current span.
    
    Args:
        event_name: Name of the event
        attributes: Event attributes
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        span.add_event(event_name, attributes=attributes or {})


def set_span_attribute(key: str, value: Any) -> None:
    """
    Set an attribute on the current span.
    
    Args:
        key: Attribute key
        value: Attribute value
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute(key, value)


def record_exception_in_span(exception: Exception) -> None:
    """
    Record an exception in the current span.
    
    Args:
        exception: Exception to record
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        span.record_exception(exception)
        span.set_status(Status(StatusCode.ERROR, str(exception)))
