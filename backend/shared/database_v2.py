"""
Database configuration and utilities for PantryPal v2.0
Includes async engine setup, session management, and helper functions
"""

import os
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy import text, event
from sqlalchemy.pool import NullPool, QueuePool

from .models_v2 import Base


# ============================================================================
# DATABASE URL CONFIGURATION
# ============================================================================

def get_database_url() -> str:
    """
    Construct database URL from environment variables
    
    Supports:
    - Direct DATABASE_URL (priority)
    - Component-based (DB_HOST, DB_PORT, etc.)
    """
    # Option 1: Direct URL (for production)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        # Convert postgres:// to postgresql+asyncpg:// for async support
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif not database_url.startswith("postgresql+asyncpg://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return database_url
    
    # Option 2: Component-based (for development)
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "pantrypal")
    
    return f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# ============================================================================
# ENGINE CONFIGURATION
# ============================================================================

def create_database_engine(
    url: Optional[str] = None,
    pool_size: int = 20,
    max_overflow: int = 40,
    echo: bool = False,
    pool_pre_ping: bool = True,
    pool_recycle: int = 3600,
) -> AsyncEngine:
    """
    Create async database engine with optimized settings
    
    Args:
        url: Database URL (defaults to get_database_url())
        pool_size: Number of permanent connections (20 for production)
        max_overflow: Additional connections on demand (40 for production)
        echo: Log all SQL queries (False in production)
        pool_pre_ping: Test connections before use (True for reliability)
        pool_recycle: Recycle connections after N seconds (avoid stale connections)
    
    Returns:
        AsyncEngine configured for production use
    """
    if url is None:
        url = get_database_url()
    
    # Determine pool class based on environment
    if os.getenv("TESTING") == "true":
        # Use NullPool for testing (one connection per session)
        poolclass = NullPool
        pool_size = 0
        max_overflow = 0
    else:
        # Use QueuePool for production (connection pooling)
        poolclass = QueuePool
    
    engine = create_async_engine(
        url,
        poolclass=poolclass,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
        pool_recycle=pool_recycle,
        echo=echo,
        # Connection arguments for PostgreSQL
        connect_args={
            "server_settings": {
                "application_name": "pantrypal_api",
                "jit": "off",  # Disable JIT for predictable performance
            },
            "command_timeout": 60,  # 60 second query timeout
            "timeout": 10,  # 10 second connection timeout
        },
    )
    
    return engine


# ============================================================================
# GLOBAL ENGINE & SESSION FACTORY
# ============================================================================

# Create global engine (initialized once per app lifecycle)
engine: AsyncEngine = create_database_engine()

# Create session factory (use this to get new sessions)
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit (for better performance)
    autoflush=False,  # Manual control over flushing
    autocommit=False,  # Always use transactions
)


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency to get database session
    
    Usage:
        @app.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            result = await session.execute(select(Item))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database session (for non-FastAPI code)
    
    Usage:
        async with get_session_context() as session:
            result = await session.execute(select(User))
            users = result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ============================================================================
# ROW LEVEL SECURITY HELPERS
# ============================================================================

async def set_session_user_id(session: AsyncSession, user_id: str) -> None:
    """
    Set user_id for Row Level Security (RLS)
    
    Call this in FastAPI middleware or dependency to enable RLS policies
    
    Args:
        session: AsyncSession
        user_id: UUID of current user (as string)
    
    Usage:
        @app.middleware("http")
        async def set_rls_context(request: Request, call_next):
            user_id = request.state.user_id  # From JWT
            async with get_session_context() as session:
                await set_session_user_id(session, str(user_id))
                response = await call_next(request)
            return response
    """
    await session.execute(
        text("SET LOCAL app.user_id = :user_id"),
        {"user_id": user_id}
    )


async def get_session_user_id(session: AsyncSession) -> Optional[str]:
    """
    Get current user_id from session (for debugging RLS)
    
    Returns:
        UUID string or None
    """
    result = await session.execute(text("SELECT current_setting('app.user_id', true)"))
    return result.scalar()


# ============================================================================
# DATABASE LIFECYCLE
# ============================================================================

async def init_database():
    """
    Initialize database schema (for development/testing only)
    
    WARNING: Do not use in production! Use Alembic migrations instead.
    
    Usage:
        @app.on_event("startup")
        async def startup():
            if os.getenv("ENVIRONMENT") == "development":
                await init_database()
    """
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        
        # Install extensions
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pg_trgm"'))
        
        print("✅ Database schema initialized")


async def drop_database():
    """
    Drop all tables (for testing only)
    
    WARNING: This will DELETE ALL DATA!
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        print("⚠️ Database schema dropped")


async def close_database():
    """
    Close database connections (call on app shutdown)
    
    Usage:
        @app.on_event("shutdown")
        async def shutdown():
            await close_database()
    """
    await engine.dispose()
    print("✅ Database connections closed")


# ============================================================================
# HEALTH CHECK
# ============================================================================

async def check_database_health() -> dict:
    """
    Check database connectivity and return health metrics
    
    Returns:
        {
            "status": "healthy" | "unhealthy",
            "postgres_version": "15.3",
            "pool_size": 20,
            "pool_overflow": 5,
            "response_time_ms": 15.3
        }
    """
    import time
    
    try:
        start_time = time.time()
        
        async with AsyncSessionLocal() as session:
            # Test query
            result = await session.execute(text("SELECT version()"))
            version = result.scalar()
            
            # Pool stats
            pool = engine.pool
            pool_size = pool.size()
            pool_overflow = pool.overflow()
        
        response_time_ms = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "postgres_version": version.split()[1] if version else "unknown",
            "pool_size": pool_size,
            "pool_overflow": pool_overflow,
            "response_time_ms": round(response_time_ms, 2),
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


# ============================================================================
# QUERY HELPERS
# ============================================================================

async def execute_raw_sql(query: str, params: dict = None) -> list:
    """
    Execute raw SQL query and return results
    
    Usage:
        results = await execute_raw_sql(
            "SELECT * FROM users WHERE email = :email",
            {"email": "user@example.com"}
        )
    """
    async with get_session_context() as session:
        result = await session.execute(text(query), params or {})
        return result.fetchall()


async def get_table_count(table_name: str) -> int:
    """
    Get row count for a table
    
    Usage:
        count = await get_table_count("users")
    """
    async with get_session_context() as session:
        result = await session.execute(
            text(f"SELECT COUNT(*) FROM {table_name}")
        )
        return result.scalar()


async def vacuum_table(table_name: str) -> None:
    """
    Run VACUUM ANALYZE on a table (for maintenance)
    
    Usage:
        await vacuum_table("pantry_entries")
    """
    # Use raw connection (cannot run VACUUM in transaction)
    async with engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT").execute(
            text(f"VACUUM ANALYZE {table_name}")
        )


# ============================================================================
# EVENTS & LOGGING
# ============================================================================

@event.listens_for(AsyncEngine, "connect")
def set_postgresql_pragma(dbapi_conn, connection_record):
    """
    Set PostgreSQL connection settings on every new connection
    """
    cursor = dbapi_conn.cursor()
    
    # Set timezone to UTC
    cursor.execute("SET timezone TO 'UTC'")
    
    # Set search path
    cursor.execute("SET search_path TO public")
    
    # Disable JIT for predictable performance
    cursor.execute("SET jit TO off")
    
    cursor.close()


# ============================================================================
# TRANSACTION HELPERS
# ============================================================================

class TransactionContext:
    """
    Context manager for explicit transaction control
    
    Usage:
        async with TransactionContext() as session:
            user = User(email="test@example.com")
            session.add(user)
            # Auto-commits on exit, rolls back on exception
    """
    
    def __init__(self):
        self.session: Optional[AsyncSession] = None
    
    async def __aenter__(self) -> AsyncSession:
        self.session = AsyncSessionLocal()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.session.rollback()
        else:
            await self.session.commit()
        await self.session.close()


# ============================================================================
# TESTING UTILITIES
# ============================================================================

async def reset_database_for_testing():
    """
    Drop and recreate all tables (for integration tests)
    
    Usage:
        @pytest.fixture(autouse=True)
        async def reset_db():
            await reset_database_for_testing()
    """
    await drop_database()
    await init_database()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Engine & Session
    "engine",
    "AsyncSessionLocal",
    "get_session",
    "get_session_context",
    
    # RLS
    "set_session_user_id",
    "get_session_user_id",
    
    # Lifecycle
    "init_database",
    "drop_database",
    "close_database",
    
    # Health
    "check_database_health",
    
    # Helpers
    "execute_raw_sql",
    "get_table_count",
    "vacuum_table",
    "TransactionContext",
    
    # Testing
    "reset_database_for_testing",
]
