"""
FastAPI Dependencies
Common dependencies for authentication, authorization, and database access.
"""

from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.shared.database_v2 import get_session, set_session_user_id
from backend.shared.models_v2 import User, HouseholdUser, UserRole
from backend.api.config import get_settings

settings = get_settings()
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_session),
) -> User:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer token
        session: Database session
        
    Returns:
        User: Authenticated user object
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        token = credentials.credentials
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
        user_uuid = UUID(user_id)
        
    except (JWTError, ValueError):
        raise credentials_exception
    
    # Fetch user from database
    result = await session.execute(
        select(User).where(
            User.id == user_uuid,
            User.deleted_at.is_(None),
            User.is_active == True,
        )
    )
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
    # Set RLS context for this session
    await set_session_user_id(session, str(user_uuid))
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency to ensure user is active.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Active user object
        
    Raises:
        HTTPException: If user is not active or email not verified
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    if not current_user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please verify your email to continue.",
        )
    
    return current_user


async def get_optional_user(
    authorization: Optional[str] = Header(None),
    session: AsyncSession = Depends(get_session),
) -> Optional[User]:
    """
    Dependency to optionally get current user (for public endpoints).
    
    Args:
        authorization: Optional Authorization header
        session: Database session
        
    Returns:
        Optional[User]: User if authenticated, None otherwise
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    try:
        token = authorization.split(" ")[1]
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        
        user_id = payload.get("sub")
        if user_id is None:
            return None
        
        user_uuid = UUID(user_id)
        
        result = await session.execute(
            select(User).where(
                User.id == user_uuid,
                User.deleted_at.is_(None),
                User.is_active,
            )
        )
        user = result.scalar_one_or_none()
        
        if user:
            await set_session_user_id(session, str(user_uuid))
        
        return user
        
    except Exception:
        return None


async def require_household_access(
    household_id: UUID,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
    min_role: UserRole = UserRole.MEMBER,
) -> HouseholdUser:
    """
    Dependency to ensure user has access to a household with minimum role.
    
    Args:
        household_id: Household UUID
        current_user: Current authenticated user
        session: Database session
        min_role: Minimum required role (default: MEMBER)
        
    Returns:
        HouseholdUser: Household membership record
        
    Raises:
        HTTPException: If user doesn't have access or insufficient permissions
    """
    result = await session.execute(
        select(HouseholdUser).where(
            HouseholdUser.household_id == household_id,
            HouseholdUser.user_id == current_user.id,
            HouseholdUser.deleted_at.is_(None),
        )
    )
    membership = result.scalar_one_or_none()
    
    if not membership:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Household not found or access denied",
        )
    
    # Check role permissions
    role_hierarchy = {
        UserRole.VIEWER: 1,
        UserRole.MEMBER: 2,
        UserRole.ADMIN: 3,
        UserRole.OWNER: 4,
    }
    
    if role_hierarchy.get(membership.role, 0) < role_hierarchy.get(min_role, 0):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Requires {min_role.value} role or higher.",
        )
    
    return membership


async def require_admin_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Dependency to ensure user is an admin.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User: Admin user object
        
    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    
    return current_user


def pagination_params(
    skip: int = 0,
    limit: int = 100,
) -> dict:
    """
    Dependency for pagination parameters.
    
    Args:
        skip: Number of records to skip (default: 0)
        limit: Maximum number of records to return (default: 100, max: 1000)
        
    Returns:
        dict: Pagination parameters
    """
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip parameter must be non-negative",
        )
    
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit parameter must be between 1 and 1000",
        )
    
    return {"skip": skip, "limit": limit}
