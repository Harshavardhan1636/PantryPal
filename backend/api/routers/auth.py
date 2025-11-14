"""
Authentication Router
Handles user registration, login, token refresh, and logout.
"""

import logging
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from jose import jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.config import get_settings
from backend.api.schemas import (
    RegisterRequest,
    LoginRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserResponse,
    MessageResponse,
)
from backend.api.dependencies import get_current_active_user
from backend.shared.database_v2 import get_session
from backend.shared.models_v2 import User, UserSession

settings = get_settings()
logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: str) -> tuple[str, datetime]:
    """
    Create JWT access token.
    
    Returns:
        tuple: (token, expiration_time)
    """
    expires_delta = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.utcnow() + expires_delta
    
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }
    
    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    
    return token, expire


def create_refresh_token(user_id: str) -> tuple[str, datetime]:
    """
    Create JWT refresh token.
    
    Returns:
        tuple: (token, expiration_time)
    """
    expires_delta = timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    expire = datetime.utcnow() + expires_delta
    
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
        "jti": str(uuid4()),  # Token ID for revocation
    }
    
    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    
    return token, expire


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Create a new user account with email and password",
)
async def register(
    request: RegisterRequest,
    session: AsyncSession = Depends(get_session),
) -> User:
    """Register a new user."""
    # Check if email already exists
    result = await session.execute(
        select(User).where(User.email == request.email)
    )
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )
    
    # Create new user
    user = User(
        email=request.email,
        password_hash=hash_password(request.password),
        name=request.name,
        email_verified=False,  # Require email verification
        is_active=True,
    )
    
    session.add(user)
    await session.commit()
    await session.refresh(user)
    
    logger.info(f"New user registered: {user.email}")
    
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User login",
    description="Authenticate user and return JWT tokens",
)
async def login(
    request: LoginRequest,
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    """Authenticate user and return JWT tokens."""
    # Find user by email
    result = await session.execute(
        select(User).where(
            User.email == request.email,
            User.deleted_at.is_(None),
        )
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    # Create tokens
    access_token, access_expire = create_access_token(str(user.id))
    refresh_token, refresh_expire = create_refresh_token(str(user.id))
    
    # Store refresh token in database
    user_session = UserSession(
        user_id=user.id,
        refresh_token=refresh_token,
        expires_at=refresh_expire,
        user_agent=None,  # Can be extracted from request headers
        ip_address=None,  # Can be extracted from request
    )
    
    session.add(user_session)
    
    # Update last login
    user.last_login_at = datetime.utcnow()
    
    await session.commit()
    
    logger.info(f"User logged in: {user.email}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="Get a new access token using refresh token",
)
async def refresh_token(
    request: RefreshTokenRequest,
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    """Refresh access token using refresh token."""
    try:
        # Decode refresh token
        payload = jwt.decode(
            request.refresh_token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )
        
        user_id = payload.get("sub")
        token_id = payload.get("jti")
        
        if not user_id or not token_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    
    # Verify refresh token exists in database
    result = await session.execute(
        select(UserSession).where(
            UserSession.refresh_token == request.refresh_token,
            UserSession.revoked_at.is_(None),
        )
    )
    user_session = result.scalar_one_or_none()
    
    if not user_session or user_session.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    
    # Create new tokens
    access_token, access_expire = create_access_token(user_id)
    new_refresh_token, refresh_expire = create_refresh_token(user_id)
    
    # Revoke old refresh token and create new one
    user_session.revoked_at = datetime.utcnow()
    
    new_session = UserSession(
        user_id=user_session.user_id,
        refresh_token=new_refresh_token,
        expires_at=refresh_expire,
        user_agent=user_session.user_agent,
        ip_address=user_session.ip_address,
    )
    
    session.add(new_session)
    await session.commit()
    
    logger.info(f"Token refreshed for user: {user_id}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="User logout",
    description="Revoke refresh token and logout user",
)
async def logout(
    request: RefreshTokenRequest,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Logout user by revoking refresh token."""
    # Revoke refresh token
    result = await session.execute(
        select(UserSession).where(
            UserSession.refresh_token == request.refresh_token,
            UserSession.user_id == current_user.id,
            UserSession.revoked_at.is_(None),
        )
    )
    user_session = result.scalar_one_or_none()
    
    if user_session:
        user_session.revoked_at = datetime.utcnow()
        await session.commit()
    
    logger.info(f"User logged out: {current_user.email}")
    
    return MessageResponse(
        message="Logged out successfully",
        success=True,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get authenticated user profile",
)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Get current authenticated user profile."""
    return current_user
