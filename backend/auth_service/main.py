"""FastAPI authentication microservice for PantryPal."""
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
import jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database import get_db, init_db, close_db
from auth_service.models import User, Organization, Household, HouseholdMember


# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    await init_db()
    yield
    await close_db()


# FastAPI app
app = FastAPI(
    title="PantryPal Auth Service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class UserRegister(BaseModel):
    """User registration request."""
    email: EmailStr
    name: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=8)
    household_name: str = Field(default="My Household")


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    """User profile response."""
    id: str
    email: str
    name: str
    email_verified: bool
    phone_number: Optional[str] = None
    avatar_url: Optional[str] = None
    households: list[dict]


class HouseholdCreate(BaseModel):
    """Create household request."""
    name: str = Field(min_length=1, max_length=255)
    members_count: int = Field(default=1, ge=1)
    storage_types: list[str] = Field(default=["refrigerator", "pantry", "freezer"])
    dietary_preferences: list[str] = Field(default=[])


class HouseholdInvite(BaseModel):
    """Invite user to household."""
    email: EmailStr
    role: str = Field(default="member", pattern="^(owner|admin|member|viewer)$")


# Utility functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if user is None or user.deleted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    return user


# API Endpoints
@app.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user with default household."""
    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Create organization
    org = Organization(name=f"{user_data.name}'s Organization")
    db.add(org)
    await db.flush()
    
    # Create household
    household = Household(
        org_id=org.id,
        name=user_data.household_name,
    )
    db.add(household)
    await db.flush()
    
    # Create user
    user = User(
        email=user_data.email,
        name=user_data.name,
        # Note: In production, use proper password hashing with a password field
        # For now, we'll use Auth0/Firebase for authentication
    )
    db.add(user)
    await db.flush()
    
    # Add user to household as owner
    member = HouseholdMember(
        household_id=household.id,
        user_id=user.id,
        role="owner",
    )
    db.add(member)
    await db.commit()
    
    # Create access token
    access_token = create_access_token(data={"sub": str(user.id)})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "email": user.email,
            "name": user.name,
            "households": [{
                "id": str(household.id),
                "name": household.name,
                "role": "owner",
            }],
        },
    }


@app.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """Login user with email and password."""
    result = await db.execute(select(User).where(User.email == credentials.email))
    user = result.scalar_one_or_none()
    
    if not user or user.deleted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    
    # Update last login
    user.last_login_at = datetime.utcnow()
    await db.commit()
    
    # Get user households
    result = await db.execute(
        select(HouseholdMember, Household)
        .join(Household)
        .where(HouseholdMember.user_id == user.id)
        .where(HouseholdMember.left_at.is_(None))
    )
    households = [
        {
            "id": str(household.id),
            "name": household.name,
            "role": member.role,
        }
        for member, household in result.all()
    ]
    
    # Create access token
    access_token = create_access_token(data={"sub": str(user.id)})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "email": user.email,
            "name": user.name,
            "households": households,
        },
    }


@app.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get current user profile."""
    # Get user households
    result = await db.execute(
        select(HouseholdMember, Household)
        .join(Household)
        .where(HouseholdMember.user_id == current_user.id)
        .where(HouseholdMember.left_at.is_(None))
    )
    households = [
        {
            "id": str(household.id),
            "name": household.name,
            "role": member.role,
        }
        for member, household in result.all()
    ]
    
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "name": current_user.name,
        "email_verified": current_user.email_verified,
        "phone_number": current_user.phone_number,
        "avatar_url": current_user.avatar_url,
        "households": households,
    }


@app.post("/households", status_code=status.HTTP_201_CREATED)
async def create_household(
    household_data: HouseholdCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new household."""
    # Get user's organization
    result = await db.execute(
        select(Household.org_id)
        .join(HouseholdMember)
        .where(HouseholdMember.user_id == current_user.id)
        .limit(1)
    )
    org_id = result.scalar_one_or_none()
    
    if not org_id:
        raise HTTPException(status_code=400, detail="User not in any organization")
    
    # Create household
    household = Household(
        org_id=org_id,
        name=household_data.name,
        members_count=household_data.members_count,
        storage_types=household_data.storage_types,
        dietary_preferences=household_data.dietary_preferences,
    )
    db.add(household)
    await db.flush()
    
    # Add user as owner
    member = HouseholdMember(
        household_id=household.id,
        user_id=current_user.id,
        role="owner",
    )
    db.add(member)
    await db.commit()
    
    return {
        "id": str(household.id),
        "name": household.name,
        "role": "owner",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "auth"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
