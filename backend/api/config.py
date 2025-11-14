"""
Application Configuration
Settings management using Pydantic for environment variables.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    # Application
    APP_NAME: str = "PantryPal API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"  # development, staging, production
    
    # Database
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_ECHO: bool = False
    
    # JWT Authentication
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    
    # Security
    BCRYPT_ROUNDS: int = 12
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGITS: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_MAX_REQUESTS: int = 10000
    RATE_LIMIT_WINDOW_SECONDS: int = 3600
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:19006",
        "https://pantrypal.com",
        "https://app.pantrypal.com",
    ]
    
    # Redis (for caching and rate limiting)
    REDIS_URL: Optional[str] = None
    REDIS_TTL_SECONDS: int = 3600
    
    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: list[str] = ["image/jpeg", "image/png", "image/webp"]
    UPLOAD_DIR: str = "uploads"
    
    # Google Cloud Vision API (for OCR)
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    
    # OpenAI (for recipe recommendations)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4"
    
    # Open Food Facts API
    OPEN_FOOD_FACTS_API_URL: str = "https://world.openfoodfacts.org/api/v0"
    OPEN_FOOD_FACTS_USER_AGENT: str = "PantryPal/2.0"
    
    # Twilio (for SMS notifications)
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None
    
    # SendGrid (for email notifications)
    SENDGRID_API_KEY: Optional[str] = None
    SENDGRID_FROM_EMAIL: str = "noreply@pantrypal.com"
    SENDGRID_FROM_NAME: str = "PantryPal"
    
    # Firebase Cloud Messaging (for push notifications)
    FCM_SERVER_KEY: Optional[str] = None
    FCM_SENDER_ID: Optional[str] = None
    
    # ML Model Endpoints
    ML_PREDICTION_SERVICE_URL: Optional[str] = "http://prediction-service:5000"
    ML_RECOMMENDATION_SERVICE_URL: Optional[str] = "http://recommendation-service:5000"
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    
    # Feature Flags
    FEATURE_OCR_ENABLED: bool = True
    FEATURE_ML_PREDICTIONS_ENABLED: bool = True
    FEATURE_RECIPE_RECOMMENDATIONS_ENABLED: bool = True
    FEATURE_NOTIFICATIONS_ENABLED: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()
