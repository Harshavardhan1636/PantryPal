# PantryPal FastAPI Backend

Production-ready REST API for PantryPal food waste reduction platform.

## 🚀 Features

- **Authentication**: JWT-based auth with refresh tokens
- **Multi-tenant**: Household-based data isolation with Row Level Security
- **Barcode Scanning**: Automatic item cataloging via barcode lookup
- **OCR Receipt Processing**: Extract items from receipt photos
- **ML Predictions**: AI-powered waste risk predictions
- **Recipe Recommendations**: Smart recipe suggestions for at-risk items
- **Notifications**: Multi-channel alerts (push, SMS, email)
- **Analytics**: Comprehensive household metrics and waste tracking
- **Auto-generated Docs**: OpenAPI/Swagger documentation

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL 15+ with extensions:
  - `uuid-ossp`
  - `pgcrypto`
  - `pg_trgm`
  - `pgvector`
- Redis (for caching and rate limiting)
- Google Cloud Project (for OCR)

## 🛠️ Installation

### 1. Clone and Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

**Critical settings to configure:**
- `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET_KEY`: Generate a secure key (min 32 characters)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to GCP service account key
- API keys for Twilio, SendGrid, OpenAI (optional but recommended)

### 3. Initialize Database

```bash
# Run migrations
alembic upgrade head

# Or apply schema directly
psql -U postgres -d pantrypal -f ../infrastructure/database/schema_v2_production.sql
```

### 4. Run Development Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 📚 API Documentation

### Authentication Endpoints

#### Register New User
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "name": "John Doe"
}
```

**Response:**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "name": "John Doe",
  "email_verified": false,
  "is_active": true,
  "is_admin": false,
  "created_at": "2025-11-12T10:00:00Z"
}
```

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Refresh Token
```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### Household Management

#### Create Household
```http
POST /households
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "name": "My Family",
  "timezone": "America/New_York",
  "members_count": 4,
  "currency": "USD"
}
```

#### List Households
```http
GET /households?skip=0&limit=100
Authorization: Bearer <access_token>
```

### Pantry Management

#### Add Pantry Item
```http
POST /pantry
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "household_id": "uuid",
  "barcode": "8901234567890",
  "name": "Green capsicum",
  "quantity": 2,
  "unit": "pcs",
  "purchase_date": "2025-11-01T00:00:00Z",
  "expiry_date": "2025-11-15T00:00:00Z",
  "storage_location": "fridge",
  "purchase_price": 3.99
}
```

**Response:** Returns pantry entry with `canonical_item` if barcode resolved.

#### List Pantry Items
```http
GET /pantry/{household_id}?storage_location=fridge&include_depleted=false
Authorization: Bearer <access_token>
```

### Purchase & Receipt Processing

#### Upload Receipt
```http
POST /purchases
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

household_id=uuid
receipt_image=@receipt.jpg
```

**Response:** Returns purchase record with OCR results and auto-created pantry entries.

### Consumption & Waste

#### Log Consumption
```http
POST /consume
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "pantry_entry_id": "uuid",
  "quantity": 1.5,
  "unit": "pcs",
  "recipe_id": "uuid",
  "notes": "Used for dinner"
}
```

#### Log Waste
```http
POST /waste
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

pantry_entry_id=uuid
quantity=0.5
unit=kg
reason=SPOILED
notes=Found mold
photo=@waste_photo.jpg
```

**Response:** Includes estimated cost and carbon footprint.

### Predictions & Recommendations

#### Get Waste Risk Predictions
```http
GET /predictions/{household_id}?top=10&risk_class=HIGH
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "predictions": [
    {
      "id": "uuid",
      "pantry_entry_id": "uuid",
      "risk_score": 0.9,
      "risk_class": "HIGH",
      "predicted_waste_date": "2025-11-14T00:00:00Z",
      "confidence_score": 0.87,
      "recommended_actions": {
        "priority": "urgent",
        "actions": [
          "Consume immediately",
          "Check recipe recommendations"
        ]
      }
    }
  ],
  "total": 10,
  "high_risk_count": 5,
  "medium_risk_count": 3,
  "low_risk_count": 2
}
```

#### Generate Fresh Predictions
```http
POST /predictions/generate/{household_id}
Authorization: Bearer <access_token>
```

### Recipe Management

#### Search Recipes
```http
GET /recipes/search?ingredients=tomato,onion&diet=vegetarian&max_prep_time=30
```

#### Bulk Import Recipes
```http
POST /recipes/import
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "recipes": [
    {
      "name": "Tomato Soup",
      "ingredients": ["tomatoes", "onion", "garlic"],
      "instructions": "Cook and blend...",
      "servings": 4,
      "prep_time_minutes": 10,
      "cook_time_minutes": 20,
      "dietary_tags": ["vegetarian", "vegan"]
    }
  ]
}
```

### Notifications

#### Schedule Notification
```http
POST /notifications/schedule
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "household_id": "uuid",
  "notification_type": "EXPIRY_WARNING",
  "payload": {
    "item_name": "Milk",
    "days_until_expiry": 2
  },
  "scheduled_at": "2025-11-13T09:00:00Z",
  "channels": ["push", "email"]
}
```

### Admin Endpoints

#### Deploy ML Model
```http
POST /admin/models/deploy
Authorization: Bearer <admin_token>
Content-Type: application/json

{
  "model_artifact_uri": "gs://bucket/models/prediction_v2.pkl",
  "metadata": {
    "accuracy": 0.89,
    "trained_on": "2025-11-01"
  },
  "model_type": "prediction",
  "version": "2.1.0"
}
```

#### List Models
```http
GET /admin/models?model_type=prediction
Authorization: Bearer <admin_token>
```

### Metrics & Analytics

#### Get Household Metrics
```http
GET /household/{household_id}/metrics?days=30
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "household_id": "uuid",
  "period_start": "2025-10-13T00:00:00Z",
  "period_end": "2025-11-12T00:00:00Z",
  "waste_estimate": {
    "count": 12,
    "total_weight_kg": 3.5,
    "total_cost": 28.50,
    "carbon_footprint_kg": 9.8
  },
  "cost_saved": 45.20,
  "engagement": {
    "active_users": 4,
    "pantry_entries": 45,
    "consumption_logs": 89,
    "waste_events": 12
  },
  "top_wasted_items": [
    {
      "name": "Milk",
      "waste_count": 3,
      "total_wasted": 2.0
    }
  ],
  "waste_reasons_breakdown": {
    "EXPIRED": 7,
    "SPOILED": 3,
    "OVERCOOKED": 2
  }
}
```

## 🔒 Authentication

All protected endpoints require a JWT access token in the Authorization header:

```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

**Token Lifecycle:**
- Access tokens expire after 30 minutes
- Refresh tokens expire after 30 days
- Use `/auth/refresh` to get new access token
- Tokens are revoked on logout

## 🏗️ Architecture

```
backend/
├── api/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Settings management
│   ├── dependencies.py          # Auth & common dependencies
│   ├── schemas.py               # Pydantic models
│   ├── middleware/
│   │   ├── request_id.py        # Request tracking
│   │   ├── logging.py           # Request/response logging
│   │   └── rate_limit.py        # Rate limiting
│   ├── routers/
│   │   ├── auth.py              # Authentication
│   │   ├── households.py        # Household management
│   │   ├── pantry.py            # Pantry inventory
│   │   ├── purchases.py         # Receipt processing
│   │   ├── consumption.py       # Consumption logs
│   │   ├── waste.py             # Waste tracking
│   │   ├── predictions.py       # ML predictions
│   │   ├── recipes.py           # Recipe management
│   │   ├── notifications.py     # Notifications
│   │   ├── admin.py             # Admin endpoints
│   │   └── metrics.py           # Analytics
│   └── services/
│       ├── ocr_service.py       # Google Vision OCR
│       ├── ml_service.py        # ML predictions
│       └── notification_service.py  # Multi-channel notifications
├── shared/
│   ├── database_v2.py           # Database connection
│   └── models_v2.py             # SQLAlchemy models
├── requirements.txt
└── .env.example
```

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_auth.py -v
```

## 📦 Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pantrypal-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pantrypal-api
  template:
    metadata:
      labels:
        app: pantrypal-api
    spec:
      containers:
      - name: api
        image: pantrypal-api:2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pantrypal-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

## 🔧 Configuration

### Environment Variables

All configuration is via environment variables (see `.env.example`).

**Required:**
- `DATABASE_URL`
- `JWT_SECRET_KEY`

**Optional but recommended:**
- `GOOGLE_APPLICATION_CREDENTIALS` (for OCR)
- `OPENAI_API_KEY` (for recipe recommendations)
- `TWILIO_*` (for SMS)
- `SENDGRID_*` (for email)
- `FCM_*` (for push notifications)
- `SENTRY_DSN` (for error tracking)

## 📊 Monitoring

- **Health Check**: `GET /health` - Database and API status
- **Readiness**: `GET /ready` - Kubernetes readiness probe
- **Metrics**: Integrate Prometheus for detailed metrics
- **Logging**: Structured JSON logging to stdout
- **Tracing**: Request ID tracking via `X-Request-ID` header

## 🤝 Contributing

1. Follow Python PEP 8 style guide
2. Write unit tests for new features
3. Update API documentation
4. Use type hints everywhere
5. Run `black` and `isort` before committing

## 📄 License

MIT License - see LICENSE file

## 🆘 Support

- **Documentation**: http://localhost:8000/docs
- **Issues**: GitHub Issues
- **Email**: support@pantrypal.com
