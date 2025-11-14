"""
Integration Tests for PantryPal Core API Flows

Tests:
- Database migrations
- API flow: pantry add → prediction created
- User registration and authentication
- Receipt processing pipeline
- Shopping list synchronization
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi import status
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import os

from backend.api.main import app
from backend.database import Base, get_db
from backend.shared.models import User, Household, PantryItem, WastePrediction, Receipt


# ============================================================================
# Test Database Setup
# ============================================================================

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://test:test@localhost:5432/pantrypal_test"
)

test_engine = create_engine(TEST_DATABASE_URL)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session")
def setup_test_database():
    """Create test database schema."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session(setup_test_database):
    """Create a fresh database session for each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def override_get_db(db_session):
    """Override database dependency for testing."""
    async def _override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def async_client(override_get_db):
    """Async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# ============================================================================
# Database Migration Tests
# ============================================================================

class TestDatabaseMigrations:
    """Test database migrations and schema."""
    
    def test_all_tables_created(self, db_session):
        """Verify all tables are created."""
        from sqlalchemy import inspect
        
        inspector = inspect(test_engine)
        tables = inspector.get_table_names()
        
        expected_tables = [
            'users', 'households', 'pantry_items', 'shopping_items',
            'receipts', 'waste_predictions', 'user_feedback',
            'recipe_recommendations'
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not found"
    
    def test_foreign_key_constraints(self, db_session):
        """Verify foreign key relationships."""
        from sqlalchemy import inspect
        
        inspector = inspect(test_engine)
        
        # Check pantry_items → households FK
        fks = inspector.get_foreign_keys('pantry_items')
        assert any(fk['referred_table'] == 'households' for fk in fks)
        
        # Check waste_predictions → pantry_items FK
        fks = inspector.get_foreign_keys('waste_predictions')
        assert any(fk['referred_table'] == 'pantry_items' for fk in fks)
    
    def test_unique_constraints(self, db_session):
        """Verify unique constraints."""
        from sqlalchemy import inspect
        
        inspector = inspect(test_engine)
        
        # Check users.email is unique
        constraints = inspector.get_unique_constraints('users')
        email_unique = any('email' in c['column_names'] for c in constraints)
        assert email_unique, "Email should have unique constraint"


# ============================================================================
# User Registration and Authentication Tests
# ============================================================================

class TestUserAuthentication:
    """Test user registration and authentication flow."""
    
    @pytest.mark.asyncio
    async def test_user_registration(self, async_client):
        """Test user registration endpoint."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePassword123!",
                "household_name": "Test Household"
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        assert "user_id" in data
        assert "household_id" in data
        assert data["email"] == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_duplicate_email_registration(self, async_client):
        """Test duplicate email registration fails."""
        user_data = {
            "email": "duplicate@example.com",
            "password": "SecurePassword123!",
            "household_name": "Test Household"
        }
        
        # First registration
        response1 = await async_client.post("/api/v1/auth/register", json=user_data)
        assert response1.status_code == status.HTTP_201_CREATED
        
        # Duplicate registration
        response2 = await async_client.post("/api/v1/auth/register", json=user_data)
        assert response2.status_code == status.HTTP_400_BAD_REQUEST
    
    @pytest.mark.asyncio
    async def test_user_login(self, async_client):
        """Test user login and JWT token generation."""
        # Register user
        await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "login@example.com",
                "password": "SecurePassword123!",
                "household_name": "Test Household"
            }
        )
        
        # Login
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "login@example.com",
                "password": "SecurePassword123!"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_invalid_login_credentials(self, async_client):
        """Test login with invalid credentials."""
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "WrongPassword"
            }
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ============================================================================
# Core API Flow Tests: Pantry Add → Prediction Created
# ============================================================================

class TestPantryPredictionFlow:
    """Test complete flow from adding pantry item to prediction creation."""
    
    @pytest.fixture
    async def authenticated_client(self, async_client):
        """Create authenticated client with JWT token."""
        # Register and login
        await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "flow@example.com",
                "password": "SecurePassword123!",
                "household_name": "Flow Test Household"
            }
        )
        
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "flow@example.com",
                "password": "SecurePassword123!"
            }
        )
        
        token = login_response.json()["access_token"]
        async_client.headers["Authorization"] = f"Bearer {token}"
        
        yield async_client
    
    @pytest.mark.asyncio
    async def test_add_pantry_item_creates_prediction(self, authenticated_client, db_session):
        """Test that adding a pantry item triggers prediction creation."""
        
        # Step 1: Add pantry item
        pantry_response = await authenticated_client.post(
            "/api/v1/pantry-items",
            json={
                "name": "Whole Milk",
                "category": "dairy",
                "quantity": 1.0,
                "unit": "gallon",
                "purchase_date": datetime.now().isoformat(),
                "expiration_date": (datetime.now() + timedelta(days=7)).isoformat(),
                "location": "refrigerator"
            }
        )
        
        assert pantry_response.status_code == status.HTTP_201_CREATED
        pantry_item = pantry_response.json()
        pantry_item_id = pantry_item["id"]
        
        # Step 2: Wait for async prediction job (simulate)
        await asyncio.sleep(2)
        
        # Step 3: Verify prediction was created
        prediction_response = await authenticated_client.get(
            f"/api/v1/predictions?pantry_item_id={pantry_item_id}"
        )
        
        assert prediction_response.status_code == status.HTTP_200_OK
        predictions = prediction_response.json()
        
        assert len(predictions) > 0
        prediction = predictions[0]
        
        assert prediction["pantry_item_id"] == pantry_item_id
        assert "waste_risk_score" in prediction
        assert 0 <= prediction["waste_risk_score"] <= 1
        assert "days_until_waste" in prediction
    
    @pytest.mark.asyncio
    async def test_update_pantry_item_updates_prediction(self, authenticated_client):
        """Test that updating a pantry item updates its prediction."""
        
        # Add pantry item
        pantry_response = await authenticated_client.post(
            "/api/v1/pantry-items",
            json={
                "name": "Chicken Breast",
                "category": "meat",
                "quantity": 2.0,
                "unit": "pound",
                "purchase_date": datetime.now().isoformat(),
                "expiration_date": (datetime.now() + timedelta(days=3)).isoformat(),
                "location": "refrigerator"
            }
        )
        pantry_item_id = pantry_response.json()["id"]
        
        # Get initial prediction
        pred_response_1 = await authenticated_client.get(
            f"/api/v1/predictions?pantry_item_id={pantry_item_id}"
        )
        initial_prediction = pred_response_1.json()[0]
        initial_risk_score = initial_prediction["waste_risk_score"]
        
        # Update quantity (consumed some)
        await authenticated_client.put(
            f"/api/v1/pantry-items/{pantry_item_id}",
            json={"quantity": 0.5}
        )
        
        await asyncio.sleep(1)
        
        # Get updated prediction
        pred_response_2 = await authenticated_client.get(
            f"/api/v1/predictions?pantry_item_id={pantry_item_id}"
        )
        updated_prediction = pred_response_2.json()[0]
        updated_risk_score = updated_prediction["waste_risk_score"]
        
        # Risk score should change based on consumption
        assert updated_risk_score != initial_risk_score


# ============================================================================
# Receipt Processing Pipeline Tests
# ============================================================================

class TestReceiptProcessing:
    """Test receipt OCR and item extraction pipeline."""
    
    @pytest.fixture
    async def authenticated_client(self, async_client):
        """Create authenticated client."""
        await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "receipt@example.com",
                "password": "SecurePassword123!",
                "household_name": "Receipt Test Household"
            }
        )
        
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={"email": "receipt@example.com", "password": "SecurePassword123!"}
        )
        
        token = login_response.json()["access_token"]
        async_client.headers["Authorization"] = f"Bearer {token}"
        
        yield async_client
    
    @pytest.mark.asyncio
    async def test_upload_receipt_image(self, authenticated_client):
        """Test receipt image upload and OCR processing."""
        
        # Simulate image upload
        files = {
            "file": ("receipt.jpg", b"fake_image_data", "image/jpeg")
        }
        
        response = await authenticated_client.post(
            "/api/v1/receipts/upload",
            files=files
        )
        
        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        
        assert "receipt_id" in data
        assert "status" in data
        assert data["status"] == "processing"
    
    @pytest.mark.asyncio
    async def test_receipt_ocr_creates_pantry_items(self, authenticated_client):
        """Test that OCR results create pantry items."""
        
        # Upload receipt
        files = {"file": ("receipt.jpg", b"fake_image_data", "image/jpeg")}
        upload_response = await authenticated_client.post(
            "/api/v1/receipts/upload",
            files=files
        )
        receipt_id = upload_response.json()["receipt_id"]
        
        # Wait for OCR processing
        await asyncio.sleep(3)
        
        # Check pantry items created
        pantry_response = await authenticated_client.get("/api/v1/pantry-items")
        pantry_items = pantry_response.json()
        
        # Should have items from receipt
        assert len(pantry_items) > 0


# ============================================================================
# Shopping List Integration Tests
# ============================================================================

class TestShoppingListIntegration:
    """Test shopping list synchronization and purchase flow."""
    
    @pytest.fixture
    async def authenticated_client(self, async_client):
        """Create authenticated client."""
        await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "shopping@example.com",
                "password": "SecurePassword123!",
                "household_name": "Shopping Test Household"
            }
        )
        
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={"email": "shopping@example.com", "password": "SecurePassword123!"}
        )
        
        token = login_response.json()["access_token"]
        async_client.headers["Authorization"] = f"Bearer {token}"
        
        yield async_client
    
    @pytest.mark.asyncio
    async def test_add_item_to_shopping_list(self, authenticated_client):
        """Test adding item to shopping list."""
        response = await authenticated_client.post(
            "/api/v1/shopping-list",
            json={
                "name": "Eggs",
                "quantity": 12,
                "unit": "count",
                "notes": "Large eggs"
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        assert data["name"] == "Eggs"
        assert data["purchased"] is False
    
    @pytest.mark.asyncio
    async def test_mark_item_purchased_creates_pantry_item(self, authenticated_client):
        """Test that marking shopping item as purchased creates pantry item."""
        
        # Add to shopping list
        shopping_response = await authenticated_client.post(
            "/api/v1/shopping-list",
            json={"name": "Bread", "quantity": 1, "unit": "loaf"}
        )
        shopping_item_id = shopping_response.json()["id"]
        
        # Mark as purchased
        purchase_response = await authenticated_client.put(
            f"/api/v1/shopping-list/{shopping_item_id}/purchase"
        )
        
        assert purchase_response.status_code == status.HTTP_200_OK
        
        # Verify pantry item created
        pantry_response = await authenticated_client.get("/api/v1/pantry-items")
        pantry_items = pantry_response.json()
        
        bread_items = [item for item in pantry_items if item["name"] == "Bread"]
        assert len(bread_items) > 0


# ============================================================================
# Run Integration Tests
# ============================================================================

# Run with: pytest backend/tests/integration/ -v -s
# Requires PostgreSQL and Redis running locally for full integration tests
