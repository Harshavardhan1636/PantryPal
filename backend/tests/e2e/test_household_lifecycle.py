"""
End-to-End Tests using Playwright

Tests complete household lifecycle:
- User registration
- Receipt upload
- Pantry management
- Waste predictions
- Recipe recommendations
- Shopping list
"""

import pytest
from playwright.async_api import async_playwright, Page, Browser, expect
import asyncio
from datetime import datetime


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def browser():
    """Launch browser for testing."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser: Browser):
    """Create new page for each test."""
    context = await browser.new_context()
    page = await context.new_page()
    yield page
    await page.close()
    await context.close()


# ============================================================================
# E2E Test: Complete Household Lifecycle
# ============================================================================

@pytest.mark.e2e
class TestHouseholdLifecycle:
    """Test complete household lifecycle from registration to waste tracking."""
    
    BASE_URL = "http://localhost:3000"
    
    @pytest.mark.asyncio
    async def test_complete_household_journey(self, page: Page):
        """
        Complete user journey:
        1. Register account
        2. Upload receipt
        3. Verify pantry items created
        4. View waste predictions
        5. Get recipe recommendations
        6. Add to shopping list
        7. Mark item as consumed
        """
        
        # Step 1: Registration
        await page.goto(f"{self.BASE_URL}/register")
        
        await page.fill('input[name="email"]', f"e2e-test-{datetime.now().timestamp()}@example.com")
        await page.fill('input[name="password"]', "SecurePassword123!")
        await page.fill('input[name="household_name"]', "E2E Test Household")
        
        await page.click('button[type="submit"]')
        
        # Should redirect to dashboard
        await page.wait_for_url(f"{self.BASE_URL}/dashboard")
        assert "Dashboard" in await page.title()
        
        # Step 2: Upload receipt
        await page.goto(f"{self.BASE_URL}/receipts/upload")
        
        # Upload test receipt image
        await page.set_input_files(
            'input[type="file"]',
            'tests/fixtures/sample-receipt.jpg'
        )
        
        await page.click('button:has-text("Upload Receipt")')
        
        # Wait for processing
        await page.wait_for_selector('text=Processing complete', timeout=30000)
        
        # Step 3: Verify pantry items created
        await page.goto(f"{self.BASE_URL}/pantry")
        
        # Should see pantry items from receipt
        pantry_items = await page.locator('.pantry-item').count()
        assert pantry_items > 0, "No pantry items found after receipt upload"
        
        # Click on first item to view details
        await page.click('.pantry-item >> nth=0')
        
        # Should see item details
        await page.wait_for_selector('.item-details')
        
        # Step 4: View waste predictions
        await page.goto(f"{self.BASE_URL}/predictions")
        
        # Should see waste risk predictions
        risk_indicators = await page.locator('.risk-indicator').count()
        assert risk_indicators > 0, "No waste predictions found"
        
        # Check for high-risk items
        high_risk_items = await page.locator('.risk-indicator.high-risk').count()
        if high_risk_items > 0:
            # Click on high-risk item
            await page.click('.risk-indicator.high-risk >> nth=0')
            
            # Should see recommendation
            await page.wait_for_selector('text=Use soon')
        
        # Step 5: Get recipe recommendations
        await page.goto(f"{self.BASE_URL}/recipes")
        
        await page.click('button:has-text("Get Recommendations")')
        
        # Wait for AI recommendations
        await page.wait_for_selector('.recipe-card', timeout=10000)
        
        recipes = await page.locator('.recipe-card').count()
        assert recipes > 0, "No recipe recommendations found"
        
        # View recipe details
        await page.click('.recipe-card >> nth=0')
        await page.wait_for_selector('.recipe-details')
        
        # Check ingredients
        ingredients = await page.locator('.ingredient-item').count()
        assert ingredients > 0, "No ingredients listed"
        
        # Step 6: Add missing ingredient to shopping list
        await page.click('button:has-text("Add to Shopping List")')
        
        await page.goto(f"{self.BASE_URL}/shopping-list")
        
        shopping_items = await page.locator('.shopping-item').count()
        assert shopping_items > 0, "No shopping items found"
        
        # Step 7: Mark pantry item as consumed
        await page.goto(f"{self.BASE_URL}/pantry")
        
        # Click on item
        await page.click('.pantry-item >> nth=0')
        
        # Update quantity (consumed half)
        await page.fill('input[name="quantity"]', "0.5")
        await page.click('button:has-text("Update")')
        
        # Should see success message
        await page.wait_for_selector('text=Item updated')
        
        # Verify consumption recorded
        consumption_events = await page.locator('.consumption-event').count()
        assert consumption_events > 0, "Consumption not recorded"


# ============================================================================
# E2E Test: Receipt Processing
# ============================================================================

@pytest.mark.e2e
class TestReceiptProcessing:
    """Test receipt upload and OCR processing."""
    
    BASE_URL = "http://localhost:3000"
    
    @pytest.mark.asyncio
    async def test_receipt_upload_and_ocr(self, page: Page):
        """Test complete receipt processing flow."""
        
        # Login as existing user (created in setup)
        await page.goto(f"{self.BASE_URL}/login")
        await page.fill('input[name="email"]', "test@example.com")
        await page.fill('input[name="password"]', "SecurePassword123!")
        await page.click('button[type="submit"]')
        
        await page.wait_for_url(f"{self.BASE_URL}/dashboard")
        
        # Navigate to receipt upload
        await page.goto(f"{self.BASE_URL}/receipts/upload")
        
        # Upload receipt
        await page.set_input_files(
            'input[type="file"]',
            'tests/fixtures/grocery-receipt.jpg'
        )
        
        # Should show preview
        await page.wait_for_selector('.receipt-preview')
        
        # Click process
        await page.click('button:has-text("Process Receipt")')
        
        # Wait for OCR processing
        await page.wait_for_selector('text=Processing...', timeout=5000)
        await page.wait_for_selector('text=Complete', timeout=30000)
        
        # Should show extracted items
        extracted_items = await page.locator('.extracted-item').count()
        assert extracted_items > 0, "No items extracted from receipt"
        
        # Verify item details
        first_item = page.locator('.extracted-item >> nth=0')
        await expect(first_item).to_contain_text('$')  # Has price
        
        # Confirm and add to pantry
        await page.click('button:has-text("Add to Pantry")')
        
        await page.wait_for_selector('text=Items added to pantry')
        
        # Navigate to pantry
        await page.goto(f"{self.BASE_URL}/pantry")
        
        # Should see new items
        pantry_count = await page.locator('.pantry-item').count()
        assert pantry_count >= extracted_items


# ============================================================================
# E2E Test: Waste Prediction Flow
# ============================================================================

@pytest.mark.e2e
class TestWastePredictionFlow:
    """Test waste prediction and recommendations."""
    
    BASE_URL = "http://localhost:3000"
    
    @pytest.mark.asyncio
    async def test_waste_prediction_lifecycle(self, page: Page):
        """Test waste prediction from creation to action."""
        
        # Login
        await page.goto(f"{self.BASE_URL}/login")
        await page.fill('input[name="email"]', "test@example.com")
        await page.fill('input[name="password"]', "SecurePassword123!")
        await page.click('button[type="submit"]')
        
        # Add item that will be at risk
        await page.goto(f"{self.BASE_URL}/pantry/add")
        
        await page.fill('input[name="name"]', "Strawberries")
        await page.select_option('select[name="category"]', "fruit")
        await page.fill('input[name="quantity"]', "1")
        await page.select_option('select[name="unit"]', "lb")
        
        # Set expiration to 2 days from now (high risk)
        from datetime import datetime, timedelta
        expiry = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
        await page.fill('input[name="expiration_date"]', expiry)
        
        await page.click('button:has-text("Add Item")')
        
        # Wait for prediction generation
        await page.wait_for_timeout(3000)
        
        # Check predictions page
        await page.goto(f"{self.BASE_URL}/predictions")
        
        # Should see strawberries with high risk
        strawberry_item = page.locator('text=Strawberries').first
        await strawberry_item.wait_for()
        
        # Check risk indicator
        risk_badge = page.locator('.risk-indicator.high-risk', has=page.locator('text=Strawberries'))
        await expect(risk_badge).to_be_visible()
        
        # Click to view details
        await strawberry_item.click()
        
        # Should see recommendations
        await page.wait_for_selector('text=Use within')
        await page.wait_for_selector('.recipe-suggestion')
        
        # Click recipe suggestion
        await page.click('.recipe-suggestion >> nth=0')
        
        # Should navigate to recipe
        await page.wait_for_selector('.recipe-details')


# ============================================================================
# E2E Test: Multi-User Household
# ============================================================================

@pytest.mark.e2e
class TestMultiUserHousehold:
    """Test multi-user household collaboration."""
    
    BASE_URL = "http://localhost:3000"
    
    @pytest.mark.asyncio
    async def test_household_member_collaboration(self, page: Page):
        """Test that household members see shared data."""
        
        # User 1: Create household and add item
        await page.goto(f"{self.BASE_URL}/register")
        
        email1 = f"user1-{datetime.now().timestamp()}@example.com"
        await page.fill('input[name="email"]', email1)
        await page.fill('input[name="password"]', "Password123!")
        await page.fill('input[name="household_name"]', "Shared Household")
        await page.click('button[type="submit"]')
        
        # Add pantry item
        await page.goto(f"{self.BASE_URL}/pantry/add")
        await page.fill('input[name="name"]', "Shared Milk")
        await page.click('button:has-text("Add Item")')
        
        # Get household invite link
        await page.goto(f"{self.BASE_URL}/household/settings")
        invite_code = await page.locator('.invite-code').text_content()
        
        # Logout
        await page.click('button:has-text("Logout")')
        
        # User 2: Register and join household
        await page.goto(f"{self.BASE_URL}/register")
        
        email2 = f"user2-{datetime.now().timestamp()}@example.com"
        await page.fill('input[name="email"]', email2)
        await page.fill('input[name="password"]', "Password123!")
        await page.fill('input[name="invite_code"]', invite_code)
        await page.click('button[type="submit"]')
        
        # Should see shared pantry item
        await page.goto(f"{self.BASE_URL}/pantry")
        
        shared_item = page.locator('text=Shared Milk')
        await expect(shared_item).to_be_visible()


# ============================================================================
# Run E2E Tests
# ============================================================================

# Run with: pytest backend/tests/e2e/ -v -m e2e
# Requires frontend running at localhost:3000
# Install: pip install pytest-playwright && playwright install
