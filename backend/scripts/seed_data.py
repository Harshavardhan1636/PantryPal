"""
Database Seed Script
Senior SDE-3 Level Implementation

Seeds the database with:
- Demo organization and household
- Sample users
- Item catalog (common groceries)
- Sample pantry inventory
- Sample recipes
- Shopping list
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://pantrypal:pantrypal_dev_password@localhost:5432/pantrypal")

# ==============================================
# SAMPLE DATA
# ==============================================

SAMPLE_USERS = [
    {
        "email": "demo@pantrypal.com",
        "name": "Demo User",
        "password": "demo123",
    },
    {
        "email": "john@example.com",
        "name": "John Smith",
        "password": "password123",
    },
]

SAMPLE_ITEMS_CATALOG = [
    # Dairy
    {"name": "Milk (Whole)", "category": "dairy", "avg_shelf_life_days": 7, "typical_storage_type": "fridge", "unit": "L"},
    {"name": "Yogurt (Plain)", "category": "dairy", "avg_shelf_life_days": 14, "typical_storage_type": "fridge", "unit": "g"},
    {"name": "Cheese (Cheddar)", "category": "dairy", "avg_shelf_life_days": 21, "typical_storage_type": "fridge", "unit": "g"},
    {"name": "Butter", "category": "dairy", "avg_shelf_life_days": 30, "typical_storage_type": "fridge", "unit": "g"},
    
    # Vegetables
    {"name": "Tomatoes", "category": "vegetables", "avg_shelf_life_days": 5, "typical_storage_type": "counter", "unit": "piece"},
    {"name": "Lettuce", "category": "vegetables", "avg_shelf_life_days": 7, "typical_storage_type": "fridge", "unit": "piece"},
    {"name": "Carrots", "category": "vegetables", "avg_shelf_life_days": 14, "typical_storage_type": "fridge", "unit": "piece"},
    {"name": "Onions", "category": "vegetables", "avg_shelf_life_days": 30, "typical_storage_type": "pantry", "unit": "piece"},
    {"name": "Potatoes", "category": "vegetables", "avg_shelf_life_days": 30, "typical_storage_type": "pantry", "unit": "piece"},
    {"name": "Bell Peppers", "category": "vegetables", "avg_shelf_life_days": 7, "typical_storage_type": "fridge", "unit": "piece"},
    
    # Fruits
    {"name": "Apples", "category": "fruits", "avg_shelf_life_days": 14, "typical_storage_type": "counter", "unit": "piece"},
    {"name": "Bananas", "category": "fruits", "avg_shelf_life_days": 5, "typical_storage_type": "counter", "unit": "piece"},
    {"name": "Oranges", "category": "fruits", "avg_shelf_life_days": 14, "typical_storage_type": "counter", "unit": "piece"},
    {"name": "Strawberries", "category": "fruits", "avg_shelf_life_days": 3, "typical_storage_type": "fridge", "unit": "g"},
    
    # Meat & Poultry
    {"name": "Chicken Breast", "category": "meat", "avg_shelf_life_days": 2, "typical_storage_type": "fridge", "unit": "g"},
    {"name": "Ground Beef", "category": "meat", "avg_shelf_life_days": 2, "typical_storage_type": "fridge", "unit": "g"},
    {"name": "Salmon Fillet", "category": "seafood", "avg_shelf_life_days": 2, "typical_storage_type": "fridge", "unit": "g"},
    
    # Pantry Staples
    {"name": "Rice (White)", "category": "grains", "avg_shelf_life_days": 365, "typical_storage_type": "pantry", "unit": "g"},
    {"name": "Pasta", "category": "grains", "avg_shelf_life_days": 365, "typical_storage_type": "pantry", "unit": "g"},
    {"name": "Bread (White)", "category": "bakery", "avg_shelf_life_days": 5, "typical_storage_type": "counter", "unit": "piece"},
    {"name": "Eggs", "category": "dairy", "avg_shelf_life_days": 21, "typical_storage_type": "fridge", "unit": "piece"},
]

SAMPLE_RECIPES = [
    {
        "name": "Tomato Pasta",
        "description": "Simple pasta with fresh tomato sauce",
        "ingredients": [
            {"name": "Pasta", "quantity": 200, "unit": "g"},
            {"name": "Tomatoes", "quantity": 3, "unit": "piece"},
            {"name": "Onions", "quantity": 1, "unit": "piece"},
        ],
        "instructions": "1. Cook pasta. 2. Make tomato sauce. 3. Combine and serve.",
        "prep_time_minutes": 10,
        "cook_time_minutes": 20,
        "servings": 2,
        "difficulty": "easy",
        "cuisine": "Italian",
        "dietary_tags": ["vegetarian"],
    },
    {
        "name": "Greek Salad",
        "description": "Fresh and healthy Greek salad",
        "ingredients": [
            {"name": "Lettuce", "quantity": 1, "unit": "piece"},
            {"name": "Tomatoes", "quantity": 2, "unit": "piece"},
            {"name": "Cheese (Cheddar)", "quantity": 100, "unit": "g"},
        ],
        "instructions": "1. Chop vegetables. 2. Add cheese. 3. Dress and serve.",
        "prep_time_minutes": 15,
        "cook_time_minutes": 0,
        "servings": 2,
        "difficulty": "easy",
        "cuisine": "Greek",
        "dietary_tags": ["vegetarian", "healthy"],
    },
]

# ==============================================
# SEED FUNCTIONS
# ==============================================

async def seed_database():
    """Main seeding function"""
    print("=" * 60)
    print("  PantryPal Database Seeding Script")
    print("  Senior SDE-3 Level Implementation")
    print("=" * 60)
    print()
    
    # Create async engine
    engine = create_async_engine(DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        try:
            # Seed organization and household
            print("[STEP] Creating organization and household...")
            org_id, household_id = await seed_organization_and_household(session)
            print(f"[✓] Organization ID: {org_id}")
            print(f"[✓] Household ID: {household_id}")
            
            # Seed users
            print("\n[STEP] Creating users...")
            user_ids = await seed_users(session, household_id)
            print(f"[✓] Created {len(user_ids)} users")
            
            # Seed items catalog
            print("\n[STEP] Seeding items catalog...")
            item_ids = await seed_items_catalog(session)
            print(f"[✓] Created {len(item_ids)} catalog items")
            
            # Seed inventory
            print("\n[STEP] Seeding inventory...")
            batch_ids = await seed_inventory(session, household_id, item_ids, user_ids[0])
            print(f"[✓] Created {len(batch_ids)} inventory batches")
            
            # Seed recipes
            print("\n[STEP] Seeding recipes...")
            recipe_ids = await seed_recipes(session)
            print(f"[✓] Created {len(recipe_ids)} recipes")
            
            # Seed shopping list
            print("\n[STEP] Creating shopping list...")
            list_id = await seed_shopping_list(session, household_id, user_ids[0])
            print(f"[✓] Shopping list ID: {list_id}")
            
            # Commit all changes
            await session.commit()
            
            print("\n" + "=" * 60)
            print("  DATABASE SEEDED SUCCESSFULLY")
            print("=" * 60)
            print("\nDemo Credentials:")
            print("  Email: demo@pantrypal.com")
            print("  Password: demo123")
            print()
            
        except Exception as e:
            await session.rollback()
            print(f"\n[✗] Error: {str(e)}")
            raise
        finally:
            await engine.dispose()

async def seed_organization_and_household(session: AsyncSession):
    """Seed organization and household"""
    from sqlalchemy import text
    
    # Insert organization
    org_query = text("""
        INSERT INTO organizations (name, plan, max_households)
        VALUES ('Demo Organization', 'pro', 10)
        ON CONFLICT DO NOTHING
        RETURNING id
    """)
    result = await session.execute(org_query)
    org_row = result.fetchone()
    
    if not org_row:
        # Get existing org
        result = await session.execute(text("SELECT id FROM organizations WHERE name = 'Demo Organization'"))
        org_row = result.fetchone()
    
    org_id = org_row[0]
    
    # Insert household
    household_query = text("""
        INSERT INTO households (org_id, name, members_count, storage_types, dietary_preferences, timezone)
        VALUES (:org_id, 'Demo Household', 2, 
                '["refrigerator", "pantry", "freezer"]'::jsonb, 
                '["vegetarian"]'::jsonb, 
                'Asia/Kolkata')
        ON CONFLICT DO NOTHING
        RETURNING id
    """)
    result = await session.execute(household_query, {"org_id": org_id})
    household_row = result.fetchone()
    
    if not household_row:
        result = await session.execute(text("SELECT id FROM households WHERE name = 'Demo Household'"))
        household_row = result.fetchone()
    
    household_id = household_row[0]
    
    return org_id, household_id

async def seed_users(session: AsyncSession, household_id: str):
    """Seed demo users"""
    from sqlalchemy import text
    
    user_ids = []
    
    for user_data in SAMPLE_USERS:
        password_hash = pwd_context.hash(user_data["password"])
        
        query = text("""
            INSERT INTO users (email, name, password_hash, email_verified)
            VALUES (:email, :name, :password_hash, true)
            ON CONFLICT (email) DO NOTHING
            RETURNING id
        """)
        
        result = await session.execute(query, {
            "email": user_data["email"],
            "name": user_data["name"],
            "password_hash": password_hash,
        })
        
        row = result.fetchone()
        if row:
            user_id = row[0]
        else:
            result = await session.execute(text("SELECT id FROM users WHERE email = :email"), {"email": user_data["email"]})
            user_id = result.fetchone()[0]
        
        user_ids.append(user_id)
        
        # Add user to household
        member_query = text("""
            INSERT INTO household_members (household_id, user_id, role)
            VALUES (:household_id, :user_id, 'owner')
            ON CONFLICT DO NOTHING
        """)
        await session.execute(member_query, {"household_id": household_id, "user_id": user_id})
    
    return user_ids

async def seed_items_catalog(session: AsyncSession):
    """Seed items catalog"""
    from sqlalchemy import text
    
    item_ids = []
    
    for item in SAMPLE_ITEMS_CATALOG:
        query = text("""
            INSERT INTO items_catalog (name, category, avg_shelf_life_days, typical_storage_type)
            VALUES (:name, :category, :avg_shelf_life_days, :typical_storage_type)
            ON CONFLICT DO NOTHING
            RETURNING id
        """)
        
        result = await session.execute(query, item)
        row = result.fetchone()
        if row:
            item_ids.append(row[0])
    
    return item_ids

async def seed_inventory(session: AsyncSession, household_id: str, item_ids: list, user_id: str):
    """Seed inventory batches"""
    from sqlalchemy import text
    
    batch_ids = []
    today = datetime.now()
    
    # Create some items that are at risk
    inventory_items = [
        {"item_idx": 0, "quantity": 1.5, "unit": "L", "days_old": 5, "expiry_days": 2},  # Milk - at risk
        {"item_idx": 4, "quantity": 5, "unit": "piece", "days_old": 4, "expiry_days": 1},  # Tomatoes - at risk
        {"item_idx": 13, "quantity": 250, "unit": "g", "days_old": 2, "expiry_days": 1},  # Strawberries - at risk
        {"item_idx": 3, "quantity": 200, "unit": "g", "days_old": 10, "expiry_days": 20},  # Butter - ok
        {"item_idx": 6, "quantity": 10, "unit": "piece", "days_old": 5, "expiry_days": 9},  # Carrots - ok
    ]
    
    for item_data in inventory_items:
        if item_data["item_idx"] >= len(item_ids):
            continue
            
        purchase_date = today - timedelta(days=item_data["days_old"])
        expiry_date = today + timedelta(days=item_data["expiry_days"])
        
        query = text("""
            INSERT INTO inventory_batches 
            (household_id, item_id, quantity, unit, purchase_date, expiry_date, storage_type, status, created_by)
            VALUES (:household_id, :item_id, :quantity, :unit, :purchase_date, :expiry_date, :storage_type, 'active', :created_by)
            RETURNING id
        """)
        
        result = await session.execute(query, {
            "household_id": household_id,
            "item_id": item_ids[item_data["item_idx"]],
            "quantity": item_data["quantity"],
            "unit": item_data["unit"],
            "purchase_date": purchase_date.date(),
            "expiry_date": expiry_date.date(),
            "storage_type": "fridge",
            "created_by": user_id,
        })
        
        batch_ids.append(result.fetchone()[0])
    
    return batch_ids

async def seed_recipes(session: AsyncSession):
    """Seed recipes"""
    from sqlalchemy import text
    
    recipe_ids = []
    
    for recipe in SAMPLE_RECIPES:
        query = text("""
            INSERT INTO recipes 
            (name, description, ingredients, instructions, prep_time_minutes, cook_time_minutes, 
             servings, difficulty, cuisine, dietary_tags)
            VALUES (:name, :description, :ingredients, :instructions, :prep_time_minutes, :cook_time_minutes,
                    :servings, :difficulty, :cuisine, :dietary_tags)
            ON CONFLICT DO NOTHING
            RETURNING id
        """)
        
        result = await session.execute(query, {
            **recipe,
            "ingredients": json.dumps(recipe["ingredients"]),
            "dietary_tags": json.dumps(recipe["dietary_tags"]),
        })
        
        row = result.fetchone()
        if row:
            recipe_ids.append(row[0])
    
    return recipe_ids

async def seed_shopping_list(session: AsyncSession, household_id: str, user_id: str):
    """Seed shopping list"""
    from sqlalchemy import text
    
    # Create shopping list
    list_query = text("""
        INSERT INTO shopping_lists (household_id, name, status, created_by)
        VALUES (:household_id, 'Weekly Shopping', 'active', :created_by)
        RETURNING id
    """)
    
    result = await session.execute(list_query, {"household_id": household_id, "created_by": user_id})
    list_id = result.fetchone()[0]
    
    # Add items to list
    items = [
        {"name": "Milk", "quantity": 2, "unit": "L"},
        {"name": "Bread", "quantity": 1, "unit": "piece"},
        {"name": "Eggs", "quantity": 12, "unit": "piece"},
    ]
    
    for item in items:
        item_query = text("""
            INSERT INTO shopping_list_items (list_id, custom_item_name, quantity, unit)
            VALUES (:list_id, :custom_item_name, :quantity, :unit)
        """)
        
        await session.execute(item_query, {"list_id": list_id, **item})
    
    return list_id

# ==============================================
# MAIN
# ==============================================

if __name__ == "__main__":
    asyncio.run(seed_database())
