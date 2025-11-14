"""
Enhanced Synthetic Data Generator with 50+ Features
Generates production-ready training data for waste risk prediction.

Features:
- 5 household archetypes with realistic behaviors
- 13 food categories with specific characteristics
- 50+ engineered features
- 25% target waste rate
- Temporal patterns (weekends, holidays, seasons)
- Consumption modeling
- Purchase patterns

Usage:
    python generate_data_enhanced.py --n-households 1000 --output data/training_100k.parquet
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import random
from enum import Enum
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from faker import Faker

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class HouseholdArchetype(str, Enum):
    """Behavioral household archetypes"""
    YOUNG_PROFESSIONAL = "young_professional"
    FAMILY_WITH_KIDS = "family_with_kids"
    RETIREE = "retiree"
    COLLEGE_STUDENTS = "college_students"
    BUSY_PARENTS = "busy_parents"


class EnhancedSyntheticDataGenerator:
    """Generate high-quality synthetic training data with 50+ features."""
    
    # Household archetypes with behavioral parameters
    ARCHETYPES = {
        HouseholdArchetype.YOUNG_PROFESSIONAL: {
            "household_size_range": (1, 2),
            "base_waste_rate": 0.32,  # Higher waste
            "shopping_frequency_per_week": (2, 3),
            "consumption_consistency": 0.6,  # Less consistent
            "impulse_buy_rate": 0.20,
            "bulk_purchase_rate": 0.10,
            "forgotten_item_rate": 0.20,
            "planning_ahead_days": (1, 3),
        },
        HouseholdArchetype.FAMILY_WITH_KIDS: {
            "household_size_range": (4, 5),
            "base_waste_rate": 0.25,  # Medium waste
            "shopping_frequency_per_week": (1, 2),
            "consumption_consistency": 0.8,
            "impulse_buy_rate": 0.15,
            "bulk_purchase_rate": 0.40,  # High bulk buying
            "forgotten_item_rate": 0.15,
            "planning_ahead_days": (5, 7),
        },
        HouseholdArchetype.RETIREE: {
            "household_size_range": (1, 2),
            "base_waste_rate": 0.15,  # Low waste
            "shopping_frequency_per_week": (2, 3),
            "consumption_consistency": 0.9,
            "impulse_buy_rate": 0.05,
            "bulk_purchase_rate": 0.20,
            "forgotten_item_rate": 0.10,
            "planning_ahead_days": (7, 14),
        },
        HouseholdArchetype.COLLEGE_STUDENTS: {
            "household_size_range": (2, 4),
            "base_waste_rate": 0.30,
            "shopping_frequency_per_week": (1, 2),
            "consumption_consistency": 0.5,  # Very inconsistent
            "impulse_buy_rate": 0.25,
            "bulk_purchase_rate": 0.15,
            "forgotten_item_rate": 0.25,
            "planning_ahead_days": (1, 2),
        },
        HouseholdArchetype.BUSY_PARENTS: {
            "household_size_range": (3, 4),
            "base_waste_rate": 0.28,
            "shopping_frequency_per_week": (1, 2),
            "consumption_consistency": 0.7,
            "impulse_buy_rate": 0.18,
            "bulk_purchase_rate": 0.35,
            "forgotten_item_rate": 0.18,
            "planning_ahead_days": (3, 5),
        },
    }
    
    # 13 food categories
    FOOD_CATEGORIES = {
        "dairy": {
            "items": ["Whole Milk", "2% Milk", "Cheddar Cheese", "Mozzarella", "Greek Yogurt", "Butter", "Cream Cheese"],
            "shelf_life_range": (5, 21),
            "base_waste_prob": 0.28,
            "price_range": (2, 10),
            "typical_consumption_rate_days": 7,
        },
        "produce": {
            "items": ["Apples", "Bananas", "Lettuce", "Tomatoes", "Carrots", "Broccoli", "Spinach", "Potatoes", "Onions", "Avocados"],
            "shelf_life_range": (3, 14),
            "base_waste_prob": 0.38,  # Highest
            "price_range": (1, 8),
            "typical_consumption_rate_days": 5,
        },
        "meat": {
            "items": ["Chicken Breast", "Ground Beef", "Pork Chops", "Salmon", "Turkey", "Bacon", "Sausage"],
            "shelf_life_range": (2, 7),
            "base_waste_prob": 0.22,
            "price_range": (5, 25),
            "typical_consumption_rate_days": 4,
        },
        "grains": {
            "items": ["White Rice", "Brown Rice", "Pasta", "Bread", "Bagels", "Cereal", "Oatmeal"],
            "shelf_life_range": (14, 365),
            "base_waste_prob": 0.12,
            "price_range": (2, 12),
            "typical_consumption_rate_days": 30,
        },
        "frozen": {
            "items": ["Frozen Pizza", "Ice Cream", "Frozen Vegetables", "Frozen Fruit", "Frozen Meals"],
            "shelf_life_range": (90, 365),
            "base_waste_prob": 0.08,
            "price_range": (3, 15),
            "typical_consumption_rate_days": 60,
        },
        "canned": {
            "items": ["Canned Beans", "Canned Tomatoes", "Soup", "Tuna", "Canned Corn"],
            "shelf_life_range": (365, 730),
            "base_waste_prob": 0.05,
            "price_range": (1, 6),
            "typical_consumption_rate_days": 90,
        },
        "snacks": {
            "items": ["Chips", "Crackers", "Cookies", "Granola Bars", "Nuts"],
            "shelf_life_range": (30, 180),
            "base_waste_prob": 0.15,
            "price_range": (2, 8),
            "typical_consumption_rate_days": 14,
        },
        "beverages": {
            "items": ["Orange Juice", "Apple Juice", "Soda", "Sports Drinks", "Coffee"],
            "shelf_life_range": (14, 90),
            "base_waste_prob": 0.18,
            "price_range": (2, 10),
            "typical_consumption_rate_days": 10,
        },
        "condiments": {
            "items": ["Ketchup", "Mustard", "Mayo", "Salad Dressing", "Hot Sauce", "Soy Sauce"],
            "shelf_life_range": (180, 365),
            "base_waste_prob": 0.10,
            "price_range": (2, 7),
            "typical_consumption_rate_days": 60,
        },
        "bakery": {
            "items": ["Croissants", "Donuts", "Muffins", "Cake", "Pies"],
            "shelf_life_range": (2, 7),
            "base_waste_prob": 0.30,
            "price_range": (3, 15),
            "typical_consumption_rate_days": 3,
        },
        "prepared": {
            "items": ["Deli Salad", "Rotisserie Chicken", "Prepared Meals", "Sushi"],
            "shelf_life_range": (1, 3),
            "base_waste_prob": 0.35,
            "price_range": (5, 20),
            "typical_consumption_rate_days": 2,
        },
        "herbs": {
            "items": ["Basil", "Cilantro", "Parsley", "Mint", "Rosemary"],
            "shelf_life_range": (5, 14),
            "base_waste_prob": 0.40,  # Very high
            "price_range": (1, 5),
            "typical_consumption_rate_days": 7,
        },
        "other": {
            "items": ["Eggs", "Tofu", "Hummus", "Pickles"],
            "shelf_life_range": (14, 60),
            "base_waste_prob": 0.20,
            "price_range": (2, 8),
            "typical_consumption_rate_days": 14,
        },
    }
    
    def __init__(self, seed=42):
        self.faker = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Initialized with seed={seed}")
    
    def generate_dataset(
        self,
        n_households=1000,
        n_items_per_household=100,
        date_range_days=365,
    ) -> pd.DataFrame:
        """Generate complete dataset with 50+ features."""
        
        logger.info("="*80)
        logger.info("GENERATING ENHANCED SYNTHETIC TRAINING DATA")
        logger.info("="*80)
        logger.info(f"Households: {n_households:,}")
        logger.info(f"Items per household: {n_items_per_household}")
        logger.info(f"Expected samples: ~{n_households * n_items_per_household:,}")
        logger.info(f"Target waste rate: 25%")
        logger.info("")
        
        all_samples = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=date_range_days)
        
        for hh_id in range(n_households):
            if (hh_id + 1) % 100 == 0:
                logger.info(f"Processing household {hh_id + 1}/{n_households}...")
            
            # Assign archetype
            archetype = random.choice(list(HouseholdArchetype))
            archetype_params = self.ARCHETYPES[archetype]
            
            # Household characteristics
            household_size = random.randint(*archetype_params["household_size_range"])
            base_waste_rate = archetype_params["base_waste_rate"]
            shopping_freq = random.uniform(*archetype_params["shopping_frequency_per_week"])
            
            # Generate items
            for _ in range(n_items_per_household):
                sample = self._generate_item_sample(
                    hh_id, household_size, archetype, archetype_params,
                    base_waste_rate, start_date, end_date, date_range_days
                )
                all_samples.append(sample)
        
        df = pd.DataFrame(all_samples)
        
        # Post-processing: Add engineered features
        df = self._add_engineered_features(df)
        
        # Statistics
        waste_rate = df['is_wasted'].mean()
        logger.info("")
        logger.info("="*80)
        logger.info("DATASET STATISTICS")
        logger.info("="*80)
        logger.info(f"Total samples: {len(df):,}")
        logger.info(f"Waste rate: {waste_rate:.1%}")
        logger.info(f"Wasted items: {df['is_wasted'].sum():,}")
        logger.info(f"Features: {len(df.columns)}")
        logger.info("")
        
        # Category breakdown
        logger.info("Category waste rates:")
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            cat_waste = cat_df['is_wasted'].mean()
            logger.info(f"  {category:12s}: {len(cat_df):6,} samples, {cat_waste:5.1%} waste")
        
        return df
    
    def _generate_item_sample(
        self, hh_id, household_size, archetype, archetype_params,
        base_waste_rate, start_date, end_date, date_range_days
    ) -> Dict:
        """Generate single item with 30+ base features."""
        
        # Select category and item
        category = random.choice(list(self.FOOD_CATEGORIES.keys()))
        category_info = self.FOOD_CATEGORIES[category]
        item_name = random.choice(category_info["items"])
        
        # Item characteristics
        shelf_life = random.randint(*category_info["shelf_life_range"])
        price = round(random.uniform(*category_info["price_range"]), 2)
        typical_consumption_days = category_info["typical_consumption_rate_days"]
        
        # Purchase date (weighted toward recent)
        days_ago = int(np.random.gamma(2, date_range_days / 6))
        days_ago = min(days_ago, (end_date - start_date).days)
        purchase_date = end_date - timedelta(days=days_ago)
        
        expiration_date = purchase_date + timedelta(days=shelf_life)
        days_until_expiration = shelf_life
        days_since_purchase = days_ago
        is_expired = expiration_date < end_date
        
        # Day of week (weekend shopping more common)
        purchase_day_of_week = purchase_date.weekday()
        is_weekend_purchase = purchase_day_of_week in [5, 6]
        
        # Season
        purchase_month = purchase_date.month
        season = "winter" if purchase_month in [12, 1, 2] else \
                 "spring" if purchase_month in [3, 4, 5] else \
                 "summer" if purchase_month in [6, 7, 8] else "fall"
        
        # Waste probability calculation
        category_waste_prob = category_info["base_waste_prob"]
        
        # Archetype modifier (increase overall waste)
        waste_multiplier = 2.5  # Increased from 1.0 to achieve ~25% waste rate
        
        # Shelf life modifier
        if shelf_life < 3:
            waste_multiplier *= 1.5
        elif shelf_life < 7:
            waste_multiplier *= 1.3
        elif shelf_life > 180:
            waste_multiplier *= 0.7
        
        # Purchase timing modifier
        if is_weekend_purchase:
            waste_multiplier *= 0.9  # Better planning on weekends
        
        # Impulse buy (higher waste)
        is_impulse_buy = random.random() < archetype_params["impulse_buy_rate"]
        if is_impulse_buy:
            waste_multiplier *= 1.4
        
        # Forgotten item (very high waste)
        is_forgotten = random.random() < archetype_params["forgotten_item_rate"]
        if is_forgotten:
            waste_multiplier *= 2.0
        
        # Consumption consistency
        consistency = archetype_params["consumption_consistency"]
        if consistency < 0.6:
            waste_multiplier *= 1.2
        
        # Calculate final waste probability
        final_waste_prob = category_waste_prob * base_waste_rate * waste_multiplier
        final_waste_prob = min(final_waste_prob, 0.95)  # Cap at 95%
        
        # Determine if wasted
        is_wasted = random.random() < final_waste_prob
        
        # Days to waste (if wasted)
        if is_wasted:
            # Most waste happens before expiration
            days_to_waste = max(1, int(np.random.gamma(2, shelf_life / 3)))
            days_to_waste = min(days_to_waste, days_since_purchase)
        else:
            days_to_waste = None
        
        # Quantity and price
        quantity = random.choice([1, 1, 1, 2, 2, 3])  # Most purchases are single items
        total_price = round(price * quantity, 2)
        price_per_day = round(price / max(shelf_life, 1), 2)
        
        return {
            # Identifiers
            "household_id": hh_id,
            "archetype": archetype.value,
            "item_name": item_name,
            "category": category,
            
            # Item characteristics
            "price": price,
            "quantity": quantity,
            "total_price": total_price,
            "price_per_day": price_per_day,
            "shelf_life_days": shelf_life,
            "typical_consumption_days": typical_consumption_days,
            
            # Dates and timing
            "purchase_date": purchase_date,
            "expiration_date": expiration_date,
            "days_until_expiration": days_until_expiration,
            "days_since_purchase": days_since_purchase,
            "is_expired": int(is_expired),
            "purchase_day_of_week": purchase_day_of_week,
            "is_weekend_purchase": int(is_weekend_purchase),
            "purchase_month": purchase_month,
            "season": season,
            
            # Household features
            "household_size": household_size,
            "household_base_waste_rate": round(base_waste_rate, 3),
            
            # Behavioral features
            "is_impulse_buy": int(is_impulse_buy),
            "is_forgotten": int(is_forgotten),
            "consumption_consistency": consistency,
            
            # Target variables
            "is_wasted": int(is_wasted),
            "days_to_waste": days_to_waste,
            "waste_probability": round(final_waste_prob, 3),
        }
    
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 20+ engineered features (total 50+)."""
        
        logger.info("Engineering additional features...")
        
        # Price features
        df['price_per_household_member'] = df['price'] / df['household_size']
        df['is_expensive'] = (df['price'] > df.groupby('category')['price'].transform('median')).astype(int)
        df['price_to_shelf_life_ratio'] = df['price'] / (df['shelf_life_days'] + 1)
        
        # Time features
        df['shelf_life_to_consumption_ratio'] = df['shelf_life_days'] / (df['typical_consumption_days'] + 1)
        df['is_short_shelf_life'] = (df['shelf_life_days'] < 7).astype(int)
        df['is_long_shelf_life'] = (df['shelf_life_days'] > 90).astype(int)
        df['days_to_expiration_categorical'] = pd.cut(
            df['days_until_expiration'],
            bins=[0, 3, 7, 14, 30, 999],
            labels=['critical', 'urgent', 'soon', 'moderate', 'safe']
        ).astype(str)
        
        # Household features
        df['is_single_person'] = (df['household_size'] == 1).astype(int)
        df['is_large_family'] = (df['household_size'] >= 4).astype(int)
        df['waste_risk_score'] = (
            df['household_base_waste_rate'] * 0.4 +
            df['waste_probability'] * 0.6
        )
        
        # Category features (one-hot encoding)
        category_dummies = pd.get_dummies(df['category'], prefix='cat')
        df = pd.concat([df, category_dummies], axis=1)
        
        # Archetype features (one-hot encoding)
        archetype_dummies = pd.get_dummies(df['archetype'], prefix='arch')
        df = pd.concat([df, archetype_dummies], axis=1)
        
        # Season features (one-hot encoding)
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        
        # Interaction features
        df['price_x_waste_rate'] = df['price'] * df['household_base_waste_rate']
        df['shelf_life_x_household_size'] = df['shelf_life_days'] * df['household_size']
        df['impulse_x_forgotten'] = df['is_impulse_buy'] * df['is_forgotten']
        
        # Statistical features per household
        df['household_avg_price'] = df.groupby('household_id')['price'].transform('mean')
        df['household_avg_shelf_life'] = df.groupby('household_id')['shelf_life_days'].transform('mean')
        df['household_item_count'] = df.groupby('household_id')['household_id'].transform('count')
        
        # Statistical features per category
        df['category_avg_waste_rate'] = df.groupby('category')['is_wasted'].transform('mean')
        df['category_item_count'] = df.groupby('category')['category'].transform('count')
        
        logger.info(f"✓ Added engineered features. Total: {len(df.columns)} features")
        
        return df


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate enhanced synthetic training data (50+ features, 25% waste rate)"
    )
    
    parser.add_argument(
        "--n-households",
        type=int,
        default=1000,
        help="Number of households (default: 1000 for ~100k samples)"
    )
    parser.add_argument(
        "--n-items-per-household",
        type=int,
        default=100,
        help="Average items per household (default: 100)"
    )
    parser.add_argument(
        "--date-range-days",
        type=int,
        default=365,
        help="Historical date range in days (default: 365)"
    )
    parser.add_argument(
        "--output",
        default="data/training_enhanced_100k.parquet",
        help="Output file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    generator = EnhancedSyntheticDataGenerator(seed=args.seed)
    
    df = generator.generate_dataset(
        n_households=args.n_households,
        n_items_per_household=args.n_items_per_household,
        date_range_days=args.date_range_days,
    )
    
    # Save
    if args.output.endswith('.parquet'):
        df.to_parquet(args.output, index=False)
    elif args.output.endswith('.csv'):
        df.to_csv(args.output, index=False)
    else:
        raise ValueError(f"Unsupported format: {args.output}")
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("")
    logger.info("="*80)
    logger.info("SUCCESS!")
    logger.info("="*80)
    logger.info(f"✓ Saved {len(df):,} samples to {args.output}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")
    logger.info(f"  Features: {len(df.columns)}")
    logger.info("")
    logger.info("Sample data (first 3 rows):")
    print(df.head(3)[['item_name', 'category', 'price', 'shelf_life_days', 'household_size', 'is_wasted']].to_string())


if __name__ == "__main__":
    main()
