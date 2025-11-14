"""
Synthetic Training Data Generator for Waste Risk Prediction.

Generates realistic pantry and waste event data for model bootstrapping.

Features:
- Realistic household behavior simulation
- Category-specific waste probabilities
- Consumption pattern modeling
- Seasonal effects
- Household size effects
- Price sensitivity

Usage:
    # Generate 100k samples
    python -m backend.ml.training.generate_synthetic_data \\
        --n-households 1000 \\
        --n-items-per-household 100 \\
        --output data/synthetic_training_data.parquet

    # With custom parameters
    python -m backend.ml.training.generate_synthetic_data \\
        --n-households 500 \\
        --n-items-per-household 200 \\
        --date-range-days 730 \\
        --output data/synthetic_2year_data.parquet \\
        --seed 42

Author: Senior SDE-3
Date: 2025-11-12
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import random

import pandas as pd
import numpy as np
from faker import Faker


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate realistic pantry and waste event data."""
    
    # Food categories with realistic characteristics
    FOOD_CATEGORIES = {
        "dairy": {
            "items": [
                "Whole Milk", "2% Milk", "Skim Milk", "Almond Milk",
                "Cheddar Cheese", "Mozzarella Cheese", "Greek Yogurt",
                "Regular Yogurt", "Butter", "Heavy Cream", "Sour Cream"
            ],
            "shelf_life_range": (5, 21),  # days
            "base_waste_prob": 0.28,
            "price_range": (2, 10),
            "storage_type": "refrigerator",
            "seasonality": {1: 0.9, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.1,
                           7: 1.1, 8: 1.1, 9: 1.0, 10: 1.0, 11: 0.9, 12: 0.9},
        },
        "produce": {
            "items": [
                "Apples", "Bananas", "Oranges", "Grapes", "Strawberries",
                "Lettuce", "Tomatoes", "Carrots", "Cucumbers", "Bell Peppers",
                "Broccoli", "Spinach", "Potatoes", "Onions", "Avocados"
            ],
            "shelf_life_range": (3, 14),
            "base_waste_prob": 0.38,  # Highest waste category
            "price_range": (1, 8),
            "storage_type": "refrigerator",
            "seasonality": {1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2,
                           7: 1.2, 8: 1.1, 9: 1.0, 10: 0.9, 11: 0.8, 12: 0.8},
        },
        "meat": {
            "items": [
                "Chicken Breast", "Ground Beef", "Pork Chops", "Salmon",
                "Tilapia", "Ground Turkey", "Bacon", "Sausage",
                "Deli Turkey", "Deli Ham"
            ],
            "shelf_life_range": (2, 7),
            "base_waste_prob": 0.22,
            "price_range": (5, 25),
            "storage_type": "refrigerator",
            "seasonality": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.1, 5: 1.1, 6: 1.2,
                           7: 1.2, 8: 1.1, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0},
        },
        "grains": {
            "items": [
                "White Rice", "Brown Rice", "Pasta", "Whole Wheat Bread",
                "White Bread", "Bagels", "Cereal", "Oatmeal",
                "Quinoa", "Couscous"
            ],
            "shelf_life_range": (14, 365),
            "base_waste_prob": 0.12,
            "price_range": (2, 12),
            "storage_type": "pantry",
            "seasonality": {i: 1.0 for i in range(1, 13)},  # No seasonality
        },
        "canned": {
            "items": [
                "Canned Beans", "Canned Tomatoes", "Canned Soup",
                "Canned Corn", "Canned Green Beans", "Canned Tuna",
                "Canned Chicken", "Tomato Sauce", "Pasta Sauce"
            ],
            "shelf_life_range": (180, 730),
            "base_waste_prob": 0.05,  # Lowest waste
            "price_range": (1, 5),
            "storage_type": "pantry",
            "seasonality": {i: 1.0 for i in range(1, 13)},
        },
        "frozen": {
            "items": [
                "Frozen Pizza", "Frozen Vegetables", "Frozen Fruit",
                "Ice Cream", "Frozen Chicken Nuggets", "Frozen Fries",
                "Frozen Meals"
            ],
            "shelf_life_range": (60, 365),
            "base_waste_prob": 0.08,
            "price_range": (3, 15),
            "storage_type": "freezer",
            "seasonality": {i: 1.0 for i in range(1, 13)},
        },
        "beverages": {
            "items": [
                "Orange Juice", "Apple Juice", "Soda", "Water Bottles",
                "Sports Drinks", "Iced Tea", "Lemonade"
            ],
            "shelf_life_range": (7, 180),
            "base_waste_prob": 0.15,
            "price_range": (2, 8),
            "storage_type": "refrigerator",
            "seasonality": {1: 0.9, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.1, 6: 1.2,
                           7: 1.2, 8: 1.2, 9: 1.1, 10: 1.0, 11: 0.9, 12: 0.9},
        },
    }
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        self.fake = Faker()
        
        logger.info(f"Initialized SyntheticDataGenerator with seed={seed}")
    
    def generate_dataset(
        self,
        n_households: int = 100,
        n_items_per_household: int = 50,
        date_range_days: int = 365,
    ) -> pd.DataFrame:
        """
        Generate complete synthetic dataset.
        
        Args:
            n_households: Number of households to simulate
            n_items_per_household: Average items per household
            date_range_days: Historical date range in days
            
        Returns:
            DataFrame with synthetic pantry entries
        """
        logger.info("="*80)
        logger.info("GENERATING SYNTHETIC TRAINING DATA")
        logger.info("="*80)
        logger.info(f"Households: {n_households}")
        logger.info(f"Items per household: {n_items_per_household}")
        logger.info(f"Date range: {date_range_days} days")
        logger.info(f"Expected samples: ~{n_households * n_items_per_household:,}")
        
        data = []
        
        for household_idx in range(n_households):
            if household_idx % 100 == 0 and household_idx > 0:
                logger.info(f"  Generated {household_idx}/{n_households} households...")
            
            # Generate household profile
            household_profile = self._generate_household_profile()
            
            # Vary items per household
            n_items = int(np.random.normal(n_items_per_household, n_items_per_household * 0.2))
            n_items = max(20, min(200, n_items))  # Clamp
            
            # Generate items for this household
            for item_idx in range(n_items):
                entry = self._generate_pantry_entry(
                    household_id=f"HH_{household_idx:05d}",
                    household_profile=household_profile,
                    date_range_days=date_range_days,
                )
                data.append(entry)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Log statistics
        logger.info("\n" + "="*80)
        logger.info("DATASET STATISTICS")
        logger.info("="*80)
        logger.info(f"Total samples: {len(df):,}")
        logger.info(f"Waste ratio: {df['was_wasted'].mean():.2%}")
        logger.info(f"Wasted items: {df['was_wasted'].sum():,}")
        logger.info(f"\nCategory distribution:")
        for category, count in df['category'].value_counts().items():
            waste_rate = df[df['category'] == category]['was_wasted'].mean()
            logger.info(f"  {category:12s}: {count:6,} samples ({waste_rate:.1%} waste)")
        logger.info(f"\nStorage type distribution:")
        for storage, count in df['storage_type'].value_counts().items():
            logger.info(f"  {storage:12s}: {count:6,} samples")
        logger.info(f"\nHousehold size distribution:")
        for size, count in df['household_size'].value_counts().sort_index().items():
            logger.info(f"  {size} members: {count:6,} households")
        logger.info("="*80)
        
        return df
    
    def _generate_household_profile(self) -> Dict[str, Any]:
        """Generate household characteristics."""
        # Household size (weighted toward smaller households)
        household_size = np.random.choice(
            [1, 2, 3, 4, 5, 6],
            p=[0.28, 0.35, 0.18, 0.12, 0.05, 0.02]  # Realistic US distribution
        )
        
        # Waste behavior profile
        # Beta distribution skewed toward lower waste rates
        household_waste_propensity = np.random.beta(2, 8)  # Mean ~0.2
        
        # Shopping/consumption patterns
        household_turnover_rate = np.random.uniform(0.4, 1.8)  # Items per day
        
        # Income level (affects purchasing behavior)
        income_level = np.random.choice(
            ["low", "medium", "high"],
            p=[0.25, 0.50, 0.25]
        )
        
        # Price sensitivity
        price_sensitivity = {
            "low": np.random.uniform(0.5, 0.8),
            "medium": np.random.uniform(0.8, 1.2),
            "high": np.random.uniform(1.2, 2.0),
        }[income_level]
        
        # Planning behavior (affects waste)
        is_planner = np.random.random() < 0.4  # 40% are good planners
        
        return {
            "household_size": household_size,
            "waste_propensity": household_waste_propensity,
            "turnover_rate": household_turnover_rate,
            "income_level": income_level,
            "price_sensitivity": price_sensitivity,
            "is_planner": is_planner,
        }
    
    def _generate_pantry_entry(
        self,
        household_id: str,
        household_profile: Dict[str, Any],
        date_range_days: int,
    ) -> Dict[str, Any]:
        """Generate single pantry entry with realistic features."""
        # Select category and item
        category = np.random.choice(list(self.FOOD_CATEGORIES.keys()))
        category_info = self.FOOD_CATEGORIES[category]
        item_name = np.random.choice(category_info["items"])
        
        # Purchase date (distributed over date range)
        days_ago = np.random.exponential(date_range_days / 3)  # More recent items
        days_ago = min(days_ago, date_range_days)
        purchase_date = datetime.now() - timedelta(days=days_ago)
        
        # Purchase month for seasonality
        purchase_month = purchase_date.month
        seasonality_factor = category_info["seasonality"][purchase_month]
        
        # Shelf life
        min_shelf, max_shelf = category_info["shelf_life_range"]
        shelf_life_days = np.random.randint(min_shelf, max_shelf + 1)
        expiry_date = purchase_date + timedelta(days=shelf_life_days)
        
        # Opened date (60% of items get opened)
        opened_date = None
        is_opened = np.random.random() < 0.6
        if is_opened:
            days_until_opened = np.random.randint(0, min(7, shelf_life_days // 2))
            opened_date = purchase_date + timedelta(days=days_until_opened)
        
        # Quantity (realistic purchase quantities)
        if category in ["produce", "meat"]:
            initial_quantity = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
        elif category in ["dairy", "beverages"]:
            initial_quantity = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        else:
            initial_quantity = np.random.choice([1, 2, 4, 6], p=[0.5, 0.3, 0.15, 0.05])
        
        # Price
        min_price, max_price = category_info["price_range"]
        base_price = np.random.uniform(min_price, max_price)
        price = base_price * initial_quantity * household_profile["price_sensitivity"]
        
        # Consumption simulation
        age_days = (datetime.now() - purchase_date).days
        
        # Base consumption rate (items per day per person)
        base_consumption_rate = household_profile["turnover_rate"] / household_profile["household_size"]
        base_consumption_rate *= seasonality_factor
        
        # Adjust for category (perishables consumed faster)
        if category == "produce":
            category_consumption_multiplier = 1.5
        elif category == "meat":
            category_consumption_multiplier = 1.3
        elif category == "dairy":
            category_consumption_multiplier = 1.2
        else:
            category_consumption_multiplier = 0.8
        
        actual_consumption_rate = base_consumption_rate * category_consumption_multiplier
        actual_consumption_rate *= np.random.uniform(0.6, 1.4)  # Random variation
        
        # Quantity consumed
        days_to_consume = min(age_days, shelf_life_days)
        quantity_consumed = actual_consumption_rate * days_to_consume * household_profile["household_size"]
        quantity_consumed = min(initial_quantity, max(0, quantity_consumed))
        
        quantity_on_hand = max(0, initial_quantity - quantity_consumed)
        quantity_consumed_ratio = quantity_consumed / initial_quantity if initial_quantity > 0 else 0
        
        # === WASTE DECISION MODEL ===
        
        # Base waste probability from category
        base_waste_prob = category_info["base_waste_prob"]
        
        # Household behavior factor
        household_factor = household_profile["waste_propensity"]
        if household_profile["is_planner"]:
            household_factor *= 0.6  # Planners waste less
        
        # Age factor (older items more likely wasted)
        age_factor = 1.0 + (age_days / shelf_life_days) * 1.5
        
        # Quantity factor (less consumed = more waste)
        quantity_factor = 1.0 + (1.0 - quantity_consumed_ratio) * 0.8
        
        # Opened factor (opened items spoil faster)
        opened_factor = 1.3 if is_opened else 1.0
        
        # Seasonality factor
        season_factor = 1.0 + (seasonality_factor - 1.0) * 0.5
        
        # Price factor (expensive items less likely wasted)
        price_factor = 1.0 / (1.0 + price / 20)
        
        # Combined waste probability
        waste_prob = base_waste_prob * household_factor * age_factor * quantity_factor
        waste_prob *= opened_factor * season_factor * price_factor
        waste_prob = np.clip(waste_prob, 0.0, 0.95)  # Max 95%
        
        # Determine if wasted
        was_wasted = np.random.random() < waste_prob
        
        # Days to waste (for wasted items)
        if was_wasted:
            # Wasted somewhere between 50% shelf life and expiry
            min_waste_day = int(shelf_life_days * 0.5)
            max_waste_day = shelf_life_days
            days_to_waste = np.random.randint(
                max(1, min_waste_day),
                max(2, max_waste_day)
            )
            days_to_waste = min(days_to_waste, age_days)  # Can't be wasted in future
        else:
            days_to_waste = None
        
        # === ENGINEERED FEATURES ===
        
        # Temporal features
        features = {
            "age_days": age_days,
            "days_to_expiry": (expiry_date - datetime.now()).days,
            "days_since_opened": (datetime.now() - opened_date).days if opened_date else 0,
            "purchase_day_of_week": purchase_date.weekday(),
            "purchase_week_of_year": purchase_date.isocalendar()[1],
        }
        
        # Quantity features
        features.update({
            "quantity_on_hand": quantity_on_hand,
            "initial_quantity": initial_quantity,
            "quantity_consumed_ratio": quantity_consumed_ratio,
        })
        
        # Consumption pattern features (simulated)
        # In real data these come from consumption logs
        features.update({
            "avg_daily_consumption_7d": actual_consumption_rate * household_profile["household_size"],
            "avg_daily_consumption_14d": actual_consumption_rate * household_profile["household_size"] * 0.95,
            "avg_daily_consumption_30d": actual_consumption_rate * household_profile["household_size"] * 0.9,
            "consumption_velocity": actual_consumption_rate * 0.1,  # Trend
            "cohort_avg_consumption": actual_consumption_rate * household_profile["household_size"] * 1.1,  # Peer comparison
        })
        
        # Household features
        features.update({
            "household_size": float(household_profile["household_size"]),
            "household_pantry_turnover_rate": household_profile["turnover_rate"],
            "household_waste_rate_historical": household_profile["waste_propensity"],
        })
        
        # Item features
        features.update({
            "storage_type": {"pantry": 0, "refrigerator": 1, "freezer": 2}[category_info["storage_type"]],
            "is_opened": 1 if is_opened else 0,
            "price_per_unit": price / initial_quantity,
            "is_bulk_purchase": 1 if initial_quantity >= 4 else 0,
            "typical_shelf_life_days": shelf_life_days,
        })
        
        # Category features (aggregates - in real data from historical stats)
        category_waste_rates = {
            "produce": 0.38, "dairy": 0.28, "meat": 0.22,
            "beverages": 0.15, "grains": 0.12, "frozen": 0.08, "canned": 0.05
        }
        features.update({
            "category_waste_rate": category_waste_rates.get(category, 0.20),
            "category_turnover_rate": {
                "produce": 1.5, "meat": 1.3, "dairy": 1.2,
                "beverages": 1.0, "grains": 0.5, "frozen": 0.4, "canned": 0.3
            }.get(category, 1.0),
            "category_seasonality_score": seasonality_factor,
        })
        
        # Interaction features
        features.update({
            "family_size_x_quantity": household_profile["household_size"] * initial_quantity,
            "price_x_quantity": price,
            "age_x_expiry_ratio": age_days / shelf_life_days if shelf_life_days > 0 else 0,
        })
        
        # External features
        is_holiday_season = purchase_month in [11, 12]
        features.update({
            "is_holiday_season": 1 if is_holiday_season else 0,
            "day_of_week_encoded": purchase_date.weekday() / 7.0,
            "promotion_flag": 1 if np.random.random() < 0.15 else 0,  # 15% on promotion
            "recommendation_response_rate": np.random.beta(3, 2),  # User engagement
        })
        
        # Return complete entry
        return {
            # Identifiers
            "entry_id": self.fake.uuid4(),
            "household_id": household_id,
            
            # Item info
            "item_name": item_name,
            "category": category,
            "storage_type": category_info["storage_type"],
            
            # Dates
            "purchase_date": purchase_date,
            "expiry_date": expiry_date,
            "opened_date": opened_date,
            
            # Quantities
            "initial_quantity": initial_quantity,
            "quantity_on_hand": quantity_on_hand,
            "quantity_consumed_ratio": quantity_consumed_ratio,
            
            # Price
            "price": price,
            
            # Labels
            "was_wasted": was_wasted,
            "days_to_waste": days_to_waste,
            "shelf_life_days": shelf_life_days,
            
            # All engineered features
            **features
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for waste risk prediction"
    )
    
    parser.add_argument(
        "--n-households",
        type=int,
        default=100,
        help="Number of households to simulate"
    )
    parser.add_argument(
        "--n-items-per-household",
        type=int,
        default=50,
        help="Average items per household"
    )
    parser.add_argument(
        "--date-range-days",
        type=int,
        default=365,
        help="Historical date range in days"
    )
    parser.add_argument(
        "--output",
        default="synthetic_training_data.parquet",
        help="Output file path (parquet or csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Generate data
    generator = SyntheticDataGenerator(seed=args.seed)
    
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
        raise ValueError(f"Unsupported output format: {args.output}")
    
    logger.info(f"\n✓ Saved {len(df):,} samples to {args.output}")
    logger.info(f"  File size: {Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")
    
    # Print sample
    logger.info("\n" + "="*80)
    logger.info("SAMPLE DATA (first 3 rows)")
    logger.info("="*80)
    print(df.head(3).to_string())


if __name__ == "__main__":
    from pathlib import Path
    main()
