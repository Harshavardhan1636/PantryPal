"""
Standalone Synthetic Data Generator - No Database Dependencies

Usage:
    python generate_data_standalone.py --n-households 100 --output data/synthetic_test.parquet
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import random

import pandas as pd
import numpy as np
from faker import Faker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import the generator class directly
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Copy the generator code inline to avoid import issues
class SyntheticDataGenerator:
    """Generate realistic pantry and waste event data."""
    
    FOOD_CATEGORIES = {
        "dairy": {
            "items": ["Whole Milk", "Cheddar Cheese", "Greek Yogurt", "Butter"],
            "shelf_life_range": (5, 21),
            "base_waste_prob": 0.28,
            "price_range": (2, 10),
        },
        "produce": {
            "items": ["Apples", "Bananas", "Lettuce", "Tomatoes", "Carrots"],
            "shelf_life_range": (3, 14),
            "base_waste_prob": 0.38,
            "price_range": (1, 8),
        },
        "meat": {
            "items": ["Chicken Breast", "Ground Beef", "Salmon"],
            "shelf_life_range": (2, 7),
            "base_waste_prob": 0.22,
            "price_range": (5, 25),
        },
        "grains": {
            "items": ["Rice", "Pasta", "Bread", "Cereal"],
            "shelf_life_range": (14, 365),
            "base_waste_prob": 0.12,
            "price_range": (2, 12),
        },
    }
    
    def __init__(self, seed=42):
        self.faker = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_dataset(self, n_households=100, n_items_per_household=50, date_range_days=365):
        """Generate complete synthetic dataset."""
        logger.info(f"Generating synthetic data...")
        logger.info(f"  Households: {n_households}")
        logger.info(f"  Items per household: {n_items_per_household}")
        logger.info(f"  Total samples: ~{n_households * n_items_per_household:,}")
        
        all_samples = []
        
        for hh_id in range(n_households):
            if hh_id % 10 == 0:
                logger.info(f"  Processing household {hh_id}/{n_households}...")
            
            # Household characteristics
            household_size = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.35, 0.25, 0.15, 0.10])
            base_waste_rate = np.random.beta(2, 5)  # Skewed toward lower waste
            
            # Generate items for this household
            for _ in range(n_items_per_household):
                sample = self._generate_item_sample(hh_id, household_size, base_waste_rate, date_range_days)
                all_samples.append(sample)
        
        df = pd.DataFrame(all_samples)
        
        # Print statistics
        waste_rate = df['is_wasted'].mean()
        logger.info(f"\n✓ Generated {len(df):,} samples")
        logger.info(f"  Waste rate: {waste_rate:.1%}")
        logger.info(f"  Unique households: {df['household_id'].nunique()}")
        logger.info(f"  Date range: {df['purchase_date'].min()} to {df['purchase_date'].max()}")
        
        return df
    
    def _generate_item_sample(self, hh_id, household_size, base_waste_rate, date_range_days):
        """Generate a single item sample."""
        # Select category and item
        category = random.choice(list(self.FOOD_CATEGORIES.keys()))
        category_info = self.FOOD_CATEGORIES[category]
        item_name = random.choice(category_info["items"])
        
        # Item characteristics
        shelf_life = random.randint(*category_info["shelf_life_range"])
        price = random.uniform(*category_info["price_range"])
        
        # Dates
        purchase_date = datetime.now() - timedelta(days=random.randint(0, date_range_days))
        expiration_date = purchase_date + timedelta(days=shelf_life)
        
        # Waste prediction
        category_waste_prob = category_info["base_waste_prob"]
        household_modifier = base_waste_rate
        
        # Days until expiration (at time of purchase)
        days_until_expiration = shelf_life
        
        # Adjust waste probability based on shelf life
        if days_until_expiration < 3:
            waste_multiplier = 1.5
        elif days_until_expiration < 7:
            waste_multiplier = 1.2
        else:
            waste_multiplier = 1.0
        
        final_waste_prob = category_waste_prob * household_modifier * waste_multiplier
        final_waste_prob = min(final_waste_prob, 0.9)  # Cap at 90%
        
        is_wasted = random.random() < final_waste_prob
        
        # Days to waste (if wasted)
        if is_wasted:
            # Items typically wasted before expiration or shortly after
            days_to_waste = max(1, int(np.random.gamma(2, shelf_life / 4)))
        else:
            days_to_waste = None
        
        # Features
        return {
            # Identifiers
            "household_id": hh_id,
            "item_name": item_name,
            "category": category,
            
            # Item characteristics
            "price": round(price, 2),
            "shelf_life_days": shelf_life,
            "storage_type": category_info.get("storage_type", "pantry"),
            
            # Dates
            "purchase_date": purchase_date,
            "expiration_date": expiration_date,
            "days_until_expiration": days_until_expiration,
            
            # Household features
            "household_size": household_size,
            "household_waste_rate": round(base_waste_rate, 3),
            
            # Target variables
            "is_wasted": int(is_wasted),
            "days_to_waste": days_to_waste,
            "waste_probability": round(final_waste_prob, 3),
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
        default="data/synthetic_training_data.parquet",
        help="Output file path (parquet or csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"\n✓ Saved {len(df):,} samples to {args.output}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")
    
    # Print sample
    logger.info("\n" + "="*80)
    logger.info("SAMPLE DATA (first 3 rows)")
    logger.info("="*80)
    print(df.head(3).to_string())
    
    logger.info("\n" + "="*80)
    logger.info("COLUMN SUMMARY")
    logger.info("="*80)
    print(df.describe())


if __name__ == "__main__":
    main()
