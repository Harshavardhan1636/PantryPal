'use server';

import type { Recipe, ShoppingListItem, PantryItem, ShoppingListAdjustment } from './types';

export async function findRecipesForAtRiskItems(items: PantryItem[]): Promise<Recipe[]> {
  const atRiskItems = items.filter(item => item.riskClass === 'High').map(item => item.name);
  if (atRiskItems.length === 0) {
    return [];
  }

  try {
    // Call the ML backend API for recipe recommendations
    const response = await fetch('http://localhost:5000/recommend_recipes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ingredients: atRiskItems,
        dietary_preferences: [],
        top_k: 10,
      }),
    });

    if (!response.ok) {
      console.warn('ML backend unavailable - returning empty results');
      return [];
    }

    const data = await response.json();
    
    // Transform backend response to Recipe format
    const recipes: Recipe[] = data.recipes.map((recipe: any, index: number) => ({
      id: recipe.id || `recipe-${index}`,
      title: recipe.title || recipe.name || 'Untitled Recipe',
      name: recipe.name || recipe.title,
      ingredients: recipe.ingredients || [],
      steps: recipe.steps || recipe.instructions || '',
      prep_time: recipe.prep_time || `${recipe.minutes || 30} min`,
      servings: recipe.servings || `${recipe.n_ingredients || 4} ingredients`,
      tags: recipe.tags || [],
      source: recipe.source || 'community',
      imageUrl: recipe.imageUrl,
      imageHint: recipe.imageHint,
    }));

    return recipes;
  } catch (error) {
    console.error('Error getting recipe recommendations:', error);
    return [];
  }
}

export async function getShoppingListSuggestions(
  pantry: PantryItem[],
  shoppingList: ShoppingListItem[]
): Promise<ShoppingListAdjustment[]> {
  if (shoppingList.length === 0) {
    return [];
  }

  // Analyze each pantry item with comprehensive metrics
  const pantryAnalysis = pantry.map(item => {
    const expiryDate = item.expiryDate || new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString();
    const daysUntilExpiry = Math.ceil(
      (new Date(expiryDate).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24)
    );
    
    const daysSincePurchase = Math.ceil(
      (new Date().getTime() - new Date(item.purchaseDate).getTime()) / (1000 * 60 * 60 * 24)
    );
    
    // Calculate daily consumption rate
    const totalLifespan = daysSincePurchase + Math.max(daysUntilExpiry, 1);
    const dailyConsumptionRate = daysSincePurchase > 0 ? item.quantity / totalLifespan : 0;
    
    // Determine item status
    const isExpiringSoon = daysUntilExpiry <= 3;
    const isHighRisk = (item.riskClass === 'High' || (item.riskScore || 0) > 0.7);
    const hasStockRemaining = item.quantity > 0;
    const isAbundant = item.quantity > (dailyConsumptionRate * 7); // More than 1 week supply
    
    return {
      itemId: item.id,
      name: item.name,
      quantity: item.quantity,
      unit: item.unit,
      daysUntilExpiry,
      daysSincePurchase,
      dailyConsumptionRate,
      riskScore: item.riskScore || 0,
      riskClass: item.riskClass,
      isExpiringSoon,
      isHighRisk,
      hasStockRemaining,
      isAbundant,
    };
  });

  // Smart suggestions based on comprehensive analysis
  const adjustments: ShoppingListAdjustment[] = shoppingList
    .map(item => {
      const pantryItem = pantryAnalysis.find(p => p.name.toLowerCase() === item.name.toLowerCase());
      
      if (!pantryItem) {
        // Item not in pantry - no suggestion needed
        return null;
      }
      
      let adjustment = 0;
      let reasoning = '';
      
      // Priority 1: High risk items expiring soon with stock remaining
      if (pantryItem.isHighRisk && pantryItem.isExpiringSoon && pantryItem.hasStockRemaining) {
        if (pantryItem.quantity >= item.quantity) {
          // Already have enough that's about to expire
          adjustment = item.quantity; // Reduce to zero (remove from list)
          reasoning = `You have ${pantryItem.quantity} ${pantryItem.unit} expiring in ${pantryItem.daysUntilExpiry} day(s) - use existing stock first`;
        } else {
          // Some stock expiring, reduce purchase amount
          adjustment = Math.floor(pantryItem.quantity);
          reasoning = `${pantryItem.quantity} ${pantryItem.unit} expiring in ${pantryItem.daysUntilExpiry} day(s) - reduce purchase and use existing stock`;
        }
      }
      // Priority 2: Abundant stock (more than 1 week supply) without high risk
      else if (pantryItem.isAbundant && !pantryItem.isHighRisk) {
        const excessQuantity = Math.floor(pantryItem.quantity - (pantryItem.dailyConsumptionRate * 3)); // Keep only 3 days buffer
        if (excessQuantity >= item.quantity) {
          adjustment = item.quantity;
          reasoning = `Already have ${pantryItem.quantity} ${pantryItem.unit} in stock (${Math.floor(pantryItem.quantity / Math.max(pantryItem.dailyConsumptionRate, 0.1))} days supply) - skip this purchase`;
        } else if (excessQuantity > 0) {
          adjustment = Math.floor(excessQuantity);
          reasoning = `${pantryItem.quantity} ${pantryItem.unit} already in stock - reduce purchase amount`;
        }
      }
      // Priority 3: High consumption rate with low/no stock
      else if (pantryItem.dailyConsumptionRate > 0 && pantryItem.quantity < (pantryItem.dailyConsumptionRate * 2)) {
        // Running low with high consumption
        const increase = Math.ceil(item.quantity * 0.5); // Increase by 50%
        adjustment = -increase; // Negative = increase
        reasoning = `High consumption (${pantryItem.dailyConsumptionRate.toFixed(1)} ${pantryItem.unit}/day), low stock - increase quantity`;
      }
      // Priority 4: Medium stock with normal consumption
      else if (pantryItem.hasStockRemaining && pantryItem.quantity >= (pantryItem.dailyConsumptionRate * 5)) {
        // Have 5+ days supply
        const reduction = Math.min(Math.ceil(item.quantity * 0.3), Math.floor(pantryItem.quantity / 2));
        if (reduction > 0) {
          adjustment = reduction;
          reasoning = `${pantryItem.quantity} ${pantryItem.unit} in stock (${Math.floor(pantryItem.quantity / Math.max(pantryItem.dailyConsumptionRate, 0.1))} days supply) - slight reduction recommended`;
        }
      }
      
      // Only return if there's an actual adjustment
      if (adjustment === 0) {
        return null;
      }
      
      return {
        itemId: item.id,
        name: item.name,
        quantityToReduce: adjustment,
        unit: item.unit,
        reason: reasoning,
      };
    })
    .filter(adj => adj !== null) as ShoppingListAdjustment[];

  return adjustments;
}

// New: Generate shopping list for next month based on usage
export async function generateMonthlyShoppingList(pantry: PantryItem[]): Promise<ShoppingListItem[]> {
  const consumptionAnalysis = pantry.map(item => {
    const daysSincePurchase = Math.ceil(
      (new Date().getTime() - new Date(item.purchaseDate).getTime()) / (1000 * 60 * 60 * 24)
    );
    const daysUntilExpiry = item.expiryDate ? Math.ceil(
      (new Date(item.expiryDate).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24)
    ) : 30;
    
    const totalShelfLife = daysSincePurchase + daysUntilExpiry;
    const consumptionRate = daysSincePurchase > 0 ? (item.quantity / totalShelfLife) : 0;
    
    return {
      name: item.name,
      category: item.category,
      unit: item.unit,
      dailyConsumption: consumptionRate,
    };
  });

  return consumptionAnalysis
    .filter(item => item.dailyConsumption > 0)
    .map((item, index) => ({
      id: `monthly-${index}`,
      name: item.name,
      quantity: Math.ceil(item.dailyConsumption * 30),
      unit: item.unit,
      checked: false,
      category: item.category,
    }));
}

// New: Analyze waste risk with proactive recommendations
export async function analyzeWasteRisk(pantry: PantryItem[]): Promise<{
  highRiskItems: PantryItem[];
  wasteValue: number;
  recommendations: string[];
}> {
  const highRiskItems = pantry.filter(item => item.riskClass === 'High');
  
  // Estimate waste value (average $5 per item)
  const averageItemCost = 5;
  const wasteValue = highRiskItems.reduce((sum, item) => sum + (item.quantity * averageItemCost), 0);
  
  const recommendations: string[] = [];
  
  // Check for items expiring soon
  const expiringTomorrow = pantry.filter(item => {
    if (!item.expiryDate) return false;
    const days = Math.ceil((new Date(item.expiryDate).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24));
    return days <= 1;
  });
  
  if (expiringTomorrow.length > 0) {
    recommendations.push(`Urgent: ${expiringTomorrow.map(i => i.name).join(', ')} expires within 24 hours!`);
  }
  
  if (highRiskItems.length > 0) {
    recommendations.push(`Cook recipes using: ${highRiskItems.slice(0, 3).map(i => i.name).join(', ')}`);
  }
  
  if (highRiskItems.length > 5) {
    recommendations.push('Consider meal planning to reduce waste');
  }
  
  if (wasteValue > 20) {
    recommendations.push(`Potential waste: $${wasteValue.toFixed(2)} - prioritize using at-risk items`);
  }
  
  return { highRiskItems, wasteValue, recommendations };
}
