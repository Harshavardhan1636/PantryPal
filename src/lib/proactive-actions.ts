'use server';

/**
 * Proactive ML-Powered Actions
 * This module implements intelligent, automated behaviors using ML models
 * to transform the app from reactive (user-triggered) to proactive (automatic)
 */

import type { PantryItem, Recipe, ShoppingListItem } from './types';
import { findRecipesForAtRiskItems, analyzeWasteRisk, getShoppingListSuggestions } from './actions';

export type ProactiveAlert = {
  id: string;
  type: 'waste_warning' | 'recipe_suggestion' | 'shopping_optimization' | 'stock_alert';
  priority: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  message: string;
  action?: {
    label: string;
    data: any;
  };
  timestamp: Date;
  dismissed: boolean;
};

/**
 * Proactive Waste Prevention System (Enhanced)
 * Automatically detects waste risks and generates actionable alerts
 * with predictive intelligence and smart clustering
 */
export async function proactiveWasteMonitoring(
  pantryItems: PantryItem[]
): Promise<ProactiveAlert[]> {
  const alerts: ProactiveAlert[] = [];
  const now = new Date();

  // Detect IMMEDIATE urgency (12 hours - more aggressive)
  const immediateItems = pantryItems.filter(item => {
    if (!item.expiryDate) return false;
    const hoursUntilExpiry = (new Date(item.expiryDate).getTime() - now.getTime()) / (1000 * 60 * 60);
    return hoursUntilExpiry <= 12 && hoursUntilExpiry > 0;
  });

  if (immediateItems.length > 0) {
    alerts.push({
      id: `immediate-expiry-${Date.now()}`,
      type: 'waste_warning',
      priority: 'critical',
      title: 'URGENT: Items Expiring Today',
      message: `${immediateItems.length} item(s) expire within 12 hours: ${immediateItems.map(i => i.name).join(', ')}. Take immediate action!`,
      action: {
        label: 'Generate Meal Plan',
        data: immediateItems,
      },
      timestamp: now,
      dismissed: false,
    });
  }

  // Detect critical 24-hour window
  const criticalItems = pantryItems.filter(item => {
    if (!item.expiryDate) return false;
    const hoursUntilExpiry = (new Date(item.expiryDate).getTime() - now.getTime()) / (1000 * 60 * 60);
    return hoursUntilExpiry > 12 && hoursUntilExpiry <= 24;
  });

  if (criticalItems.length > 0) {
    alerts.push({
      id: `critical-expiry-${Date.now()}`,
      type: 'waste_warning',
      priority: 'critical',
      title: 'Critical: Items Expiring Today',
      message: `${criticalItems.length} item(s) expire within 24 hours: ${criticalItems.map(i => i.name).join(', ')}`,
      action: {
        label: 'View Recipes',
        data: criticalItems,
      },
      timestamp: now,
      dismissed: false,
    });
  }

  // Smart clustering: detect high-risk groups expiring together (enhanced)
  const next2Days = pantryItems.filter(item => {
    if (!item.expiryDate) return false;
    const daysUntilExpiry = Math.ceil((new Date(item.expiryDate).getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
    return daysUntilExpiry > 1 && daysUntilExpiry <= 2;
  });

  if (next2Days.length >= 2) { // Lowered threshold for faster alerts
    alerts.push({
      id: `cluster-expiry-${Date.now()}`,
      type: 'waste_warning',
      priority: 'high',
      title: `Multiple Items Expiring in 2 Days`,
      message: `${next2Days.length} items need attention: ${next2Days.slice(0, 3).map(i => i.name).join(', ')}${next2Days.length > 3 ? '...' : ''}`,
      action: {
        label: 'Generate Meal Plan',
        data: next2Days,
      },
      timestamp: now,
      dismissed: false,
    });
  }

  // Predictive alert: items expiring in 3-4 days (proactive warning)
  const upcoming = pantryItems.filter(item => {
    if (!item.expiryDate) return false;
    const daysUntilExpiry = Math.ceil((new Date(item.expiryDate).getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
    return daysUntilExpiry > 2 && daysUntilExpiry <= 4;
  });

  if (upcoming.length >= 3) {
    alerts.push({
      id: `upcoming-expiry-${Date.now()}`,
      type: 'waste_warning',
      priority: 'medium',
      title: `Plan Ahead: ${upcoming.length} Items Expiring Soon`,
      message: `Consider using these items in the next few days: ${upcoming.slice(0, 3).map(i => i.name).join(', ')}${upcoming.length > 3 ? '...' : ''}`,
      action: {
        label: 'View Recipes',
        data: upcoming,
      },
      timestamp: now,
      dismissed: false,
    });
  }

  return alerts;
}

/**
 * Proactive Recipe Recommendation System (Enhanced)
 * Automatically suggests recipes with smart timing and better triggers
 */
export async function proactiveRecipeRecommendations(
  pantryItems: PantryItem[],
  lastRecommendationTime?: Date
): Promise<{ recipes: Recipe[]; alert: ProactiveAlert | null }> {
  // Dynamic timing based on risk level
  const hasHighRisk = pantryItems.some(item => item.riskClass === 'High');
  const hasMediumRisk = pantryItems.some(item => item.riskClass === 'Medium');
  const hoursInterval = hasHighRisk ? 3 : (hasMediumRisk ? 5 : 6);
  
  const hoursSinceLastRec = lastRecommendationTime 
    ? (Date.now() - lastRecommendationTime.getTime()) / (1000 * 60 * 60)
    : 999;

  if (hoursSinceLastRec < hoursInterval) {
    return { recipes: [], alert: null };
  }

  // Enhanced: Include medium-risk items expiring soon
  const atRiskItems = pantryItems.filter(item => {
    if (!item.expiryDate) return false;
    const daysUntilExpiry = Math.ceil(
      (new Date(item.expiryDate).getTime() - Date.now()) / (1000 * 60 * 60 * 24)
    );
    return item.riskClass === 'High' || 
           (item.riskScore && item.riskScore > 0.6) ||
           (item.riskClass === 'Medium' && daysUntilExpiry <= 3);
  });

  if (atRiskItems.length === 0) {
    return { recipes: [], alert: null };
  }

  // Get recipe recommendations
  const recipes = await findRecipesForAtRiskItems(atRiskItems);

  if (recipes.length > 0) {
    // Smart priority based on urgency
    const criticalCount = atRiskItems.filter(item => item.riskClass === 'High').length;
    const priority = criticalCount >= 2 ? 'high' : 'medium';
    
    const alert: ProactiveAlert = {
      id: `recipe-rec-${Date.now()}`,
      type: 'recipe_suggestion',
      priority,
      title: `${recipes.length} Smart Recipe${recipes.length > 1 ? 's' : ''} Found`,
      message: `Using ${atRiskItems.length} at-risk item${atRiskItems.length > 1 ? 's' : ''}: ${atRiskItems.slice(0, 3).map(i => i.name).join(', ')}${atRiskItems.length > 3 ? '...' : ''}`,
      action: {
        label: 'View Recipes',
        data: recipes,
      },
      timestamp: new Date(),
      dismissed: false,
    };

    return { recipes, alert };
  }

  return { recipes: [], alert: null };
}

/**
 * Proactive Shopping List Optimizer (Enhanced)
 * Automatically optimizes shopping list with smart analysis
 */
export async function proactiveShoppingOptimization(
  pantryItems: PantryItem[],
  shoppingList: ShoppingListItem[]
): Promise<ProactiveAlert | null> {
  if (shoppingList.length === 0) {
    return null;
  }

  const suggestions = await getShoppingListSuggestions(pantryItems, shoppingList);

  if (suggestions.length > 0) {
    const reductions = suggestions.filter(s => s.quantityToReduce > 0).length;
    const increases = suggestions.filter(s => s.quantityToReduce < 0).length;
    
    // Calculate potential savings
    const savings = suggestions
      .filter(s => s.quantityToReduce > 0)
      .reduce((sum, s) => sum + (s.quantityToReduce * 2), 0); // Estimate $2 per unit

    // Smart priority based on impact
    const priority = savings > 10 || reductions > 3 ? 'high' : 'medium';
    
    const message = savings > 0
      ? `Optimize ${reductions + increases} item(s) - potential savings: $${savings.toFixed(2)}`
      : `${reductions} item(s) to reduce, ${increases} item(s) to increase`;

    return {
      id: `shopping-opt-${Date.now()}`,
      type: 'shopping_optimization',
      priority,
      title: 'Smart Shopping Optimization',
      message,
      action: {
        label: 'Review Suggestions',
        data: suggestions,
      },
      timestamp: new Date(),
      dismissed: false,
    };
  }

  return null;
}

/**
 * Proactive Stock Level Monitoring
 * Detects low stock situations and suggests restocking
 */
export async function proactiveStockMonitoring(
  pantryItems: PantryItem[]
): Promise<ProactiveAlert[]> {
  const alerts: ProactiveAlert[] = [];
  const now = new Date();

  // Analyze consumption patterns
  const lowStockItems = pantryItems.filter(item => {
    const daysSincePurchase = Math.ceil(
      (now.getTime() - new Date(item.purchaseDate).getTime()) / (1000 * 60 * 60 * 24)
    );

    const daysUntilExpiry = item.expiryDate 
      ? Math.ceil((new Date(item.expiryDate).getTime() - now.getTime()) / (1000 * 60 * 60 * 24))
      : 30;

    const totalLifespan = daysSincePurchase + daysUntilExpiry;
    const dailyConsumption = daysSincePurchase > 0 ? item.quantity / totalLifespan : 0;
    const daysSupplyRemaining = dailyConsumption > 0 ? item.quantity / dailyConsumption : 999;

    // Low stock: less than 2 days supply
    return daysSupplyRemaining < 2 && daysSupplyRemaining > 0 && item.riskClass !== 'High';
  });

  if (lowStockItems.length > 0) {
    alerts.push({
      id: `low-stock-${Date.now()}`,
      type: 'stock_alert',
      priority: 'medium',
      title: 'Low Stock Alert',
      message: `${lowStockItems.length} item(s) running low: ${lowStockItems.map(i => i.name).slice(0, 3).join(', ')}`,
      action: {
        label: 'Add to Shopping List',
        data: lowStockItems,
      },
      timestamp: now,
      dismissed: false,
    });
  }

  // Detect items running out before expiry (good consumption)
  const efficientItems = pantryItems.filter(item => {
    if (!item.expiryDate) return false;
    
    const daysUntilExpiry = Math.ceil(
      (new Date(item.expiryDate).getTime() - now.getTime()) / (1000 * 60 * 60 * 24)
    );

    // Will run out 3+ days before expiry
    return daysUntilExpiry > 3 && item.quantity < 1 && item.riskClass === 'Low';
  });

  if (efficientItems.length >= 3) {
    alerts.push({
      id: `efficient-usage-${Date.now()}`,
      type: 'stock_alert',
      priority: 'low',
      title: 'Efficient Pantry Management',
      message: `Great job! ${efficientItems.length} items consumed efficiently with minimal waste`,
      timestamp: now,
      dismissed: false,
    });
  }

  return alerts;
}

/**
 * Master Proactive Intelligence Engine
 * Runs all proactive checks and returns prioritized alerts
 */
export async function runProactiveIntelligence(
  pantryItems: PantryItem[],
  shoppingList: ShoppingListItem[],
  lastRecipeTime?: Date
): Promise<{
  alerts: ProactiveAlert[];
  recipes: Recipe[];
  shouldNotify: boolean;
}> {
  const allAlerts: ProactiveAlert[] = [];

  // Run all proactive checks in parallel
  const [wasteAlerts, recipeResult, shoppingAlert, stockAlerts] = await Promise.all([
    proactiveWasteMonitoring(pantryItems),
    proactiveRecipeRecommendations(pantryItems, lastRecipeTime),
    proactiveShoppingOptimization(pantryItems, shoppingList),
    proactiveStockMonitoring(pantryItems),
  ]);

  allAlerts.push(...wasteAlerts);
  if (recipeResult.alert) allAlerts.push(recipeResult.alert);
  if (shoppingAlert) allAlerts.push(shoppingAlert);
  allAlerts.push(...stockAlerts);

  // Sort by priority
  const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
  allAlerts.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

  // Should notify if there are critical or high priority alerts
  const shouldNotify = allAlerts.some(a => a.priority === 'critical' || a.priority === 'high');

  return {
    alerts: allAlerts,
    recipes: recipeResult.recipes,
    shouldNotify,
  };
}

/**
 * Auto-add items to shopping list based on consumption patterns
 */
export async function autoGenerateShoppingList(
  pantryItems: PantryItem[]
): Promise<ShoppingListItem[]> {
  const suggestions: ShoppingListItem[] = [];
  const now = new Date();

  pantryItems.forEach(item => {
    const daysSincePurchase = Math.ceil(
      (now.getTime() - new Date(item.purchaseDate).getTime()) / (1000 * 60 * 60 * 24)
    );

    const daysUntilExpiry = item.expiryDate 
      ? Math.ceil((new Date(item.expiryDate).getTime() - now.getTime()) / (1000 * 60 * 60 * 24))
      : 30;

    const totalLifespan = daysSincePurchase + daysUntilExpiry;
    const dailyConsumption = daysSincePurchase > 0 ? item.quantity / totalLifespan : 0;
    const daysSupplyRemaining = dailyConsumption > 0 ? item.quantity / dailyConsumption : 999;

    // Auto-suggest restocking if less than 2 days supply and not high risk
    if (daysSupplyRemaining < 2 && item.riskClass !== 'High') {
      const suggestedQuantity = Math.ceil(dailyConsumption * 7); // 1 week supply

      suggestions.push({
        id: `auto-${item.id}-${Date.now()}`,
        name: item.name,
        quantity: Math.max(suggestedQuantity, 1),
        unit: item.unit,
      });
    }
  });

  return suggestions;
}
