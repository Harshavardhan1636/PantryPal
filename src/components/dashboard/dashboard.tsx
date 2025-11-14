'use client';

import { useState, useEffect } from 'react';
import type { PantryItem, Recipe, ShoppingListItem, ShoppingListAdjustment } from '@/lib/types';
import { PantryInventory } from '@/components/pantry/pantry-inventory';
import { AtRiskItems } from '@/components/recommendations/at-risk-items';
import { ShoppingList } from '@/components/shopping/shopping-list';
import { useToast } from '@/hooks/use-toast';
import { useUser } from '@/hooks/use-user';
import { useLocalStorage } from '@/hooks/use-local-storage';
import { STORAGE_KEYS } from '@/lib/local-storage';
import { analyzeWasteRisk, findRecipesForAtRiskItems } from '@/lib/actions';
import { runProactiveIntelligence, autoGenerateShoppingList, type ProactiveAlert } from '@/lib/proactive-actions';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertTriangle, DollarSign, Bell, X, Sparkles, TrendingUp } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';

export function Dashboard() {
  const { user } = useUser();
  const {
    data: pantryItems,
    loading: pantryLoading,
    addItem: addPantryItemInternal,
    removeItem: removePantryItemInternal,
    updateItem: updatePantryItemInternal,
  } = useLocalStorage<PantryItem>(STORAGE_KEYS.PANTRY_ITEMS);

  const [shoppingList, setShoppingList] = useState<ShoppingListItem[]>([]);
  const [recipes, setRecipes] = useState<Recipe[] | null>(null);
  const [adjustments, setAdjustments] = useState<ShoppingListAdjustment[] | null>(null);
  const [isRecipeLoading, setIsRecipeLoading] = useState(false);
  const [isShoppingListLoading, setIsShoppingListLoading] = useState(false);
  const [wasteAnalysis, setWasteAnalysis] = useState<{
    highRiskItems: PantryItem[];
    wasteValue: number;
    recommendations: string[];
  } | null>(null);
  const [proactiveAlerts, setProactiveAlerts] = useState<ProactiveAlert[]>([]);
  const [lastRecipeRecommendation, setLastRecipeRecommendation] = useState<Date>();
  const [isProactivePanelOpen, setIsProactivePanelOpen] = useState(false);
  const [hasOpenedOnce, setHasOpenedOnce] = useState(false);
  const { toast } = useToast();

  // Recalculate risk scores for all items periodically (enhanced frequency)
  useEffect(() => {
    if (!pantryItems || pantryItems.length === 0 || pantryLoading) return;
    
    const recalculateRisks = async () => {
      const updates: Array<{ id: string; riskClass: 'Low' | 'Medium' | 'High'; riskScore: number }> = [];
      
      for (const item of pantryItems) {
        if (!item.expiryDate) continue;
        
        const { riskClass, riskScore } = await calculateWasteRisk({
          name: item.name,
          category: item.category,
          quantity: item.quantity,
          unit: item.unit,
          purchaseDate: item.purchaseDate,
          expiryDate: item.expiryDate,
        });
        
        // Check if risk changed significantly
        const significantChange = item.riskClass !== riskClass || 
          Math.abs((item.riskScore || 0) - riskScore) > 0.15;
        
        if (significantChange) {
          updates.push({ id: item.id, riskClass, riskScore });
        }
      }
      
      // Apply all updates
      updates.forEach(({ id, riskClass, riskScore }) => {
        updatePantryItemInternal(id, { riskClass, riskScore } as Partial<PantryItem>);
      });
      
      if (updates.length > 0) {
        console.log(`Updated risk scores for ${updates.length} items`);
        
        // Only show toast for significant updates
        const criticalUpdates = updates.filter(u => u.riskClass === 'High').length;
        if (criticalUpdates > 0) {
          toast({
            title: "Risk Scores Updated",
            description: `${criticalUpdates} item(s) now at high risk`,
            variant: 'destructive',
            duration: 5000,
          });
        } else {
          toast({
            title: "Risk Scores Updated",
            description: `Auto-refreshed ${updates.length} ${updates.length === 1 ? 'item' : 'items'}`,
            duration: 3000,
          });
        }
      }
    };
    
    // Run once after items load, then every 3 minutes (faster updates)
    const timer = setTimeout(recalculateRisks, 2000);
    const interval = setInterval(recalculateRisks, 3 * 60 * 1000);
    
    return () => {
      clearTimeout(timer);
      clearInterval(interval);
    };
  }, [pantryItems?.length, pantryLoading]); // Re-run when pantry size changes

  // Automated waste analysis - runs when pantry changes
  useEffect(() => {
    if (pantryItems && pantryItems.length > 0) {
      analyzeWasteRisk(pantryItems).then(setWasteAnalysis);
    } else {
      setWasteAnalysis(null);
    }
  }, [pantryItems]);

  // Auto-open proactive panel only on first login/load
  useEffect(() => {
    const hasShownPanel = sessionStorage.getItem('hasShownProactivePanelThisSession');
    if (!hasShownPanel && pantryItems && pantryItems.length > 0 && !hasOpenedOnce) {
      // Delay opening to ensure UI is fully loaded
      const timer = setTimeout(() => {
        setIsProactivePanelOpen(true);
        setHasOpenedOnce(true);
        sessionStorage.setItem('hasShownProactivePanelThisSession', 'true');
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [pantryItems, hasOpenedOnce]);

  // Proactive Intelligence Engine - runs automatically every 5 minutes (increased frequency)
  useEffect(() => {
    if (!pantryItems || pantryItems.length === 0) return;

    const runProactiveChecks = async () => {
      const result = await runProactiveIntelligence(
        pantryItems,
        shoppingList,
        lastRecipeRecommendation
      );

      // Update alerts only if changed (prevent unnecessary re-renders)
      setProactiveAlerts(prev => {
        const hasChanges = JSON.stringify(prev) !== JSON.stringify(result.alerts);
        return hasChanges ? result.alerts : prev;
      });

      // Auto-load recipes if found and panel is not already showing recipes
      if (result.recipes.length > 0 && !recipes) {
        setRecipes(result.recipes);
        setLastRecipeRecommendation(new Date());
      }

      // Show notification for critical/high priority alerts (only once per alert)
      if (result.shouldNotify) {
        const criticalAlert = result.alerts.find(a => a.priority === 'critical' && !a.dismissed);
        if (criticalAlert) {
          toast({
            title: criticalAlert.title,
            description: criticalAlert.message,
            variant: 'destructive',
            duration: 10000, // Longer duration for critical
          });
          // Don't auto-open panel - let user open it manually
        } else {
          const highAlert = result.alerts.find(a => a.priority === 'high' && !a.dismissed);
          if (highAlert) {
            toast({
              title: highAlert.title,
              description: highAlert.message,
              duration: 7000,
            });
          }
        }
      }
    };

    // Run immediately on load
    runProactiveChecks();

    // Then run every 5 minutes (increased from 10 minutes)
    const interval = setInterval(runProactiveChecks, 5 * 60 * 1000);

    return () => clearInterval(interval);
  }, [pantryItems?.length, shoppingList.length, recipes]);

  // Auto-generate shopping list suggestions when pantry is low (enhanced)
  useEffect(() => {
    if (!pantryItems || pantryItems.length === 0) return;

    const autoSuggestItems = async () => {
      const suggestions = await autoGenerateShoppingList(pantryItems);
      
      if (suggestions.length > 0) {
        // Smart merge: only add items not already in list
        const existingNames = shoppingList.map(item => item.name.toLowerCase());
        const newSuggestions = suggestions.filter(
          s => !existingNames.includes(s.name.toLowerCase())
        );
        
        if (newSuggestions.length > 0) {
          toast({
            title: 'Smart Shopping Suggestions',
            description: `Added ${newSuggestions.length} items based on your consumption patterns`,
            duration: 6000,
          });
          setShoppingList(prev => [...prev, ...newSuggestions]);
        }
      }
    };

    // Run after 3 seconds delay (faster response)
    const timer = setTimeout(autoSuggestItems, 3000);
    return () => clearTimeout(timer);
  }, [pantryItems?.length]);

  // Helper to calculate waste risk using ML backend via Next.js API
  const calculateWasteRisk = async (item: Omit<PantryItem, 'id' | 'riskClass' | 'riskScore'>) => {
    const expiryDate = item.expiryDate || new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString();
    
    try {
      const response = await fetch('/api/predict-waste-risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          item_name: item.name,
          category: item.category,
          quantity: item.quantity,
          days_until_expiry: Math.ceil(
            (new Date(expiryDate).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24)
          ),
          purchase_date: item.purchaseDate,
          storage_location: 'refrigerator',
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        return {
          riskClass: data.risk_class as 'Low' | 'Medium' | 'High',
          riskScore: data.risk_probability,
        };
      }
    } catch (error) {
      console.error('Failed to calculate waste risk - using fallback calculation:', error);
      toast({
        title: "Using Basic Risk Calculation",
        description: "ML backend unavailable. Using fallback algorithm.",
        variant: "default",
      });
    }
    
    // Fallback to basic calculation if API fails
    const daysUntilExpiry = Math.ceil(
      (new Date(expiryDate).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24)
    );
    
    let riskScore = 0.1;
    let riskClass: 'Low' | 'Medium' | 'High' = 'Low';
    
    if (daysUntilExpiry <= 2) {
      riskScore = 0.9;
      riskClass = 'High';
    } else if (daysUntilExpiry <= 5) {
      riskScore = 0.6;
      riskClass = 'Medium';
    }
    
    return { riskClass, riskScore };
  };

  const addPantryItem = async (item: Omit<PantryItem, 'id' | 'riskClass' | 'riskScore'>) => {
    // Show optimistic loading
    toast({
      title: "Adding Item",
      description: `Analyzing ${item.name}...`,
      duration: 2000,
    });

    const { riskClass, riskScore } = await calculateWasteRisk(item);
    
    const newItemWithDefaults = {
      ...item,
      riskClass,
      riskScore,
      householdId: user?.id,
    };
    const addedItem = addPantryItemInternal(newItemWithDefaults);
    
    // Smart notification based on risk
    if (riskClass === 'High') {
      toast({
        title: "Item Added - High Risk Detected",
        description: `${item.name} expires soon. Consider using it quickly!`,
        variant: 'destructive',
        duration: 6000,
      });
    } else {
      toast({
        title: "Item Added Successfully",
        description: `${item.name} is in your pantry (${riskClass} risk)`,
        duration: 3000,
      });
    }

    // Trigger immediate proactive check for the new item
    setTimeout(() => {
      if ((riskClass === 'High' || riskScore > 0.7) && addedItem) {
        findRecipesForAtRiskItems([addedItem])
          .then(recipes => {
            if (recipes.length > 0) {
              setRecipes(recipes);
              toast({
                title: "Recipe Suggestions Ready",
                description: `Found ${recipes.length} recipes using ${item.name}`,
                duration: 5000,
              });
            }
          });
      }
    }, 1000);
  };
  
  const removePantryItem = (id: string) => {
    const item = pantryItems?.find((i: PantryItem) => i.id === id);
    removePantryItemInternal(id);
    if (item) {
        toast({
            title: "Item Removed",
            description: `${item.name} has been removed from your pantry.`,
        });
    }
  };
  
  const addShoppingListItem = (item: Omit<ShoppingListItem, 'id'>) => {
    const newItem: ShoppingListItem = {
      ...item,
      id: new Date().getTime().toString(),
    };
    setShoppingList(prev => [newItem, ...prev]);
  };
  
  const removeShoppingListItem = (id: string) => {
    setShoppingList(prev => prev.filter(item => item.id !== id));
  };

  const dismissAlert = (alertId: string) => {
    setProactiveAlerts(prev => prev.filter(a => a.id !== alertId));
  };

  const handleAlertAction = async (alert: ProactiveAlert) => {
    if (alert.type === 'recipe_suggestion' && alert.action) {
      // Load recipes from alert data
      setRecipes(alert.action.data);
      setLastRecipeRecommendation(new Date());
      setIsProactivePanelOpen(false);
      toast({
        title: 'Recipes Loaded',
        description: `${alert.action.data.length} recipes ready to view`,
      });
    } else if (alert.type === 'stock_alert' && alert.action) {
      // Add low stock items to shopping list
      const items = alert.action.data as PantryItem[];
      const newItems = items.map(item => ({
        id: `auto-${item.id}-${Date.now()}`,
        name: item.name,
        quantity: 2,
        unit: item.unit,
      }));
      setShoppingList(prev => [...prev, ...newItems]);
      setIsProactivePanelOpen(false);
      toast({
        title: 'Added to Shopping List',
        description: `${newItems.length} items added`,
      });
    } else if (alert.type === 'waste_warning' && alert.action) {
      // Generate recipes for at-risk items
      const items = alert.action.data as PantryItem[];
      
      if (alert.action.label === 'Generate Meal Plan') {
        // Generate comprehensive meal plan
        setIsRecipeLoading(true);
        try {
          const recipes = await findRecipesForAtRiskItems(items);
          setRecipes(recipes);
          setLastRecipeRecommendation(new Date());
          setIsProactivePanelOpen(false);
          toast({
            title: 'Meal Plan Generated',
            description: `${recipes.length} recipes created using ${items.length} at-risk items`,
            duration: 5000,
          });
        } catch (error) {
          toast({
            title: 'Error',
            description: 'Failed to generate meal plan',
            variant: 'destructive',
          });
        } finally {
          setIsRecipeLoading(false);
        }
      } else {
        // View Recipes action
        setIsRecipeLoading(true);
        try {
          const recipes = await findRecipesForAtRiskItems(items);
          setRecipes(recipes);
          setLastRecipeRecommendation(new Date());
          setIsProactivePanelOpen(false);
          toast({
            title: 'Recipes Found',
            description: `${recipes.length} recipes loaded`,
          });
        } catch (error) {
          toast({
            title: 'Error',
            description: 'Failed to load recipes',
            variant: 'destructive',
          });
        } finally {
          setIsRecipeLoading(false);
        }
      }
    } else if (alert.type === 'shopping_optimization' && alert.action) {
      // Show shopping list optimization suggestions
      setAdjustments(alert.action.data);
      setIsProactivePanelOpen(false);
      toast({
        title: 'Suggestions Ready',
        description: 'Review shopping list optimizations below',
      });
    }
    dismissAlert(alert.id);
  };

  const getPriorityColor = (priority: ProactiveAlert['priority']) => {
    switch (priority) {
      case 'critical': return 'bg-red-50 border-red-300 text-red-900';
      case 'high': return 'bg-amber-50 border-amber-300 text-amber-900';
      case 'medium': return 'bg-blue-50 border-blue-300 text-blue-900';
      case 'low': return 'bg-green-50 border-green-300 text-green-900';
    }
  };

  return (
    <div className="container mx-auto p-4 md:p-8 relative">
      {/* Floating Proactive Intelligence Button */}
      {proactiveAlerts.length > 0 && (
        <>
          {/* Backdrop */}
          {isProactivePanelOpen && (
            <div 
              className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 transition-opacity duration-300"
              onClick={() => setIsProactivePanelOpen(false)}
            />
          )}
          
          {/* Floating Button */}
          <div className="fixed bottom-6 right-6 z-50">
            <button
              onClick={() => setIsProactivePanelOpen(!isProactivePanelOpen)}
              className="relative group"
            >
              {/* Pulse animation ring */}
              <span className="absolute inset-0 rounded-full bg-primary/30 animate-ping" />
              
              {/* Main button */}
              <div className="relative flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-primary to-primary/80 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-110">
                <Sparkles className="h-7 w-7 text-primary-foreground animate-pulse" />
                
                {/* Alert badge */}
                <div className="absolute -top-1 -right-1 flex items-center justify-center w-6 h-6 rounded-full bg-destructive text-destructive-foreground text-xs font-bold shadow-md animate-bounce">
                  {proactiveAlerts.length}
                </div>
              </div>
            </button>
          </div>

          {/* Sliding Panel */}
          <div 
            className={`fixed bottom-6 right-6 z-40 transition-all duration-500 ease-in-out ${
              isProactivePanelOpen 
                ? 'translate-x-0 opacity-100' 
                : 'translate-x-[150%] opacity-0 pointer-events-none'
            }`}
          >
            <Card className="w-[400px] max-h-[600px] shadow-2xl border-2 border-primary/30 backdrop-blur-xl bg-background/95">
              <CardHeader className="pb-3 border-b">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary/10">
                      <Sparkles className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">Proactive Intelligence</CardTitle>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {proactiveAlerts.length} active {proactiveAlerts.length === 1 ? 'alert' : 'alerts'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setProactiveAlerts([])}
                      className="h-8 text-xs"
                    >
                      Clear All
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => setIsProactivePanelOpen(false)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <ScrollArea className="h-[500px]">
                <CardContent className="pt-4">
                  <div className="space-y-3">
                    {proactiveAlerts.map((alert, index) => (
                      <div
                        key={alert.id}
                        className="animate-in fade-in slide-in-from-right duration-300"
                        style={{ animationDelay: `${index * 50}ms` }}
                      >
                        <Alert className={getPriorityColor(alert.priority)}>
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1">
                              <AlertTitle className="text-sm font-semibold mb-1">
                                {alert.title}
                              </AlertTitle>
                              <AlertDescription className="text-xs">
                                {alert.message}
                              </AlertDescription>
                              {alert.action && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="h-7 text-xs mt-2"
                                  onClick={() => handleAlertAction(alert)}
                                >
                                  {alert.action.label}
                                </Button>
                              )}
                            </div>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0 hover:bg-background/50"
                              onClick={() => dismissAlert(alert.id)}
                            >
                              <X className="h-3 w-3" />
                            </Button>
                          </div>
                        </Alert>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </ScrollArea>
            </Card>
          </div>
        </>
      )}

      {/* Automated Waste Alert Banner */}
      {wasteAnalysis && wasteAnalysis.recommendations.length > 0 && (
        <Alert variant="destructive" className="mb-6 border-red-300 bg-red-50">
          <AlertTriangle className="h-5 w-5 text-red-600" />
          <AlertTitle className="text-red-900 font-bold text-base">Waste Alert - Immediate Action Needed</AlertTitle>
          <AlertDescription className="text-red-800">
            <ul className="list-disc list-inside space-y-1.5 mt-2 ml-1">
              {wasteAnalysis.recommendations.map((rec, idx) => (
                <li key={idx} className="text-sm">{rec}</li>
              ))}
            </ul>
            {wasteAnalysis.wasteValue > 0 && (
              <div className="mt-3 pt-3 border-t border-red-200 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <DollarSign className="h-5 w-5 text-red-600" />
                  <span className="font-semibold text-base">
                    Potential waste value: ${wasteAnalysis.wasteValue.toFixed(2)}
                  </span>
                </div>
                <Badge variant="destructive" className="text-xs">
                  {wasteAnalysis.highRiskItems.length} high-risk items
                </Badge>
              </div>
            )}
          </AlertDescription>
        </Alert>
      )}
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 lg:items-stretch">
        <div className="lg:col-span-2">
          <PantryInventory 
            items={pantryItems || []}
            onAddItem={addPantryItem}
            onRemoveItem={removePantryItem}
            loading={pantryLoading}
          />
        </div>
        <div className="space-y-8">
          <AtRiskItems 
            items={pantryItems || []}
            recipes={recipes}
            setRecipes={setRecipes}
            isLoading={isRecipeLoading}
            setIsLoading={setIsRecipeLoading}
          />
          <ShoppingList 
            items={shoppingList}
            pantryItems={pantryItems || []}
            adjustments={adjustments}
            setAdjustments={setAdjustments}
            isLoading={isShoppingListLoading}
            setIsLoading={setIsShoppingListLoading}
            onAddItem={addShoppingListItem}
            onRemoveItem={removeShoppingListItem}
          />
        </div>
      </div>
    </div>
  );
}
