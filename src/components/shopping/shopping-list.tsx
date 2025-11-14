'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { getShoppingListSuggestions } from "@/lib/actions";
import type { ShoppingListItem, PantryItem, ShoppingListAdjustment } from "@/lib/types";
import { CheckCircle2, ListPlus, Loader2, MinusCircle, ShoppingCart, Trash2, ShoppingBag, Plus, TrendingUp, TrendingDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";

type ShoppingListProps = {
  items: ShoppingListItem[];
  pantryItems: PantryItem[];
  adjustments: ShoppingListAdjustment[] | null;
  setAdjustments: React.Dispatch<React.SetStateAction<ShoppingListAdjustment[] | null>>;
  isLoading: boolean;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
  onAddItem: (item: Omit<ShoppingListItem, 'id'>) => void;
  onRemoveItem: (id: string) => void;
};

export function ShoppingList({ items, pantryItems, adjustments, setAdjustments, isLoading, setIsLoading, onAddItem, onRemoveItem }: ShoppingListProps) {
  const [newItemName, setNewItemName] = useState('');
  const [newItemQty, setNewItemQty] = useState(1);
  const [newItemUnit, setNewItemUnit] = useState<'kg' | 'g' | 'L' | 'ml' | 'piece' | 'pack'>('piece');

  const handleAddItem = () => {
    if (newItemName) {
      onAddItem({ name: newItemName, quantity: newItemQty, unit: newItemUnit });
      setNewItemName('');
      setNewItemQty(1);
    }
  };
  
  const handleOptimizeList = async () => {
    setIsLoading(true);
    setAdjustments(null);
    const foundAdjustments = await getShoppingListSuggestions(pantryItems, items);
    setAdjustments(foundAdjustments);
    setIsLoading(false);
  };

  return (
    <Card className="shadow-md">
      <CardHeader className="p-4 sm:p-6">
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <ShoppingBag className="h-5 w-5 text-primary" />
          <CardTitle>Shopping List</CardTitle>
          <Badge variant="secondary" className="ml-2">
            {items.length} {items.length === 1 ? 'item' : 'items'}
          </Badge>
        </div>
        <CardDescription>Plan your next grocery trip with smart suggestions.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 max-h-40 sm:max-h-48 overflow-y-auto pr-2">
          {items.map((item, index) => (
            <div 
              key={item.id} 
              className="flex items-center justify-between text-sm p-2 rounded-md hover:bg-muted/50 transition-all duration-200 animate-in slide-in-from-left"
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <span className="truncate pr-2">{item.quantity} {item.unit} of {item.name}</span>
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-7 w-7 flex-shrink-0 hover:bg-destructive/10 hover:text-destructive transition-colors" 
                onClick={() => onRemoveItem(item.id)}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          ))}
          {items.length === 0 && (
            <div className="flex flex-col items-center justify-center py-6 text-muted-foreground">
              <ShoppingBag className="h-10 w-10 mb-2 opacity-20" />
              <p className="text-sm font-medium">Your shopping list is empty</p>
              <p className="text-xs mt-1">Add items below to get started</p>
            </div>
          )}
        </div>
        
        <Separator className="my-4" />
        
        <div className="space-y-2">
          <p className="text-sm font-semibold flex items-center gap-1.5">
            <Plus className="h-4 w-4" />
            Add to list
          </p>
          <div className="flex flex-col sm:flex-row gap-2 items-stretch sm:items-center">
            <Input 
              placeholder="Item name" 
              value={newItemName} 
              onChange={e => setNewItemName(e.target.value)} 
              onKeyDown={(e) => e.key === 'Enter' && handleAddItem()}
              className="flex-1"
            />
            <div className="flex gap-2">
              <Input 
                type="number" 
                min="1" 
                value={newItemQty} 
                onChange={e => setNewItemQty(Number(e.target.value))} 
                className="w-20 sm:w-20" 
              />
              <Button 
                size="icon" 
                onClick={handleAddItem} 
                aria-label="Add item to shopping list"
                className="bg-primary hover:bg-primary/90"
              >
                <ListPlus className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>

        <Button 
          onClick={handleOptimizeList} 
          disabled={isLoading || items.length === 0} 
          className="w-full mt-4 bg-gradient-to-r from-accent to-accent/80 hover:from-accent/90 hover:to-accent/70 text-accent-foreground shadow-sm"
          size="lg"
        >
          {isLoading ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <ShoppingCart className="mr-2 h-4 w-4" />
          )}
          {isLoading ? "Optimizing..." : "Optimize Shopping List"}
        </Button>

        {isLoading && <p className="text-sm text-muted-foreground text-center mt-4 animate-pulse">Optimizing...</p>}

        {adjustments && adjustments.length > 0 && (
          <div className="mt-4 space-y-2">
            <div className="flex items-center gap-2">
              <h3 className="text-md font-semibold font-headline">Smart Suggestions</h3>
              <Badge variant="outline" className="text-[10px]">
                {adjustments.length}
              </Badge>
            </div>
            {adjustments.map(adj => {
              const isIncrease = adj.quantityToReduce < 0;
              const quantity = Math.abs(adj.quantityToReduce);
              return (
                <Alert key={adj.itemId} variant="default" className={isIncrease ? "bg-green-50 border-green-200" : "bg-amber-50 border-amber-200"}>
                  {isIncrease ? (
                    <TrendingUp className="h-4 w-4 text-green-600" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-amber-600" />
                  )}
                  <AlertTitle className={isIncrease ? "text-green-800" : "text-amber-800"}>
                    {isIncrease ? 'Increase' : 'Reduce'} {adj.name}
                  </AlertTitle>
                  <AlertDescription className={isIncrease ? "text-green-700" : "text-amber-700"}>
                    {isIncrease ? 'Add' : 'Remove'} {quantity} {adj.unit} • {adj.reason}
                  </AlertDescription>
                </Alert>
              );
            })}
          </div>
        )}
        {!isLoading && adjustments && adjustments.length === 0 && (
          <Alert variant="default" className="mt-4 bg-primary/10 border-primary/20 text-primary">
             <CheckCircle2 className="h-4 w-4" />
             <AlertTitle>List Optimized</AlertTitle>
             <AlertDescription>Your shopping list looks good! No suggestions at this time.</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}
