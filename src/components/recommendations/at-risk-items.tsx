'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import type { PantryItem, Recipe } from "@/lib/types";
import { findRecipesForAtRiskItems } from "@/lib/actions";
import { RecipeCard } from "./recipe-card";
import { AlertCircle, ChefHat, Loader2 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

type AtRiskItemsProps = {
  items: PantryItem[];
  recipes: Recipe[] | null;
  setRecipes: React.Dispatch<React.SetStateAction<Recipe[] | null>>;
  isLoading: boolean;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
};

export function AtRiskItems({ items, recipes, setRecipes, isLoading, setIsLoading }: AtRiskItemsProps) {
  const atRiskItems = items.filter(item => item.riskClass === 'High');

  const handleFindRecipes = async () => {
    setIsLoading(true);
    setRecipes(null);
    const foundRecipes = await findRecipesForAtRiskItems(items);
    setRecipes(foundRecipes);
    setIsLoading(false);
  };

  return (
    <Card>
      <CardHeader className="p-4 sm:p-6">
        <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
          <ChefHat className="h-4 w-4 sm:h-5 sm:w-5" />
          AI Recipe Recommendations
        </CardTitle>
        <CardDescription className="text-xs sm:text-sm">Smart recipes to use your at-risk items</CardDescription>
      </CardHeader>
      <CardContent className="p-4 sm:p-6 pt-0">
        {atRiskItems.length === 0 ? (
          <Alert className="bg-green-50 border-green-200 animate-in fade-in slide-in-from-bottom-2 duration-500">
            <AlertCircle className="h-4 w-4 text-green-600" />
            <AlertTitle className="text-green-800">All Good!</AlertTitle>
            <AlertDescription className="text-green-700">
              No items at high risk. Your pantry is well-managed!
            </AlertDescription>
          </Alert>
        ) : (
          <Alert variant="destructive" className="bg-destructive/10">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle className="font-semibold">
              {atRiskItems.length} {atRiskItems.length === 1 ? 'item' : 'items'} at high risk!
            </AlertTitle>
            <AlertDescription className="mt-1">
              <span className="font-medium">
                {atRiskItems.map(item => item.name).join(', ')}
              </span>
            </AlertDescription>
          </Alert>
        )}
        <Button 
          onClick={handleFindRecipes} 
          disabled={isLoading || atRiskItems.length === 0} 
          className="w-full mt-4"
          size="lg"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Finding Recipes...
            </>
          ) : (
            <>
              <ChefHat className="mr-2 h-4 w-4" />
              Find Recipes to Cook
            </>
          )}
        </Button>
        {recipes && recipes.length > 0 && (
          <>
            <Separator className="my-4" />
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold font-headline">Recipe Suggestions</h3>
              <Badge variant="outline" className="text-xs">{recipes.length} recipes</Badge>
            </div>
            <ScrollArea className="h-[250px] sm:h-[300px]">
              <div className="space-y-3 pr-2 sm:pr-4">
                {recipes.map(recipe => (
                  <RecipeCard key={recipe.id} recipe={recipe} />
                ))}
              </div>
            </ScrollArea>
          </>
        )}
        {recipes && recipes.length === 0 && !isLoading && (
            <>
                <Separator className="my-4" />
                <div className="text-center py-6">
                  <ChefHat className="h-12 w-12 mx-auto text-muted-foreground/30 mb-2" />
                  <p className="text-sm text-muted-foreground">No recipes found for these items.</p>
                  <p className="text-xs text-muted-foreground/70 mt-1">Try adding more items or different categories.</p>
                </div>
            </>
        )}
         {isLoading && (
          <>
            <Separator className="my-4" />
            <div className="space-y-4 pr-4">
                <div className="flex items-center space-x-4">
                    <Skeleton className="h-24 w-24" />
                    <div className="space-y-2">
                        <Skeleton className="h-4 w-[150px]" />
                        <Skeleton className="h-4 w-[100px]" />
                    </div>
                </div>
                <div className="flex items-center space-x-4">
                    <Skeleton className="h-24 w-24" />
                    <div className="space-y-2">
                        <Skeleton className="h-4 w-[150px]" />
                        <Skeleton className="h-4 w-[100px]" />
                    </div>
                </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
