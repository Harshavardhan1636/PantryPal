import { useState } from "react";
import Image from "next/image";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type { Recipe } from "@/lib/types";
import { Clock, Users, ChefHat, Utensils, ListOrdered } from "lucide-react";

type RecipeCardProps = {
  recipe: Recipe;
};

export function RecipeCard({ recipe }: RecipeCardProps) {
  const [isOpen, setIsOpen] = useState(false);
  
  // Ensure we have valid data
  const title = recipe.title || recipe.name || 'Untitled Recipe';
  const prepTime = recipe.prep_time || 'N/A';
  const servings = recipe.servings || 'N/A';
  const ingredientCount = recipe.ingredients?.length || 0;
  
  // Check if we have full recipe data
  const hasFullRecipe = recipe.ingredients && recipe.ingredients.length > 0 && recipe.steps;

  // Parse steps if it's a string
  const stepsList = typeof recipe.steps === 'string' 
    ? recipe.steps.split(/\d+\.\s+/).filter(s => s.trim())
    : [];
  
  return (
    <>
      <Card 
        className="overflow-hidden transition-all hover:shadow-lg hover:scale-[1.02] cursor-pointer"
        onClick={() => hasFullRecipe && setIsOpen(true)}
      >
      <div className="flex flex-col sm:flex-row">
        {recipe.imageUrl && (
          <div className="relative w-full sm:w-1/3 h-32 sm:h-auto sm:min-w-[100px] bg-muted">
            <Image
              src={recipe.imageUrl}
              alt={title}
              fill
              className="object-cover"
              data-ai-hint={recipe.imageHint}
            />
          </div>
        )}
        {!recipe.imageUrl && (
          <div className="relative w-full sm:w-1/3 h-32 sm:h-auto sm:min-w-[100px] bg-muted flex items-center justify-center">
            <ChefHat className="h-10 w-10 text-muted-foreground/30" />
          </div>
        )}
        <div className="flex-1">
          <CardHeader className="p-3 pb-2">
            <CardTitle className="text-base leading-tight font-headline line-clamp-2">
              {title}
            </CardTitle>
            {recipe.source && (
              <CardDescription className="text-[10px] mt-1">
                Source: {recipe.source} {hasFullRecipe && '• Click to view full recipe'}
              </CardDescription>
            )}
          </CardHeader>
          <CardContent className="p-3 pt-0 text-xs text-muted-foreground">
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              <div className="flex items-center">
                <Clock className="mr-1 h-3 w-3" />
                <span>{prepTime}</span>
              </div>
              <div className="flex items-center">
                <Users className="mr-1 h-3 w-3" />
                <span>{servings}</span>
              </div>
              {ingredientCount > 0 && (
                <div className="flex items-center">
                  <Utensils className="mr-1 h-3 w-3" />
                  <span>{ingredientCount} ingredients</span>
                </div>
              )}
            </div>
            {recipe.tags && recipe.tags.length > 0 &&
              <div className="mt-2 flex flex-wrap gap-1">
                {recipe.tags.slice(0, 3).map((tag, idx) => (
                  <Badge key={`${tag}-${idx}`} variant="secondary" className="text-[10px] px-1.5 py-0">
                    {tag}
                  </Badge>
                ))}
              </div>
            }
          </CardContent>
        </div>
      </div>
    </Card>

    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogContent className="max-w-[95vw] sm:max-w-2xl lg:max-w-3xl max-h-[90vh] sm:max-h-[85vh] flex flex-col p-4 sm:p-6">
        <DialogHeader>
          <DialogTitle className="text-2xl font-headline pr-8">{title}</DialogTitle>
          <div className="flex flex-wrap gap-2 items-center pt-2 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Clock className="h-4 w-4" />
              {prepTime}
            </div>
            <Separator orientation="vertical" className="h-4" />
            <div className="flex items-center gap-1">
              <Utensils className="h-4 w-4" />
              {ingredientCount} ingredients
            </div>
            {stepsList.length > 0 && (
              <>
                <Separator orientation="vertical" className="h-4" />
                <div className="flex items-center gap-1">
                  <ListOrdered className="h-4 w-4" />
                  {stepsList.length} steps
                </div>
              </>
            )}
          </div>
          {recipe.tags && recipe.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 pt-2">
              {recipe.tags.map((tag, idx) => (
                <Badge key={idx} variant="outline" className="text-xs">
                  {tag}
                </Badge>
              ))}
            </div>
          )}
        </DialogHeader>
        
        <ScrollArea className="flex-1 pr-4 overflow-y-auto">
          <div className="space-y-6 py-4">
            {/* Ingredients Section */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Utensils className="h-5 w-5 text-primary" />
                Ingredients
              </h3>
              <ul className="space-y-2">
                {recipe.ingredients?.map((ingredient, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm">
                    <span className="text-primary mt-1">•</span>
                    <span>{ingredient}</span>
                  </li>
                ))}
              </ul>
            </div>

            <Separator />

            {/* Steps Section */}
            {recipe.steps && (
              <div>
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <ListOrdered className="h-5 w-5 text-primary" />
                  Instructions
                </h3>
                {stepsList.length > 0 ? (
                  <ol className="space-y-3">
                    {stepsList.map((step, idx) => (
                      <li key={idx} className="flex gap-3 text-sm">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-xs font-semibold">
                          {idx + 1}
                        </span>
                        <span className="pt-0.5 leading-relaxed">{step.trim()}</span>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="text-sm leading-relaxed">{recipe.steps}</p>
                )}
              </div>
            )}

            {/* Source */}
            <div className="pt-4 border-t">
              <p className="text-xs text-muted-foreground">
                Source: <span className="font-medium">{recipe.source || 'Community'}</span>
              </p>
            </div>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
    </>
  );
}
