import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { PantryItem } from "@/lib/types";
import { PantryItemCard } from "./pantry-item-card";
import { AddPantryItemDialog } from "./add-pantry-item-dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Package } from "lucide-react";

type PantryInventoryProps = {
  items: PantryItem[];
  onAddItem: (item: Omit<PantryItem, 'id' | 'riskClass' | 'riskScore'>) => void;
  onRemoveItem: (id: string) => void;
  loading: boolean;
};

export function PantryInventory({ items, onAddItem, onRemoveItem, loading }: PantryInventoryProps) {
  // Calculate risk distribution
  const highRisk = items.filter(item => item.riskClass === 'High').length;
  const mediumRisk = items.filter(item => item.riskClass === 'Medium').length;
  const lowRisk = items.filter(item => item.riskClass === 'Low').length;

  return (
    <Card className="h-full shadow-md flex flex-col">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <Package className="h-5 w-5 text-primary" />
            <CardTitle>Pantry Inventory</CardTitle>
            <Badge variant="secondary" className="ml-2">
              {items.length} {items.length === 1 ? 'item' : 'items'}
            </Badge>
          </div>
          <CardDescription className="flex items-center gap-2 flex-wrap mt-2">
            <span>Risk Distribution:</span>
            {highRisk > 0 && (
              <Badge variant="destructive" className="text-[10px] px-1.5 py-0">
                {highRisk} High
              </Badge>
            )}
            {mediumRisk > 0 && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 bg-amber-50 border-amber-200 text-amber-700">
                {mediumRisk} Medium
              </Badge>
            )}
            {lowRisk > 0 && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 bg-green-50 border-green-200 text-green-700">
                {lowRisk} Low
              </Badge>
            )}
          </CardDescription>
        </div>
        <AddPantryItemDialog onAddItem={onAddItem} />
      </CardHeader>
      <CardContent className="p-4 sm:p-6 flex-1 flex flex-col min-h-0">
        <ScrollArea className="h-full pr-2 sm:pr-4">
          {loading ? (
            <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="space-y-3">
                  <Skeleton className="h-32 w-full rounded-lg" />
                  <div className="space-y-2">
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-3 w-1/2" />
                  </div>
                </div>
              ))}
            </div>
          ) : items.length > 0 ? (
            <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {items.map(item => (
                <PantryItemCard key={item.id} item={item} onRemove={onRemoveItem} />
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-40 text-muted-foreground">
              <Package className="h-12 w-12 mb-3 opacity-20" />
              <p className="text-lg font-medium">Your pantry is empty</p>
              <p className="text-sm mt-1">Add some items to get started!</p>
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
