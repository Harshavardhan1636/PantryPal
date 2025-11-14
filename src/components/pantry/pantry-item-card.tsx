import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { PantryItem } from "@/lib/types";
import { differenceInDays, format } from 'date-fns';
import { Banana, Milk, CakeSlice, Package, Trash2, CalendarDays, ShoppingCart, Tag, Leaf } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

type PantryItemCardProps = {
  item: PantryItem;
  onRemove: (id: string) => void;
};

const categoryIcons: { [key: string]: React.ReactNode } = {
  'Fruit': <Banana className="h-4 w-4" />,
  'Dairy': <Milk className="h-4 w-4" />,
  'Bakery': <CakeSlice className="h-4 w-4" />,
  'Vegetable': <Leaf className="h-4 w-4" />,
  'Default': <Package className="h-4 w-4" />,
};

const getRiskColor = (riskClass?: 'High' | 'Medium' | 'Low') => {
  switch (riskClass) {
    case 'High':
      return 'bg-destructive/10 text-destructive border-destructive/20';
    case 'Medium':
      return 'bg-accent/20 text-accent-foreground border-accent/30';
    case 'Low':
    default:
      return 'bg-primary/10 text-primary border-primary/20';
  }
};

export function PantryItemCard({ item, onRemove }: PantryItemCardProps) {
  const daysLeft = item.expiryDate ? differenceInDays(new Date(item.expiryDate), new Date()) : null;
  
  // Enhanced urgency detection
  const isCritical = daysLeft !== null && daysLeft === 0;
  const isUrgent = daysLeft !== null && daysLeft > 0 && daysLeft <= 2;
  const isWarning = daysLeft !== null && daysLeft > 2 && daysLeft <= 5;

  return (
    <Card className={cn(
      "flex flex-col shadow-sm transition-all duration-200",
      isCritical && "border-destructive shadow-destructive/40 [animation:border-pulse_1.5s_ease-in-out_infinite]",
      isUrgent && "border-destructive/50 shadow-destructive/20 hover:shadow-destructive/30 hover:scale-[1.02]",
      isWarning && "border-amber-500/50 shadow-amber-500/20 hover:shadow-amber-500/30",
      !isUrgent && !isWarning && !isCritical && "hover:shadow-md hover:scale-[1.01]"
    )}>
      <CardHeader className="pb-2 sm:pb-3 p-3 sm:p-4">
        <div className="flex justify-between items-start gap-2">
            <div className="flex-1">
                <CardTitle className="text-base sm:text-lg font-semibold group-hover:text-primary transition-colors line-clamp-2">
                  {item.name}
                </CardTitle>
                <CardDescription className="flex items-center gap-2 pt-1 text-xs">
                    {categoryIcons[item.category] || categoryIcons['Default']}
                    {item.category}
                </CardDescription>
            </div>
            <div className="flex flex-col gap-1.5 items-end">
              {isCritical && (
                <Badge variant="destructive" className="whitespace-nowrap text-xs">
                  URGENT
                </Badge>
              )}
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Badge variant="outline" className={cn(
                      "whitespace-nowrap cursor-help",
                      getRiskColor(item.riskClass)
                    )}>
                        {item.riskClass} Risk
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    <p className="text-xs">Risk Score: <strong>{((item.riskScore || 0) * 100).toFixed(0)}%</strong></p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
        </div>
      </CardHeader>
      <CardContent className="flex-grow space-y-2.5 text-sm">
        <div className="flex justify-between items-center">
          <span className="text-muted-foreground flex items-center gap-1.5">
            <Tag className="w-3.5 h-3.5"/> 
            <span className="text-xs">Quantity</span>
          </span>
          <span className="font-medium">{item.quantity} {item.unit}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-muted-foreground flex items-center gap-1.5">
            <ShoppingCart className="w-3.5 h-3.5"/> 
            <span className="text-xs">Purchased</span>
          </span>
          <span className="text-xs">{format(new Date(item.purchaseDate), 'MMM d, yyyy')}</span>
        </div>
        {daysLeft !== null && (
          <div className="flex justify-between items-center pt-1 border-t">
            <span className="text-muted-foreground flex items-center gap-1.5">
              <CalendarDays className={cn(
                "w-3.5 h-3.5",
                isUrgent && "text-destructive animate-pulse",
                isWarning && "text-amber-600"
              )}/> 
              <span className="text-xs">Expires in</span>
            </span>
            <span className={cn(
                "font-bold text-sm",
                daysLeft <= 2 && "text-destructive",
                daysLeft > 2 && daysLeft <= 5 && "text-amber-600",
                daysLeft > 5 && "text-green-600"
            )}>
              {daysLeft > 0 ? `${daysLeft} days` : (daysLeft === 0 ? 'Today' : 'Expired')}
            </span>
          </div>
        )}
      </CardContent>
      <CardContent className="p-4 pt-2">
         <Button 
           variant="ghost" 
           size="sm" 
           className="w-full text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors" 
           onClick={() => onRemove(item.id)}
         >
          <Trash2 className="mr-2 h-4 w-4" />
          Mark as Used/Wasted
        </Button>
      </CardContent>
    </Card>
  );
}
