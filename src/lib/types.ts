export type PantryItem = {
  id: string;
  name: string;
  category: string;
  quantity: number;
  unit: 'kg' | 'g' | 'L' | 'ml' | 'piece' | 'pack';
  purchaseDate: string;
  expiryDate?: string;
  riskClass?: 'High' | 'Medium' | 'Low';
  riskScore?: number;
};

export type ShoppingListItem = {
  id: string;
  name: string;
  quantity: number;
  unit: 'kg' | 'g' | 'L' | 'ml' | 'piece' | 'pack';
};

export type Recipe = {
  id: string;
  title?: string;
  name?: string;
  ingredients: string[];
  steps?: string;
  prep_time?: string;
  servings?: string;
  tags?: string[];
  source?: string;
  imageUrl?: string;
  imageHint?: string;
};

export type ShoppingListAdjustment = {
  itemId: string;
  name: string;
  quantityToReduce: number; // Negative = increase, Positive = reduce
  unit: string;
  reason: string;
};

export type HouseholdProfile = {
  householdSize: number;
  dietaryPreferences: string[];
  storageTypes: string[];
}

export type Household = {
  id: string;
  name: string;
  timezone: string;
  membersCount: number;
  storageTypesJson: string;
  dietaryPreferencesJson: string;
};
