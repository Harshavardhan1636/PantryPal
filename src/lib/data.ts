import type { ShoppingListItem } from './types';

// This file is now used for non-user-specific mock data.
// The demo account data is in `src/lib/mock-data.ts`.

export const initialShoppingList: ShoppingListItem[] = [
  { id: '101', name: 'Apples', quantity: 6, unit: 'piece' },
  { id: '102', name: 'Chicken Breast', quantity: 2, unit: 'pack' },
  { id: '103', name: 'Sourdough Bread', quantity: 1, unit: 'piece' },
  { id: '104', name: 'Greek Yogurt', quantity: 1, unit: 'pack' },
];

export const householdProfile = {
  householdSize: 2,
  dietaryPreferences: 'None',
};
