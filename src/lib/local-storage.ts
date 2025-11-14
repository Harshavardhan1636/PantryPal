/**
 * LocalStorage utility functions for data persistence
 */

// Storage keys
export const STORAGE_KEYS = {
  USER: 'pantrypal_user',
  HOUSEHOLD: 'pantrypal_household',
  PANTRY_ITEMS: 'pantrypal_pantry_items',
  SHOPPING_LIST: 'pantrypal_shopping_list',
  RECIPES: 'pantrypal_recipes',
  WASTE_HISTORY: 'pantrypal_waste_history',
  NOTIFICATIONS: 'pantrypal_notifications',
} as const;

/**
 * Safely get data from localStorage
 */
export function getFromStorage<T>(key: string): T | null {
  if (typeof window === 'undefined') return null;
  
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : null;
  } catch (error) {
    console.error(`Error reading from localStorage key "${key}":`, error);
    return null;
  }
}

/**
 * Safely set data to localStorage
 */
export function setToStorage<T>(key: string, value: T): void {
  if (typeof window === 'undefined') return;
  
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.error(`Error writing to localStorage key "${key}":`, error);
  }
}

/**
 * Remove data from localStorage
 */
export function removeFromStorage(key: string): void {
  if (typeof window === 'undefined') return;
  
  try {
    localStorage.removeItem(key);
  } catch (error) {
    console.error(`Error removing from localStorage key "${key}":`, error);
  }
}

/**
 * Clear all app data from localStorage
 */
export function clearAllStorage(): void {
  if (typeof window === 'undefined') return;
  
  Object.values(STORAGE_KEYS).forEach(key => {
    removeFromStorage(key);
  });
}

/**
 * Generate a unique ID
 */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}
