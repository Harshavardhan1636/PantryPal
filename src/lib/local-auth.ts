/**
 * Local authentication system using localStorage
 */

import { getFromStorage, setToStorage, removeFromStorage, STORAGE_KEYS, generateId } from './local-storage';

export type User = {
  id: string;
  name: string;
  email: string;
  isGuest: boolean;
  createdAt: string;
};

export type Household = {
  id: string;
  name: string;
  membersCount: number;
  dietaryPreferences: string[];
  storageTypes: string[];
};

/**
 * Get current user from localStorage
 */
export function getCurrentUser(): User | null {
  return getFromStorage<User>(STORAGE_KEYS.USER);
}

/**
 * Get current household from localStorage
 */
export function getCurrentHousehold(): Household | null {
  return getFromStorage<Household>(STORAGE_KEYS.HOUSEHOLD);
}

/**
 * Sign in with email (simplified - no actual authentication)
 */
export function signInWithEmail(email: string, name: string): User {
  const user: User = {
    id: generateId(),
    name,
    email,
    isGuest: false,
    createdAt: new Date().toISOString(),
  };
  
  setToStorage(STORAGE_KEYS.USER, user);
  return user;
}

/**
 * Sign in as guest
 */
export function signInAsGuest(): User {
  const user: User = {
    id: generateId(),
    name: 'Guest User',
    email: 'guest@pantrypal.local',
    isGuest: true,
    createdAt: new Date().toISOString(),
  };
  
  setToStorage(STORAGE_KEYS.USER, user);
  return user;
}

/**
 * Sign out current user
 */
export function signOut(): void {
  removeFromStorage(STORAGE_KEYS.USER);
  removeFromStorage(STORAGE_KEYS.HOUSEHOLD);
}

/**
 * Update household profile
 */
export function updateHousehold(household: Partial<Household>): Household {
  const user = getCurrentUser();
  if (!user) throw new Error('No user logged in');
  
  const existing = getCurrentHousehold();
  const updated: Household = {
    id: user.id,
    name: household.name || existing?.name || `${user.name}'s Household`,
    membersCount: household.membersCount ?? existing?.membersCount ?? 1,
    dietaryPreferences: household.dietaryPreferences ?? existing?.dietaryPreferences ?? [],
    storageTypes: household.storageTypes ?? existing?.storageTypes ?? ['refrigerator', 'pantry', 'freezer'],
  };
  
  setToStorage(STORAGE_KEYS.HOUSEHOLD, updated);
  return updated;
}

/**
 * Check if user profile is complete
 */
export function isProfileComplete(): boolean {
  const household = getCurrentHousehold();
  return household !== null && household.membersCount > 0;
}
