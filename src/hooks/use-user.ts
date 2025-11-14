'use client';

import { useState, useEffect } from 'react';
import { getCurrentUser, getCurrentHousehold, isProfileComplete, type User, type Household } from '@/lib/local-auth';

/**
 * Hook for accessing current user and household data
 */
export function useUser() {
  const [user, setUser] = useState<User | null>(null);
  const [household, setHousehold] = useState<Household | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load user and household from localStorage
    const currentUser = getCurrentUser();
    const currentHousehold = getCurrentHousehold();
    
    setUser(currentUser);
    setHousehold(currentHousehold);
    setLoading(false);
  }, []);

  // Function to refresh data
  const refresh = () => {
    setUser(getCurrentUser());
    setHousehold(getCurrentHousehold());
  };

  return {
    user,
    household,
    loading,
    profileLoading: false,
    isProfileComplete: isProfileComplete(),
    refresh,
  };
}
