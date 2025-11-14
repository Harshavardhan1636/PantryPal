'use client';

import { useState, useEffect, useCallback } from 'react';
import { getFromStorage, setToStorage, generateId } from '@/lib/local-storage';

/**
 * Hook for managing a collection of items in localStorage
 */
export function useLocalStorage<T extends { id: string }>(
  key: string,
  initialValue: T[] = []
) {
  const [data, setData] = useState<T[]>(initialValue);
  const [loading, setLoading] = useState(true);

  // Load data from localStorage on mount
  useEffect(() => {
    const stored = getFromStorage<T[]>(key);
    if (stored) {
      setData(stored);
    }
    setLoading(false);
  }, [key]);

  // Save data to localStorage whenever it changes
  useEffect(() => {
    if (!loading) {
      setToStorage(key, data);
    }
  }, [key, data, loading]);

  // Add a new item
  const addItem = useCallback((item: Omit<T, 'id'>) => {
    const newItem = { ...item, id: generateId() } as T;
    setData(prev => [newItem, ...prev]);
    return newItem;
  }, []);

  // Update an existing item
  const updateItem = useCallback((id: string, updates: Partial<T>) => {
    setData(prev =>
      prev.map(item => (item.id === id ? { ...item, ...updates } : item))
    );
  }, []);

  // Remove an item
  const removeItem = useCallback((id: string) => {
    setData(prev => prev.filter(item => item.id !== id));
  }, []);

  // Replace all data
  const setItems = useCallback((items: T[]) => {
    setData(items);
  }, []);

  // Clear all data
  const clearItems = useCallback(() => {
    setData([]);
  }, []);

  return {
    data,
    loading,
    addItem,
    updateItem,
    removeItem,
    setItems,
    clearItems,
  };
}
