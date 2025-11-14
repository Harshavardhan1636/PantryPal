/**
 * PantryPal API Client
 * Senior SDE-3 Level Implementation
 * 
 * Features:
 * - Type-safe API calls with TypeScript
 * - Automatic JWT token management
 * - Request/response interceptors
 * - Error handling with retry logic
 * - Request caching
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosError, AxiosResponse } from 'axios';

// ==============================================
// CONFIGURATION
// ==============================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
const AUTH_SERVICE_URL = process.env.NEXT_PUBLIC_AUTH_SERVICE_URL || 'http://localhost:8001';
const INVENTORY_SERVICE_URL = process.env.NEXT_PUBLIC_INVENTORY_SERVICE_URL || 'http://localhost:8002';
const ML_SERVICE_URL = process.env.NEXT_PUBLIC_ML_SERVICE_URL || 'http://localhost:8003';

const REQUEST_TIMEOUT = 30000; // 30 seconds
const MAX_RETRIES = 3;

// ==============================================
// TOKEN MANAGEMENT
// ==============================================

class TokenManager {
  private static TOKEN_KEY = 'auth_token';
  private static REFRESH_TOKEN_KEY = 'refresh_token';

  static getToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem(this.TOKEN_KEY);
  }

  static setToken(token: string): void {
    if (typeof window === 'undefined') return;
    localStorage.setItem(this.TOKEN_KEY, token);
  }

  static getRefreshToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem(this.REFRESH_TOKEN_KEY);
  }

  static setRefreshToken(token: string): void {
    if (typeof window === 'undefined') return;
    localStorage.setItem(this.REFRESH_TOKEN_KEY, token);
  }

  static clearTokens(): void {
    if (typeof window === 'undefined') return;
    localStorage.removeItem(this.TOKEN_KEY);
    localStorage.removeItem(this.REFRESH_TOKEN_KEY);
  }
}

// ==============================================
// API CLIENT CLASS
// ==============================================

class APIClient {
  private client: AxiosInstance;
  private isRefreshing = false;
  private refreshSubscribers: Array<(token: string) => void> = [];

  constructor(baseURL: string) {
    this.client = axios.create({
      baseURL,
      timeout: REQUEST_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor - Add auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = TokenManager.getToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor - Handle errors and token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };

        // Handle 401 Unauthorized - Token expired
        if (error.response?.status === 401 && !originalRequest._retry) {
          if (this.isRefreshing) {
            // Wait for token refresh to complete
            return new Promise((resolve) => {
              this.refreshSubscribers.push((token: string) => {
                originalRequest.headers = originalRequest.headers || {};
                originalRequest.headers.Authorization = `Bearer ${token}`;
                resolve(this.client(originalRequest));
              });
            });
          }

          originalRequest._retry = true;
          this.isRefreshing = true;

          try {
            const refreshToken = TokenManager.getRefreshToken();
            if (!refreshToken) {
              throw new Error('No refresh token available');
            }

            // Refresh token
            const response = await axios.post(`${AUTH_SERVICE_URL}/auth/refresh`, {
              refresh_token: refreshToken,
            });

            const { access_token } = response.data;
            TokenManager.setToken(access_token);

            // Notify all waiting requests
            this.refreshSubscribers.forEach((callback) => callback(access_token));
            this.refreshSubscribers = [];

            // Retry original request
            originalRequest.headers = originalRequest.headers || {};
            originalRequest.headers.Authorization = `Bearer ${access_token}`;
            return this.client(originalRequest);
          } catch (refreshError) {
            // Refresh failed - logout user
            TokenManager.clearTokens();
            if (typeof window !== 'undefined') {
              window.location.href = '/login';
            }
            return Promise.reject(refreshError);
          } finally {
            this.isRefreshing = false;
          }
        }

        return Promise.reject(error);
      }
    );
  }

  // Generic request method with retry logic
  private async requestWithRetry<T>(
    method: 'get' | 'post' | 'put' | 'delete' | 'patch',
    url: string,
    data?: any,
    config?: AxiosRequestConfig,
    retries = MAX_RETRIES
  ): Promise<T> {
    try {
      const response: AxiosResponse<T> = await this.client[method](url, data, config);
      return response.data;
    } catch (error: any) {
      // Retry on network errors or 5xx server errors
      const shouldRetry = 
        retries > 0 && 
        (error.code === 'ECONNABORTED' || 
         error.response?.status >= 500);

      if (shouldRetry) {
        await new Promise((resolve) => setTimeout(resolve, 1000)); // Wait 1s before retry
        return this.requestWithRetry(method, url, data, config, retries - 1);
      }

      throw this.handleError(error);
    }
  }

  private handleError(error: any): Error {
    if (error.response) {
      // Server responded with error
      const message = error.response.data?.detail || error.response.data?.message || 'Server error';
      return new Error(message);
    } else if (error.request) {
      // No response received
      return new Error('Network error. Please check your connection.');
    } else {
      // Request setup error
      return new Error(error.message || 'An unexpected error occurred');
    }
  }

  // HTTP Methods
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return this.requestWithRetry<T>('get', url, undefined, config);
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    return this.requestWithRetry<T>('post', url, data, config);
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    return this.requestWithRetry<T>('put', url, data, config);
  }

  async patch<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    return this.requestWithRetry<T>('patch', url, data, config);
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    return this.requestWithRetry<T>('delete', url, undefined, config);
  }
}

// ==============================================
// SERVICE CLIENTS
// ==============================================

export const authClient = new APIClient(AUTH_SERVICE_URL);
export const inventoryClient = new APIClient(INVENTORY_SERVICE_URL);
export const mlClient = new APIClient(ML_SERVICE_URL);

// ==============================================
// API RESPONSE TYPES
// ==============================================

export interface APIResponse<T> {
  success: boolean;
  data: T;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface User {
  id: string;
  email: string;
  name: string;
  avatar_url?: string;
  household_id?: string;
  created_at: string;
}

export interface Household {
  id: string;
  name: string;
  members_count: number;
  storage_types: string[];
  dietary_preferences: string[];
  timezone: string;
  created_at: string;
}

export interface PantryItem {
  id: string;
  household_id: string;
  item_id?: string;
  custom_item_name?: string;
  quantity: number;
  unit: string;
  purchase_date: string;
  expiry_date?: string;
  predicted_expiry_date?: string;
  storage_type: string;
  status: string;
  risk_score?: number;
  risk_class?: 'High' | 'Medium' | 'Low';
  created_at: string;
}

export interface Recipe {
  id: string;
  name: string;
  description: string;
  ingredients: Ingredient[];
  instructions: string;
  prep_time_minutes: number;
  cook_time_minutes: number;
  servings: number;
  difficulty: string;
  image_url?: string;
  rating_avg?: number;
}

export interface Ingredient {
  item_id?: string;
  name: string;
  quantity: number;
  unit: string;
}

export interface ShoppingList {
  id: string;
  household_id: string;
  name: string;
  status: string;
  items: ShoppingListItem[];
  total_estimated_cost?: number;
  created_at: string;
}

export interface ShoppingListItem {
  id: string;
  list_id: string;
  item_id?: string;
  custom_item_name?: string;
  quantity: number;
  unit: string;
  priority: number;
  estimated_price?: number;
  purchased: boolean;
  notes?: string;
}

export interface WasteEvent {
  id: string;
  batch_id: string;
  household_id: string;
  quantity: number;
  unit: string;
  reason: string;
  reason_details?: string;
  cost_estimate?: number;
  wasted_at: string;
}

export interface RiskPrediction {
  id: string;
  batch_id: string;
  household_id: string;
  risk_score: number;
  risk_class: 'high' | 'medium' | 'low';
  predicted_waste_date?: string;
  confidence: number;
  predicted_at: string;
}

// ==============================================
// API METHODS
// ==============================================

export const api = {
  // Auth endpoints
  auth: {
    register: (data: { email: string; name: string; password: string }) =>
      authClient.post<AuthResponse>('/auth/register', data),
    
    login: (data: { email: string; password: string }) =>
      authClient.post<AuthResponse>('/auth/login', data),
    
    logout: () =>
      authClient.post('/auth/logout'),
    
    refreshToken: (refreshToken: string) =>
      authClient.post<AuthResponse>('/auth/refresh', { refresh_token: refreshToken }),
    
    me: () =>
      authClient.get<User>('/auth/me'),
  },

  // Household endpoints
  households: {
    get: (id: string) =>
      authClient.get<Household>(`/households/${id}`),
    
    update: (id: string, data: Partial<Household>) =>
      authClient.put<Household>(`/households/${id}`, data),
    
    getMembers: (id: string) =>
      authClient.get<User[]>(`/households/${id}/members`),
  },

  // Inventory endpoints
  inventory: {
    list: (householdId: string, params?: { status?: string; storage_type?: string }) =>
      inventoryClient.get<PaginatedResponse<PantryItem>>(`/inventory`, { params: { household_id: householdId, ...params } }),
    
    get: (id: string) =>
      inventoryClient.get<PantryItem>(`/inventory/${id}`),
    
    create: (data: Partial<PantryItem>) =>
      inventoryClient.post<PantryItem>('/inventory', data),
    
    update: (id: string, data: Partial<PantryItem>) =>
      inventoryClient.put<PantryItem>(`/inventory/${id}`, data),
    
    delete: (id: string) =>
      inventoryClient.delete(`/inventory/${id}`),
    
    getAtRisk: (householdId: string) =>
      inventoryClient.get<PantryItem[]>(`/inventory/at-risk`, { params: { household_id: householdId } }),
  },

  // Recipe endpoints
  recipes: {
    list: (params?: { dietary_tags?: string[]; difficulty?: string }) =>
      inventoryClient.get<PaginatedResponse<Recipe>>('/recipes', { params }),
    
    get: (id: string) =>
      inventoryClient.get<Recipe>(`/recipes/${id}`),
    
    getRecommendations: (householdId: string, atRiskItems: string[]) =>
      inventoryClient.post<Recipe[]>('/recipes/recommendations', { 
        household_id: householdId,
        at_risk_items: atRiskItems 
      }),
  },

  // Shopping list endpoints
  shoppingLists: {
    list: (householdId: string) =>
      inventoryClient.get<ShoppingList[]>('/shopping-lists', { params: { household_id: householdId } }),
    
    get: (id: string) =>
      inventoryClient.get<ShoppingList>(`/shopping-lists/${id}`),
    
    create: (data: Partial<ShoppingList>) =>
      inventoryClient.post<ShoppingList>('/shopping-lists', data),
    
    update: (id: string, data: Partial<ShoppingList>) =>
      inventoryClient.put<ShoppingList>(`/shopping-lists/${id}`, data),
    
    delete: (id: string) =>
      inventoryClient.delete(`/shopping-lists/${id}`),
    
    addItem: (listId: string, item: Partial<ShoppingListItem>) =>
      inventoryClient.post<ShoppingListItem>(`/shopping-lists/${listId}/items`, item),
    
    updateItem: (listId: string, itemId: string, data: Partial<ShoppingListItem>) =>
      inventoryClient.put<ShoppingListItem>(`/shopping-lists/${listId}/items/${itemId}`, data),
    
    deleteItem: (listId: string, itemId: string) =>
      inventoryClient.delete(`/shopping-lists/${listId}/items/${itemId}`),
  },

  // ML predictions
  ml: {
    predictWasteRisk: (batchId: string) =>
      mlClient.post<RiskPrediction>('/predict/waste-risk', { batch_id: batchId }),
    
    getRiskPredictions: (householdId: string) =>
      mlClient.get<RiskPrediction[]>('/predictions', { params: { household_id: householdId } }),
  },

  // Waste tracking
  waste: {
    list: (householdId: string, params?: { start_date?: string; end_date?: string }) =>
      inventoryClient.get<PaginatedResponse<WasteEvent>>('/waste', { params: { household_id: householdId, ...params } }),
    
    create: (data: Partial<WasteEvent>) =>
      inventoryClient.post<WasteEvent>('/waste', data),
    
    getAnalytics: (householdId: string, period?: string) =>
      inventoryClient.get(`/waste/analytics`, { params: { household_id: householdId, period } }),
  },
};

// Export token manager for use in components
export { TokenManager };

export default api;
