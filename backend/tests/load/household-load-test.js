/**
 * k6 Load Test: PantryPal Household Simulation
 * 
 * Tests system performance under load with N concurrent households.
 * 
 * Run: k6 run backend/tests/load/household-load-test.js
 * 
 * Scenarios:
 * - 100 households: Baseline load
 * - 1000 households: Peak load
 * - 10000 households: Stress test
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomItem, randomString } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// ============================================================================
// Configuration
// ============================================================================

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const NUM_HOUSEHOLDS = __ENV.NUM_HOUSEHOLDS || 100;

// Custom metrics
const errorRate = new Rate('errors');
const predictionLatency = new Trend('prediction_latency');
const receiptProcessingLatency = new Trend('receipt_processing_latency');
const apiLatency = new Trend('api_latency');
const wasteEvents = new Counter('waste_events');

// ============================================================================
// Test Options
// ============================================================================

export const options = {
  scenarios: {
    // Scenario 1: Baseline load (100 households)
    baseline: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },    // Ramp up to 100 users
        { duration: '5m', target: 100 },    // Stay at 100 for 5 min
        { duration: '2m', target: 0 },      // Ramp down
      ],
      gracefulRampDown: '30s',
      tags: { scenario: 'baseline' },
    },
    
    // Scenario 2: Peak load (1000 households)
    peak: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '5m', target: 1000 },   // Ramp up to 1000 users
        { duration: '10m', target: 1000 },  // Stay at 1000 for 10 min
        { duration: '5m', target: 0 },      // Ramp down
      ],
      gracefulRampDown: '1m',
      startTime: '10m',
      tags: { scenario: 'peak' },
    },
    
    // Scenario 3: Stress test (10000 households)
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10m', target: 10000 }, // Ramp up to 10k users
        { duration: '5m', target: 10000 },  // Stay at 10k for 5 min
        { duration: '10m', target: 0 },     // Ramp down
      ],
      gracefulRampDown: '2m',
      startTime: '30m',
      tags: { scenario: 'stress' },
    },
  },
  
  thresholds: {
    // API latency: P95 < 500ms, P99 < 1s
    'http_req_duration': ['p(95)<500', 'p(99)<1000'],
    
    // Error rate: < 1%
    'errors': ['rate<0.01'],
    
    // Prediction latency: P95 < 500ms
    'prediction_latency': ['p(95)<500'],
    
    // Receipt processing: P95 < 30s
    'receipt_processing_latency': ['p(95)<30000'],
    
    // Successful requests: > 99%
    'http_req_failed': ['rate<0.01'],
  },
};

// ============================================================================
// Test Data
// ============================================================================

const PRODUCT_NAMES = [
  'Milk', 'Eggs', 'Bread', 'Chicken', 'Apples', 'Bananas', 'Yogurt',
  'Cheese', 'Orange Juice', 'Tomatoes', 'Lettuce', 'Carrots', 'Potatoes',
  'Rice', 'Pasta', 'Ground Beef', 'Salmon', 'Strawberries', 'Butter'
];

const CATEGORIES = ['dairy', 'produce', 'meat', 'bakery', 'pantry'];

const UNITS = ['lb', 'oz', 'gallon', 'unit', 'package'];

// ============================================================================
// Setup & Teardown
// ============================================================================

export function setup() {
  console.log(`Starting load test with ${NUM_HOUSEHOLDS} households`);
  return { baseUrl: BASE_URL };
}

export function teardown(data) {
  console.log('Load test completed');
}

// ============================================================================
// Helper Functions
// ============================================================================

function registerHousehold() {
  const timestamp = Date.now();
  const randomId = Math.random().toString(36).substring(7);
  
  const payload = {
    email: `loadtest-${timestamp}-${randomId}@example.com`,
    password: 'LoadTest123!',
    household_name: `LoadTest Household ${randomId}`,
  };
  
  const response = http.post(`${BASE_URL}/api/v1/auth/register`, JSON.stringify(payload), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  check(response, {
    'registration successful': (r) => r.status === 201,
    'has access token': (r) => r.json('access_token') !== undefined,
  });
  
  errorRate.add(response.status !== 201);
  apiLatency.add(response.timings.duration);
  
  return response.json('access_token');
}

function addPantryItem(token) {
  const payload = {
    name: randomItem(PRODUCT_NAMES),
    category: randomItem(CATEGORIES),
    quantity: Math.random() * 5 + 0.5, // 0.5 to 5.5
    unit: randomItem(UNITS),
    expiration_date: new Date(Date.now() + Math.random() * 14 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
  };
  
  const response = http.post(
    `${BASE_URL}/api/v1/pantry-items`,
    JSON.stringify(payload),
    {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
    }
  );
  
  check(response, {
    'pantry item added': (r) => r.status === 201,
    'has item id': (r) => r.json('id') !== undefined,
  });
  
  errorRate.add(response.status !== 201);
  apiLatency.add(response.timings.duration);
  
  return response.json('id');
}

function getPredictions(token) {
  const startTime = Date.now();
  
  const response = http.get(`${BASE_URL}/api/v1/predictions`, {
    headers: { 'Authorization': `Bearer ${token}` },
  });
  
  const duration = Date.now() - startTime;
  
  check(response, {
    'predictions retrieved': (r) => r.status === 200,
    'predictions is array': (r) => Array.isArray(r.json()),
  });
  
  errorRate.add(response.status !== 200);
  predictionLatency.add(duration);
  apiLatency.add(response.timings.duration);
  
  return response.json();
}

function updateConsumption(token, itemId) {
  const payload = {
    quantity: Math.random() * 2, // Consume random amount
  };
  
  const response = http.patch(
    `${BASE_URL}/api/v1/pantry-items/${itemId}`,
    JSON.stringify(payload),
    {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
    }
  );
  
  check(response, {
    'consumption updated': (r) => r.status === 200,
  });
  
  errorRate.add(response.status !== 200);
  apiLatency.add(response.timings.duration);
}

function markAsWaste(token, itemId) {
  const payload = {
    waste_reason: randomItem(['spoilage', 'overcooked', 'portion', 'packaging', 'other']),
  };
  
  const response = http.post(
    `${BASE_URL}/api/v1/pantry-items/${itemId}/waste`,
    JSON.stringify(payload),
    {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
    }
  );
  
  check(response, {
    'waste recorded': (r) => r.status === 201,
  });
  
  errorRate.add(response.status !== 201);
  wasteEvents.add(1);
  apiLatency.add(response.timings.duration);
}

function getRecipeRecommendations(token) {
  const response = http.get(`${BASE_URL}/api/v1/recipes/recommendations`, {
    headers: { 'Authorization': `Bearer ${token}` },
  });
  
  check(response, {
    'recipes retrieved': (r) => r.status === 200,
    'recipes is array': (r) => Array.isArray(r.json()),
  });
  
  errorRate.add(response.status !== 200);
  apiLatency.add(response.timings.duration);
}

function addToShoppingList(token) {
  const payload = {
    name: randomItem(PRODUCT_NAMES),
    quantity: Math.random() * 3 + 1,
    unit: randomItem(UNITS),
  };
  
  const response = http.post(
    `${BASE_URL}/api/v1/shopping-list`,
    JSON.stringify(payload),
    {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
    }
  );
  
  check(response, {
    'shopping item added': (r) => r.status === 201,
  });
  
  errorRate.add(response.status !== 201);
  apiLatency.add(response.timings.duration);
}

// ============================================================================
// Main Test Scenario
// ============================================================================

export default function() {
  // Simulate daily household activity
  
  group('Household Registration', () => {
    const token = registerHousehold();
    if (!token) return;
    
    // Each virtual user maintains their token
    __VU.token = token;
  });
  
  sleep(1);
  
  group('Pantry Management', () => {
    const token = __VU.token;
    if (!token) return;
    
    // Add 3-5 pantry items (simulating weekly grocery shopping)
    const numItems = Math.floor(Math.random() * 3) + 3;
    const itemIds = [];
    
    for (let i = 0; i < numItems; i++) {
      const itemId = addPantryItem(token);
      if (itemId) itemIds.push(itemId);
      sleep(0.5);
    }
    
    __VU.itemIds = itemIds;
  });
  
  sleep(2);
  
  group('Waste Predictions', () => {
    const token = __VU.token;
    if (!token) return;
    
    // Get predictions (happens automatically after item added)
    const predictions = getPredictions(token);
    
    if (predictions && predictions.length > 0) {
      console.log(`Household has ${predictions.length} predictions`);
    }
  });
  
  sleep(1);
  
  group('Recipe Recommendations', () => {
    const token = __VU.token;
    if (!token) return;
    
    getRecipeRecommendations(token);
  });
  
  sleep(2);
  
  group('Consumption Tracking', () => {
    const token = __VU.token;
    const itemIds = __VU.itemIds || [];
    
    if (!token || itemIds.length === 0) return;
    
    // Update consumption for 1-2 items
    const numUpdates = Math.min(itemIds.length, Math.floor(Math.random() * 2) + 1);
    
    for (let i = 0; i < numUpdates; i++) {
      updateConsumption(token, itemIds[i]);
      sleep(0.5);
    }
  });
  
  sleep(1);
  
  group('Shopping List', () => {
    const token = __VU.token;
    if (!token) return;
    
    // Add 1-2 items to shopping list
    const numItems = Math.floor(Math.random() * 2) + 1;
    
    for (let i = 0; i < numItems; i++) {
      addToShoppingList(token);
      sleep(0.5);
    }
  });
  
  sleep(2);
  
  group('Waste Recording', () => {
    const token = __VU.token;
    const itemIds = __VU.itemIds || [];
    
    if (!token || itemIds.length === 0) return;
    
    // 10% chance an item is wasted
    if (Math.random() < 0.1 && itemIds.length > 0) {
      const itemId = randomItem(itemIds);
      markAsWaste(token, itemId);
    }
  });
  
  // Sleep to simulate time between household activities
  sleep(Math.random() * 5 + 5); // 5-10 seconds
}

// ============================================================================
// Performance Thresholds Summary
// ============================================================================

/*
Expected Results:

100 Households (Baseline):
- P95 latency: < 200ms
- P99 latency: < 500ms
- Error rate: < 0.1%
- Throughput: ~500 req/s

1000 Households (Peak):
- P95 latency: < 500ms
- P99 latency: < 1000ms
- Error rate: < 1%
- Throughput: ~2000 req/s

10000 Households (Stress):
- P95 latency: < 1000ms
- P99 latency: < 2000ms
- Error rate: < 5%
- Throughput: ~5000 req/s

If thresholds fail, investigate:
- Database connection pool exhausted
- Memory leaks in application
- ML model prediction bottleneck
- Cloud Run instance scaling issues
*/
