# PantryPal - Testing Guide

## Overview
This guide provides step-by-step instructions to test all the newly implemented ML automation features in PantryPal.

## Prerequisites
- ✅ Backend ML API running on port 5000
- ✅ Frontend Next.js app running on port 9002
- ✅ All 3 ML models loaded successfully
- ✅ User logged in (not guest mode)

## Test Scenarios

### Test 1: Item Canonicalization ✅
**Purpose**: Verify item name normalization works correctly

**Steps**:
1. Click "Add Item" button in Pantry Inventory
2. Enter item name: "organic whole milk"
3. Fill in other fields (quantity: 1, unit: L, category: dairy)
4. Set expiry date (use default 7 days)
5. Click "Add Item"

**Expected Results**:
- Item is added to pantry
- Item name is normalized to "Milk" (canonical form)
- Toast notification shows "Item Added"
- Waste risk score is calculated automatically

**Check in Backend Logs**:
```
POST /canonicalize_item - 200 OK
```

---

### Test 2: Waste Risk Prediction ✅
**Purpose**: Verify ML model calculates waste risk correctly

**Steps**:
1. Add item with expiry date 2 days from today
2. Check the risk badge on the item card
3. Add another item with expiry date 10 days from today
4. Compare risk scores

**Expected Results**:
- Item expiring in 2 days: High risk (red badge)
- Item expiring in 10 days: Low/Medium risk (green/yellow badge)
- Risk score visible on item card
- Dashboard alert banner appears for high-risk items

**Check in Backend Logs**:
```
POST /predict_waste_risk - 200 OK
Response: {"risk_class": "High", "risk_probability": 0.85}
```

---

### Test 3: Automated Waste Alerts 🆕
**Purpose**: Verify proactive waste analysis and recommendations

**Steps**:
1. Add multiple items with different expiry dates
2. Include at least 1 item expiring within 24 hours
3. Observe the top of dashboard

**Expected Results**:
- Alert banner appears at top of dashboard
- Shows list of recommendations:
  - "Urgent: [items] expires within 24 hours!"
  - "Cook recipes using: [items]"
  - "Potential waste: $XX.XX"
- Alert updates automatically when items are added/removed
- Red warning icon visible

**Dashboard UI**:
```
┌─────────────────────────────────────────┐
│ ⚠️ Waste Alert - Action Needed         │
│ • Urgent: Milk expires within 24 hours!│
│ • Cook recipes using: Eggs, Tomatoes   │
│ • Potential waste: $15.00              │
└─────────────────────────────────────────┘
```

---

### Test 4: AI Recipe Recommendations ✅
**Purpose**: Verify ML recipe model provides relevant suggestions

**Steps**:
1. Add items with high waste risk (expiry <3 days)
2. Wait for "AI Recommendations" card to show count
3. Click "Find Recipes to Cook" button
4. Wait for recipes to load

**Expected Results**:
- Loading spinner appears
- Recipes load from ML backend (229K recipe dataset)
- Recipe cards show:
  - Recipe title
  - Ingredients list
  - Prep time and servings
  - Tags (e.g., "quick", "healthy")
- Recipes match at-risk ingredients
- Scrollable list of 5-10 recipes

**Check in Backend Logs**:
```
POST /recommend_recipes - 200 OK
Input: {"ingredients": ["milk", "eggs"], "top_k": 10}
Response: [{"id": "...", "title": "French Toast", ...}]
```

---

### Test 5: Smart Shopping List Suggestions 🆕
**Purpose**: Verify consumption-based shopping list automation

**Steps**:
1. Add items to pantry with realistic quantities
2. Let items consume over time (or manually adjust)
3. Go to Shopping List section
4. Add a few items to shopping list
5. Check for smart suggestions

**Expected Results**:
- Shopping list shows quantity adjustments
- Reasoning provided for each adjustment:
  - "High consumption rate detected - increasing quantity"
  - "Low consumption with waste risk - reducing quantity"
- Suggestions are based on pantry consumption patterns
- Original quantity vs suggested quantity visible

**Example**:
```
Milk: 2L → 3L
Reason: High consumption rate detected - increasing quantity

Bread: 2 packs → 1 pack  
Reason: Low consumption with waste risk - reducing quantity
```

---

### Test 6: Monthly Shopping List Generation 🆕
**Purpose**: Verify 30-day shopping list predictions

**How to Test** (Pending UI button):
- Currently implemented in backend
- Function: `generateMonthlyShoppingList()`
- Needs UI button to trigger
- **TODO**: Add "Generate Monthly List" button to shopping list component

**Expected Logic**:
- Analyzes consumption rate for each pantry item
- Calculates: `dailyConsumption = quantity / shelfLife`
- Projects 30 days: `monthlyQuantity = dailyConsumption * 30`
- Generates complete shopping list for next month

**Test Data Example**:
```
Pantry Item: Milk (1L, purchased 3 days ago, expires in 4 days)
- Total shelf life: 3 + 4 = 7 days
- Daily consumption: 1L / 7 = 0.14L/day
- Monthly need: 0.14 * 30 = 4.2L ≈ 5L
```

---

### Test 7: End-to-End Automation Flow 🆕
**Purpose**: Verify all ML features work together seamlessly

**Complete Workflow**:
1. **Add Items**: Add 5-10 pantry items with varied expiry dates
   - Some expiring soon (2-3 days)
   - Some medium term (7-10 days)
   - Some long term (30+ days)

2. **Observe Automation**:
   - Waste alert banner appears immediately ✅
   - Risk scores calculated automatically ✅
   - Items categorized as High/Medium/Low risk ✅

3. **Get Recipe Recommendations**:
   - Click "Find Recipes to Cook"
   - Verify recipes match at-risk items ✅
   - Recipes load from ML backend ✅

4. **Smart Shopping List**:
   - Add items to shopping list
   - Check for smart suggestions ✅
   - Verify reasoning is logical ✅

5. **Monitor Changes**:
   - Remove a high-risk item
   - Alert banner updates automatically ✅
   - Add new expiring item
   - Alert appears immediately ✅

**Success Criteria**:
- ✅ All features work without manual intervention
- ✅ No errors in browser console
- ✅ Backend logs show successful ML API calls
- ✅ UI is responsive and user-friendly
- ✅ Fallback calculations work if backend unavailable

---

## Performance Benchmarks

### Expected Response Times
- Item canonicalization: <10ms
- Waste risk prediction: 2-5ms
- Recipe recommendations: 50-100ms
- Shopping list analysis: <20ms
- Dashboard waste analysis: <30ms

### Backend Health Check
Open: `http://localhost:5000/health`

**Expected Response**:
```json
{
  "status": "healthy",
  "models_loaded": {
    "canonicalization": true,
    "recipe_recommendation": true,
    "waste_predictor": true
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

---

## Troubleshooting

### Issue: Recipes Not Loading
**Symptoms**: Clicking "Find Recipes" button does nothing or shows error

**Check**:
1. Browser console for errors
2. Network tab - verify `/api/recommend-recipes` call succeeds
3. Backend logs - check for `/recommend_recipes` endpoint

**Solution**:
- Ensure backend is running: `http://localhost:5000/health`
- Check CORS configuration in backend
- Verify recipe model is loaded (229K recipes)

---

### Issue: Waste Alert Not Appearing
**Symptoms**: No alert banner at top of dashboard

**Check**:
1. Do you have items with high waste risk?
2. Are items expiring within 24 hours?
3. Browser console for errors

**Solution**:
- Add item expiring in 1-2 days
- Check that `analyzeWasteRisk()` is called in useEffect
- Verify `wasteAnalysis` state is populated

---

### Issue: Item Canonicalization Not Working
**Symptoms**: Item names not normalized

**Check**:
1. Network tab - `/api/canonicalize-item` call
2. Backend logs for POST `/canonicalize_item`
3. Item name input in form

**Solution**:
- Verify backend model loaded (317K products)
- Check API route: `src/app/api/canonicalize-item/route.ts`
- Test with common item: "milk", "eggs", "bread"

---

### Issue: Backend Not Responding
**Symptoms**: All ML features failing, fallback calculations used

**Check**:
1. Is backend running? Check terminal
2. Port 5000 accessible? Try `http://localhost:5000/health`
3. Models loaded? Check backend startup logs

**Solution**:
- Restart backend: `cd backend/api && python app.py`
- Verify all 3 models load (takes ~11 seconds)
- Check for memory issues (models are 1.26 GB)

---

## Next.js API Routes Testing

### Test API Routes Directly

**1. Predict Waste Risk**
```bash
curl -X POST http://localhost:9002/api/predict-waste-risk \
  -H "Content-Type: application/json" \
  -d '{
    "item_name": "Milk",
    "category": "dairy",
    "quantity": 1,
    "days_until_expiry": 2,
    "purchase_date": "2025-01-13",
    "storage_location": "refrigerator"
  }'
```

**Expected Response**:
```json
{
  "risk_class": "High",
  "risk_probability": 0.87
}
```

---

**2. Canonicalize Item**
```bash
curl -X POST http://localhost:9002/api/canonicalize-item \
  -H "Content-Type: application/json" \
  -d '{
    "item_name": "organic whole milk"
  }'
```

**Expected Response**:
```json
{
  "canonical_name": "Milk",
  "confidence": 0.95
}
```

---

**3. Recommend Recipes**
```bash
curl -X POST http://localhost:9002/api/recommend-recipes \
  -H "Content-Type: application/json" \
  -d '{
    "ingredients": ["milk", "eggs", "flour"],
    "dietary_preferences": [],
    "top_k": 5
  }'
```

**Expected Response**:
```json
{
  "recipes": [
    {
      "id": "recipe-123",
      "title": "Pancakes",
      "ingredients": ["milk", "eggs", "flour", "sugar"],
      "prep_time": "15 minutes",
      "servings": "4"
    },
    ...
  ]
}
```

---

## Browser Console Checks

### Successful Operation
```
✅ POST /api/predict-waste-risk 200 OK (45ms)
✅ POST /api/canonicalize-item 200 OK (8ms)
✅ POST /api/recommend-recipes 200 OK (67ms)
```

### With Fallback
```
⚠️ Failed to calculate waste risk - using fallback calculation
ℹ️ Using Basic Risk Calculation
```

### Error Case
```
❌ POST /api/predict-waste-risk 500 Internal Server Error
❌ TypeError: Failed to fetch
```

---

## Test Checklist Summary

- [ ] Item canonicalization normalizes names correctly
- [ ] Waste risk scores calculated by ML model
- [ ] Automated waste alerts appear on dashboard
- [ ] Recipe recommendations load from ML backend
- [ ] Smart shopping list shows consumption-based suggestions
- [ ] Monthly shopping list generation works (needs UI button)
- [ ] All features update automatically on pantry changes
- [ ] Fallback calculations work when backend unavailable
- [ ] No errors in browser console
- [ ] Backend logs show successful API calls
- [ ] Response times <100ms for all operations
- [ ] UI is responsive and user-friendly

---

## Report Issues

If you encounter any issues during testing:

1. **Check Browser Console**: Look for JavaScript errors
2. **Check Network Tab**: Verify API calls succeed
3. **Check Backend Logs**: Look for Python errors or slow queries
4. **Check This Guide**: Follow troubleshooting steps

**Common Fixes**:
- Restart both frontend and backend
- Clear browser cache
- Verify environment variables
- Check port availability (5000, 9002)

---

**Last Updated**: January 2025
**Version**: v1.0 - Full ML Automation
**Status**: ✅ Ready for Testing
