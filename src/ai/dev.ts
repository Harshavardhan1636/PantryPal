import { config } from 'dotenv';
config();

import '@/ai/flows/recipe-recommendations-for-at-risk-items.ts';
import '@/ai/flows/waste-risk-prediction.ts';
import '@/ai/flows/adjust-shopping-list-based-on-predicted-consumption.ts';
