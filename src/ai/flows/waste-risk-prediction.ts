'use server';

/**
 * @fileOverview Predicts the risk of food waste for each item in the pantry.
 *
 * - predictWasteRisk - Predicts the risk of food waste for an item.
 * - WasteRiskPredictionInput - The input type for the predictWasteRisk function.
 * - WasteRiskPredictionOutput - The return type for the predictWasteRisk function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const WasteRiskPredictionInputSchema = z.object({
  itemId: z.string().describe('The ID of the item.'),
  category: z.string().describe('The category of the item.'),
  purchaseDate: z.string().describe('The date the item was purchased (YYYY-MM-DD).'),
  openedDate: z.string().optional().describe('The date the item was opened (YYYY-MM-DD), if applicable.'),
  quantity: z.number().describe('The quantity of the item.'),
  householdSize: z.number().describe('The size of the household.'),
  storageType: z.string().describe('The storage type of the item (e.g., refrigerator, pantry, freezer).'),
  typicalShelfLifeDays: z.number().describe('The typical shelf life of the item in days.'),
  price: z.number().optional().describe('The price of the item.'),
  promotionsFlag: z.boolean().optional().describe('Whether the item was purchased on promotion.'),
});
export type WasteRiskPredictionInput = z.infer<typeof WasteRiskPredictionInputSchema>;

const WasteRiskPredictionOutputSchema = z.object({
  riskScore: z.number().describe('The risk score (0-1) of the item being wasted.'),
  riskClass: z.enum(['High', 'Medium', 'Low']).describe('The risk class of the item being wasted.'),
  predictedExpiryDate: z.string().optional().describe('The predicted expiry date of the item (YYYY-MM-DD).'),
});
export type WasteRiskPredictionOutput = z.infer<typeof WasteRiskPredictionOutputSchema>;

export async function predictWasteRisk(input: WasteRiskPredictionInput): Promise<WasteRiskPredictionOutput> {
  return predictWasteRiskFlow(input);
}

const predictWasteRiskPrompt = ai.definePrompt({
  name: 'predictWasteRiskPrompt',
  input: {schema: WasteRiskPredictionInputSchema},
  output: {schema: WasteRiskPredictionOutputSchema},
  prompt: `You are an AI assistant that predicts the risk of food waste for items in a household pantry.

  Based on the provided information, determine the riskScore (0-1) and riskClass (High, Medium, Low) for the item being wasted.
  Also estimate the predictedExpiryDate if possible. Take into account all available data to make an informed decision.

  Item ID: {{{itemId}}}
  Category: {{{category}}}
  Purchase Date: {{{purchaseDate}}}
  Opened Date: {{#if openedDate}}{{{openedDate}}}{{else}}Not opened{{/if}}
  Quantity: {{{quantity}}}
  Household Size: {{{householdSize}}}
  Storage Type: {{{storageType}}}
  Typical Shelf Life (days): {{{typicalShelfLifeDays}}}
  Price: {{#if price}}{{{price}}}{{else}}Unknown{{/if}}
  Promotions Flag: {{#if promotionsFlag}}Yes{{else}}No{{/if}}
  \n
  Provide your response as a JSON object conforming to the following schema:
  {
    "riskScore": number,
    "riskClass": "High" | "Medium" | "Low",
    "predictedExpiryDate": string (YYYY-MM-DD) | null
  }
  `,
});

const predictWasteRiskFlow = ai.defineFlow(
  {
    name: 'predictWasteRiskFlow',
    inputSchema: WasteRiskPredictionInputSchema,
    outputSchema: WasteRiskPredictionOutputSchema,
  },
  async input => {
    const {output} = await predictWasteRiskPrompt(input);
    return output!;
  }
);
