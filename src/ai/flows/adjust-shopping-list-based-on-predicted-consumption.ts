'use server';

/**
 * @fileOverview Adjusts the shopping list based on predicted consumption to avoid overbuying.
 *
 * - adjustShoppingList - A function that takes the current pantry state and predicted consumption to suggest adjustments to the shopping list.
 * - AdjustShoppingListInput - The input type for the adjustShoppingList function, including the pantry and shopping list.
 * - AdjustShoppingListOutput - The return type for the adjustShoppingList function, providing a list of items to avoid buying.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

// Define the schema for a pantry item
const PantryItemSchema = z.object({
  itemId: z.string().describe('The ID of the item in the pantry.'),
  name: z.string().describe('The name of the item.'),
  quantity: z.number().describe('The quantity of the item currently in the pantry.'),
  unit: z.string().describe('The unit of measurement for the quantity (e.g., kg, lb, piece).'),
  predictedConsumption: z.number().describe('The predicted consumption of the item over the next week.'),
  riskScore: z.number().describe('The risk score of the item expiring.'),
});

// Define the schema for a shopping list item
const ShoppingListItemSchema = z.object({
  itemId: z.string().describe('The ID of the item in the catalog.'),
  name: z.string().describe('The name of the item.'),
  quantity: z.number().describe('The quantity of the item to buy.'),
  unit: z.string().describe('The unit of measurement for the quantity (e.g., kg, lb, piece).'),
});

// Define the input schema for the adjustShoppingList function
const AdjustShoppingListInputSchema = z.object({
  pantry: z.array(PantryItemSchema).describe('The current state of the user\u0027s pantry.'),
  shoppingList: z.array(ShoppingListItemSchema).describe('The user\u0027s current shopping list.'),
  householdSize: z.number().describe('The number of people in the household.'),
});
export type AdjustShoppingListInput = z.infer<typeof AdjustShoppingListInputSchema>;

// Define the output schema for the adjustShoppingList function
const AdjustShoppingListOutputSchema = z.array(
  z.object({
    itemId: z.string().describe('The ID of the item.'),
    name: z.string().describe('The name of the item to avoid buying.'),
    quantityToReduce: z.number().describe('The quantity of the item to reduce from the shopping list.'),
    unit: z.string().describe('The unit of measurement for the quantity.'),
    reason: z.string().describe('The reason for the adjustment.'),
  })
);
export type AdjustShoppingListOutput = z.infer<typeof AdjustShoppingListOutputSchema>;

// Define the prompt that will be used to adjust the shopping list
const adjustShoppingListPrompt = ai.definePrompt({
  name: 'adjustShoppingListPrompt',
  input: {schema: AdjustShoppingListInputSchema},
  output: {schema: AdjustShoppingListOutputSchema},
  prompt: `You are a helpful shopping assistant that analyzes a user's pantry and shopping list to suggest adjustments that prevent food waste and save money.

Analyze the following pantry items and shopping list, and determine if any items on the shopping list should be reduced in quantity, based on the predicted consumption of items in the pantry.

Pantry:
{{#each pantry}}
- {{name}} ({{quantity}} {{unit}}), Predicted Consumption: {{predictedConsumption}} {{unit}}, Risk Score: {{riskScore}}
{{/each}}

Shopping List:
{{#each shoppingList}}
- {{name}} ({{quantity}} {{unit}})
{{/each}}

Consider the household size: {{householdSize}} people.

For each item on the shopping list that can be reduced, provide the item ID, name, quantity to reduce, unit, and a brief reason. Only include items that should be reduced, do not include items that the user should buy as is.

Format your response as a JSON array of objects with the following fields:
- itemId: The ID of the item.
- name: The name of the item to avoid buying.
- quantityToReduce: The quantity of the item to reduce from the shopping list.
- unit: The unit of measurement for the quantity.
- reason: A brief reason for the adjustment.

If no adjustments are necessary, return an empty array.

Output: `,
});

// Define the Genkit flow for adjusting the shopping list
const adjustShoppingListFlow = ai.defineFlow(
  {
    name: 'adjustShoppingListFlow',
    inputSchema: AdjustShoppingListInputSchema,
    outputSchema: AdjustShoppingListOutputSchema,
  },
  async input => {
    const {output} = await adjustShoppingListPrompt(input);
    return output!;
  }
);

/**
 * Adjusts the shopping list based on predicted consumption.
 * @param input The input for adjusting the shopping list.
 * @returns The adjusted shopping list.
 */
export async function adjustShoppingList(input: AdjustShoppingListInput): Promise<AdjustShoppingListOutput> {
  return adjustShoppingListFlow(input);
}
