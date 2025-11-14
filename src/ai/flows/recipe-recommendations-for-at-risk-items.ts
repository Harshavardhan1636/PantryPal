'use server';

/**
 * @fileOverview Recommends recipes based on at-risk items in the user's pantry.
 *
 * - getRecipeRecommendations - A function that returns recipe recommendations for at-risk items.
 * - RecipeRecommendationsInput - The input type for the getRecipeRecommendations function.
 * - RecipeRecommendationsOutput - The return type for the getRecipeRecommendations function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const RecipeRecommendationsInputSchema = z.object({
  atRiskItems: z
    .array(z.string())
    .describe('An array of at-risk items in the user pantry.'),
  dietaryPreferences: z
    .string()
    .optional()
    .describe('The dietary preferences of the user.'),
});
export type RecipeRecommendationsInput = z.infer<
  typeof RecipeRecommendationsInputSchema
>;

const RecipeRecommendationsOutputSchema = z.object({
  recipes: z.array(
    z.object({
      id: z.string().describe('The id of the recipe.'),
      title: z.string().describe('The title of the recipe.'),
      ingredients: z.array(z.string()).describe('The ingredients of the recipe.'),
      steps: z.string().describe('The preparation steps of the recipe.'),
      prep_time: z.string().describe('The preparation time of the recipe.'),
      servings: z.string().describe('The number of servings of the recipe.'),
      tags: z.array(z.string()).describe('The tags of the recipe.'),
      source: z.string().describe('The source of the recipe.'),
    })
  ),
});
export type RecipeRecommendationsOutput = z.infer<
  typeof RecipeRecommendationsOutputSchema
>;

export async function getRecipeRecommendations(
  input: RecipeRecommendationsInput
): Promise<RecipeRecommendationsOutput> {
  return recipeRecommendationsFlow(input);
}

const prompt = ai.definePrompt({
  name: 'recipeRecommendationsPrompt',
  input: {schema: RecipeRecommendationsInputSchema},
  output: {schema: RecipeRecommendationsOutputSchema},
  prompt: `You are a recipe recommendation engine. Given a list of at-risk items, recommend recipes that use those items.

      At-risk items: {{atRiskItems}}
      Dietary preferences: {{dietaryPreferences}}

      Return a JSON array of recipes, with each recipe having the following fields:
      - id: The id of the recipe.
      - title: The title of the recipe.
      - ingredients: The ingredients of the recipe.
      - steps: The preparation steps of the recipe.
      - prep_time: The preparation time of the recipe.
      - servings: The number of servings of the recipe.
      - tags: The tags of the recipe.
      - source: The source of the recipe.
      `,
});

const recipeRecommendationsFlow = ai.defineFlow(
  {
    name: 'recipeRecommendationsFlow',
    inputSchema: RecipeRecommendationsInputSchema,
    outputSchema: RecipeRecommendationsOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
