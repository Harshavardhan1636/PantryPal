import {genkit} from 'genkit';
import {googleAI} from '@genkit-ai/google-genai';

// Ensure this only runs on the server
if (typeof window !== 'undefined') {
  throw new Error('Genkit cannot be initialized in the browser. This module should only be imported in server-side code.');
}

export const ai = genkit({
  plugins: [googleAI()],
  model: 'googleai/gemini-2.5-flash',
});
