// Server-only export that prevents client-side access
export * from './genkit';

// This file should only be imported in server-side code
if (typeof window !== 'undefined') {
  throw new Error(
    'AI modules cannot be imported in client-side code. ' +
    'Use server actions from @/lib/actions instead.'
  );
}
