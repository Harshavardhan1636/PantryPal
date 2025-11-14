// Polyfill for buffer-equal-constant-time compatibility with Node.js 23+
// This fixes the "Cannot read properties of undefined (reading 'prototype')" error

if (typeof Buffer !== 'undefined' && !Buffer.prototype) {
  // @ts-ignore
  Buffer.prototype = Object.create(Uint8Array.prototype);
}
