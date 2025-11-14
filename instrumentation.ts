// Node.js 23+ Buffer polyfill for buffer-equal-constant-time compatibility
// This must run before any other imports

if (typeof Buffer !== 'undefined' && typeof Buffer.prototype === 'undefined') {
  Object.defineProperty(Buffer, 'prototype', {
    value: Object.create(Uint8Array.prototype),
    writable: true,
    enumerable: false,
    configurable: true
  });
}

export async function register() {
  // Polyfill is already applied above
}
