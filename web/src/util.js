// Small helpers that reproduce Python / numpy semantics the logic modules rely on.
(function (O) {
'use strict';

// Python round(x, n): correctly rounded on the exact binary value, exact ties go to the even neighbour
// (Number.prototype.toFixed rounds an exact tie up, which differs only there).
function pyRound(x, n) {
  if (!isFinite(x)) return x;
  if (n === undefined) return pyRoundInt(x);
  var s = Math.pow(2, n + 1), y = Math.abs(x) * s;      // x*2^(n+1) is an odd integer  <=>  x*10^n is exactly k+0.5
  if (Number.isInteger(y) && y % 2 === 1) {
    var p = Math.abs(x) * Math.pow(10, n), j = Math.floor(p);
    if (j % 2 === 1) j += 1;
    return Number((x < 0 ? '-' : '') + j + 'e-' + n);
  }
  return Number(x.toFixed(n));
}
// Python round(x) -> integer, half to even
function pyRoundInt(x) {
  var f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

// SHA-1 of a string (UTF-8) as lowercase hex
function sha1hex(str) {
  var bytes = [];
  for (var i = 0; i < str.length; i++) {
    var c = str.codePointAt(i);
    if (c > 0xFFFF) i++;
    if (c < 0x80) bytes.push(c);
    else if (c < 0x800) bytes.push(0xC0 | c >> 6, 0x80 | c & 63);
    else if (c < 0x10000) bytes.push(0xE0 | c >> 12, 0x80 | c >> 6 & 63, 0x80 | c & 63);
    else bytes.push(0xF0 | c >> 18, 0x80 | c >> 12 & 63, 0x80 | c >> 6 & 63, 0x80 | c & 63);
  }
  var bl = bytes.length;
  bytes.push(0x80);
  while (bytes.length % 64 !== 56) bytes.push(0);
  var hi = Math.floor(bl * 8 / 4294967296), lo = (bl * 8) >>> 0;
  bytes.push(hi >>> 24, hi >>> 16 & 255, hi >>> 8 & 255, hi & 255, lo >>> 24, lo >>> 16 & 255, lo >>> 8 & 255, lo & 255);
  var h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0, w = new Array(80);
  function rol(v, s) { return (v << s | v >>> (32 - s)) >>> 0; }
  for (var o = 0; o < bytes.length; o += 64) {
    for (var t = 0; t < 16; t++) w[t] = (bytes[o + 4 * t] << 24 | bytes[o + 4 * t + 1] << 16 | bytes[o + 4 * t + 2] << 8 | bytes[o + 4 * t + 3]) >>> 0;
    for (t = 16; t < 80; t++) w[t] = rol(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1);
    var a = h0, b = h1, c = h2, d = h3, e = h4;
    for (t = 0; t < 80; t++) {
      var f, k;
      if (t < 20) { f = b & c | ~b & d; k = 0x5A827999; }
      else if (t < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
      else if (t < 60) { f = b & c | b & d | c & d; k = 0x8F1BBCDC; }
      else { f = b ^ c ^ d; k = 0xCA62C1D6; }
      var tmp = (rol(a, 5) + (f >>> 0) + e + k + w[t]) >>> 0;
      e = d; d = c; c = rol(b, 30); b = a; a = tmp;
    }
    h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0; h4 = (h4 + e) >>> 0;
  }
  return [h0, h1, h2, h3, h4].map(function (v) { return ('00000000' + v.toString(16)).slice(-8); }).join('');
}

// deep equality of plain JSON-like values (arrays, objects, primitives); NaN equals NaN
function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a === 'number' && typeof b === 'number') return a !== a && b !== b;
  if (a == null || b == null || typeof a !== 'object' || typeof b !== 'object') return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  var ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (var i = 0; i < ka.length; i++) if (!deepEqual(a[ka[i]], b[ka[i]])) return false;
  return true;
}

O.util = { pyRound: pyRound, pyRoundInt: pyRoundInt, sha1hex: sha1hex, deepEqual: deepEqual };
})(globalThis.OMR = globalThis.OMR || {});
