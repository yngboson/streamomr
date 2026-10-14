// jsc test helpers.  Usage in a runner: load('web/src/cv.js'); load('web/src/aruco.js'); load('web/test/cv/lib.js');
var FX = (typeof arguments !== 'undefined' && arguments[0]) || 'web/test/fixtures';
function readBin(path, T) { var u = readFile(path, 'binary'); return T && T !== Uint8Array ? new T(u.buffer, u.byteOffset, u.byteLength / T.BYTES_PER_ELEMENT) : u; }
function readJSON(path) { return JSON.parse(readFile(path)); }
function now() { return performance.now ? performance.now() : Date.now(); }
function bench(fn, minMs) {                  // ms per call (median of runs)
    for (var wi = 0; wi < 8; wi++) fn(); var runs = [], t0 = now();
    do { var t = now(); fn(); runs.push(now() - t); } while (now() - t0 < (minMs || 300) && runs.length < 200);
    runs.sort(function (a, b) { return a - b; }); return runs[runs.length >> 1];
}
function diffU8(a, b) { var n = a.length, c = 0, mx = 0; for (var i = 0; i < n; i++) { var d = Math.abs(a[i] - b[i]); if (d) { c++; if (d > mx) mx = d; } } return { n: n, bad: c, max: mx }; }
function report(name, ok, extra) { print((ok ? 'PASS ' : 'FAIL ') + name + (extra ? '  ' + extra : '')); }
