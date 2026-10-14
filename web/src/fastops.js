(function (O) {
'use strict';
// fastops.js - omr/fastops.py.  The Python file exists to dodge numpy call overhead; here only the numerical contract matters:
// pctRows reproduces `np.percentile(a, q, axis=1)` for float32 rows (float32 arithmetic, method 'linear').
// (nanmedian0 is used only by omr/fusion.py, which J2 ported.)
var f32 = Math.fround;

// quickselect: rearranges a[0..n) so that a[k] is the k-th smallest and returns it; everything right of k is >= a[k]
function selectK(a, n, k) {
    var l = 0, r = n - 1;
    while (l < r) {
        var pv = a[(l + r) >> 1], i = l, j = r;
        while (i <= j) {
            while (a[i] < pv) i++;
            while (a[j] > pv) j--;
            if (i <= j) { var t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
        }
        if (k <= j) r = j; else if (k >= i) l = i; else break;
    }
    return a[k];
}

// omr/fastops.py:10.  a: Float32Array row-major (n rows x m cols) -> Float32Array(n)
function pctRows(a, n, m, q) {
    var vi = (m - 1) * (q / 100), lo = Math.floor(vi), hi = lo + 1 < m ? lo + 1 : lo, t = vi - lo;
    var out = new Float32Array(n), row = new Float32Array(m);
    for (var r = 0; r < n; r++) {
        for (var j = 0; j < m; j++) row[j] = a[r * m + j];
        var x = selectK(row, m, lo), y = x, d, o;       // order statistics lo and hi without a full sort
        if (hi !== lo) { y = Infinity; for (j = lo + 1; j < m; j++) if (row[j] < y) y = row[j]; }
        d = f32(y - x); o = f32(x + f32(d * t));
        if (t >= 0.5) o = f32(y - f32(d * (1 - t)));
        out[r] = o;
    }
    return out;
}

O.fastops = { pctRows: pctRows, selectK: selectK };
})(globalThis.OMR = globalThis.OMR || {});
