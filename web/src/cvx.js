(function (O) {
'use strict';
// cvx.js - what is left of the interim primitives after the merge into cv.js: numpy-style percentile / median / mean, a tiny least-squares
// solver, a 1-row Gaussian, and the bilinear single-sample helper used by tracks.rectify.  Same conventions as cv.js: grey image = {w, h, data}.
// LEAD: if cv.js gains the same functions (resize, gaussianBlur, dilate, erode, morphOpen), cvx can be merged into it.

var cv = O.cv;
var cvx = {};
var f32 = Math.fround;
var rint = cv.rint;

var pool = {};
function scratch(name, T, n) {
    var b = pool[name];
    if (!b || b.length < n || b.constructor !== T) b = pool[name] = new T(n);
    return b;
}
function mk(w, h, T) { return { w: w, h: h, data: new (T || Uint8Array)(w * h) }; }
cvx.mk = mk;
function store(T) {            // value -> typed array element (uint8 saturates + rounds)
    return T === Uint8Array;
}

function gaussKernel(ksize, sigma) {
    var k = new Float64Array(ksize), c = (ksize - 1) / 2, s = 0, i;
    for (i = 0; i < ksize; i++) { var d = i - c; k[i] = Math.exp(-d * d / (2 * sigma * sigma)); s += k[i]; }
    for (i = 0; i < ksize; i++) k[i] /= s;
    return k;
}
// 1-D blur of a Float32Array row (cv2.GaussianBlur(row.reshape(1, -1), (0, 0), sigma)): float kernel, reflect101 along x
cvx.gaussianBlurRow = function cvx_gaussianBlurRow(a, sigma) {
    var n = a.length, ks = rint(sigma * 8 + 1) | 1, r = ks >> 1, kern = gaussKernel(ks, sigma), out = new Float32Array(n), x, i;
    for (x = 0; x < n; x++) {
        var s = 0;
        for (i = 0; i < ks; i++) s += kern[i] * a[cv.reflect101(x - r + i, n)];
        out[x] = s;
    }
    return out;
};

// ------------------------------------------------------------------ numpy-style helpers
// np.percentile(a, q) (method 'linear') of any array-like of numbers; f32: float32 arithmetic (float32 input arrays)
cvx.percentile = function cvx_percentile(a, q, f32mode) {
    var n = a.length, s = Float64Array.from(a);
    s.sort();
    var vi = (n - 1) * (q / 100), lo = Math.floor(vi), hi = lo + 1 < n ? lo + 1 : lo, t = vi - lo;
    var x = s[lo], y = s[hi];
    if (f32mode) {
        var d = f32(y - x), o = f32(x + f32(d * t));
        if (t >= 0.5) o = f32(y - f32(d * (1 - t)));
        return o;
    }
    var d2 = y - x, o2 = x + d2 * t;
    if (t >= 0.5) o2 = y - d2 * (1 - t);
    return o2;
};
cvx.median = function cvx_median(a) { return cvx.percentile(a, 50); };
cvx.mean = function cvx_mean(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s / a.length; };

// least squares  min |A x - b|  (A: array of rows, b: array of vectors/numbers; multiple right-hand sides = array of arrays) via normal equations
cvx.lstsq = function cvx_lstsq(A, B) {
    var m = A.length, n = A[0].length, nr = B[0].length === undefined ? 1 : B[0].length, i, j, k, c;
    var out = [];
    for (c = 0; c < nr; c++) {
        var AtA = new Float64Array(n * n), Atb = new Float64Array(n);
        for (i = 0; i < m; i++) {
            var bi = nr === 1 && B[0].length === undefined ? B[i] : B[i][c];
            for (j = 0; j < n; j++) { Atb[j] += A[i][j] * bi; for (k = 0; k < n; k++) AtA[j * n + k] += A[i][j] * A[i][k]; }
        }
        out.push(cv.solveLinear(AtA, Atb, n));
    }
    return nr === 1 && B[0].length === undefined ? out[0] : out;
};

function sampleF(src, w, h, fx, fy, border, cval, isU8) {
    var x0 = Math.floor(fx), y0 = Math.floor(fy), ax = fx - x0, ay = fy - y0, a, b, c, d;
    if (x0 >= 0 && y0 >= 0 && x0 < w - 1 && y0 < h - 1) {
        var p = y0 * w + x0; a = src[p]; b = src[p + 1]; c = src[p + w]; d = src[p + w + 1];
    } else if (border === 'constant') {
        if (x0 >= w || x0 + 1 < 0 || y0 >= h || y0 + 1 < 0) return cval;
        var x1 = x0 + 1, y1 = y0 + 1, okx0 = x0 >= 0 && x0 < w, okx1 = x1 >= 0 && x1 < w, oky0 = y0 >= 0 && y0 < h, oky1 = y1 >= 0 && y1 < h;
        a = okx0 && oky0 ? src[y0 * w + x0] : cval; b = okx1 && oky0 ? src[y0 * w + x1] : cval;
        c = okx0 && oky1 ? src[y1 * w + x0] : cval; d = okx1 && oky1 ? src[y1 * w + x1] : cval;
    } else {
        var xa = x0 < 0 ? 0 : x0 >= w ? w - 1 : x0, xb = x0 + 1 < 0 ? 0 : x0 + 1 >= w ? w - 1 : x0 + 1;
        var ya = y0 < 0 ? 0 : y0 >= h ? h - 1 : y0, yb = y0 + 1 < 0 ? 0 : y0 + 1 >= h ? h - 1 : y0 + 1;
        a = src[ya * w + xa]; b = src[ya * w + xb]; c = src[yb * w + xa]; d = src[yb * w + xb];
    }
    var v = (a * (1 - ax) + b * ax) * (1 - ay) + (c * (1 - ax) + d * ax) * ay;
    return isU8 ? Math.floor(v + 0.5) : v;
}
cvx.sample = sampleF;
cvx.sample = sampleF;

O.cvx = cvx;
})(globalThis.OMR = globalThis.OMR || {});
