(function (O) {
'use strict';
// cv.js - image primitives matching OpenCV (cv2 5.0) semantics.  Grey image = {w, h, data} (Uint8Array | Float32Array | Float64Array).
// Every function carries the cv2 call it stands for; rounding is cvRound (half to even), uint8 results saturate.

var cv = {};
var LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;      // the 32-bit word paths (adaptiveWords) pick bytes out of Int32 views

// ------------------------------------------------------------------ helpers
function rint(x) {                       // cvRound: round half to even
    var r = Math.floor(x + 0.5);
    if (r - x === 0.5 && (r & 1)) r -= 1;
    return r;
}
cv.rint = rint;
function sat8(x) { return x < 0 ? 0 : x > 255 ? 255 : x; }
function mat(w, h, T) { T = T || Uint8Array; return { w: w, h: h, data: new T(w * h) }; }
cv.mat = mat;
var f32 = Math.fround;

// scratch pool: hot paths reuse buffers across calls (no allocation per frame once warm)
var pool = {};
function scratch(name, T, n) {
    var b = pool[name];
    if (!b || b.length < n || b.constructor !== T) b = pool[name] = new T(n);
    return b;
}

// BORDER_REFLECT_101 index (cv::borderInterpolate)
function reflect101(p, len) {
    if (len === 1) return 0;
    while (p < 0 || p >= len) {
        if (p < 0) p = -p;
        else p = 2 * (len - 1) - p;
    }
    return p;
}
cv.reflect101 = reflect101;

// ------------------------------------------------------------------ colour
// cv2.cvtColor BGR2GRAY / RGB2GRAY (cv2 5.0 measured: 15-bit fixed point, B 3735, G 19235, R 9798, round 2^14)
cv.toGray = function cv_toGray(data, w, h, channels, order) {      // order 'bgr' | 'rgb' (alpha ignored)
    var out = new Uint8Array(w * h), n = w * h, cB = 3735, cG = 19235, cR = 9798, ro = order === 'rgb' ? 0 : 2, bo = 2 - ro;
    for (var i = 0, j = 0; i < n; i++, j += channels)
        out[i] = (data[j + bo] * cB + data[j + 1] * cG + data[j + ro] * cR + 16384) >> 15;
    return { w: w, h: h, data: out };
};

cv.bitwiseNot = function cv_bitwiseNot(img) {
    var n = img.data.length, o = new Uint8Array(n), d = img.data;
    for (var i = 0; i < n; i++) o[i] = 255 - d[i];
    return { w: img.w, h: img.h, data: o };
};

cv.countNonZero = function cv_countNonZero(img) { var c = 0, d = img.data; for (var i = 0; i < d.length; i++) if (d[i] !== 0) c++; return c; };

// cv2.addWeighted(a, alpha, b, beta, gamma) -> uint8 (or float for float input)
cv.addWeighted = function cv_addWeighted(a, alpha, b, beta, gamma) {
    var n = a.data.length, T = a.data.constructor, o = new T(n), A = a.data, B = b.data, u8 = T === Uint8Array;
    for (var i = 0; i < n; i++) { var v = A[i] * alpha + B[i] * beta + gamma; o[i] = u8 ? sat8(rint(v)) : v; }
    return { w: a.w, h: a.h, data: o };
};

// cv2.divide(a, b, scale=s, dtype=CV_32F) on uint8 images: s*a/b, 0 where b == 0
cv.divide = function cv_divide(a, b, scale) {
    var n = a.data.length, o = new Float32Array(n), A = a.data, B = b.data;
    for (var i = 0; i < n; i++) o[i] = B[i] === 0 ? 0 : f32(f32(scale * A[i]) / B[i]);
    return { w: a.w, h: a.h, data: o };
};

// ------------------------------------------------------------------ statistics
// cv2.meanStdDev on a sub-rectangle of a uint8 image -> {mean, std}
cv.meanStdDev = function cv_meanStdDev(img, x0, y0, x1, y1) {
    var s = 0, s2 = 0, n = 0, d = img.data, w = img.w;
    if (x0 === undefined) { x0 = 0; y0 = 0; x1 = img.w; y1 = img.h; }
    for (var y = y0; y < y1; y++) for (var i = y * w + x0, e = y * w + x1; i < e; i++) { var v = d[i]; s += v; s2 += v * v; n++; }
    var m = s / n, v2 = s2 / n - m * m;
    return { mean: m, std: Math.sqrt(v2 > 0 ? v2 : 0) };
};

// cv2.threshold(..., THRESH_OTSU) threshold of an 8-bit image (getThreshVal_Otsu_8u)
cv.otsuThreshold = function cv_otsuThreshold(d, n) {
    var h = scratch('otsuHist', Int32Array, 256); h.fill(0);
    for (var i = 0; i < n; i++) h[d[i]]++;
    var scale = 1 / n, mu = 0, k;
    for (k = 0; k < 256; k++) mu += k * h[k] * scale;
    var mu1 = 0, q1 = 0, maxSigma = 0, maxVal = 0, FLT_EPS = 1.1920928955078125e-7;
    for (k = 0; k < 256; k++) {
        var p = h[k] * scale; mu1 *= q1; q1 += p; var q2 = 1 - q1;
        if (Math.min(q1, q2) < FLT_EPS || Math.max(q1, q2) > 1 - FLT_EPS) continue;
        mu1 = (mu1 + k * p) / q1;
        var mu2 = (mu - q1 * mu1) / q2, sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        if (sigma > maxSigma) { maxSigma = sigma; maxVal = k; }
    }
    return maxVal;
};

// cv2.threshold: type 'binary' | 'binary_inv'; thresh 'otsu' computes it.  Returns {t, bin}
cv.threshold = function cv_threshold(img, thresh, maxval, type) {
    if (maxval === undefined) maxval = 255;
    var n = img.data.length, d = img.data, t = thresh === 'otsu' ? cv.otsuThreshold(d, n) : thresh, o = new Uint8Array(n);
    var inv = type === 'binary_inv';
    for (var i = 0; i < n; i++) o[i] = (d[i] > t) !== inv ? maxval : 0;
    return { t: t, bin: { w: img.w, h: img.h, data: o } };
};
cv.otsu = function cv_otsu(img, inv) { return cv.threshold(img, 'otsu', 255, inv ? 'binary_inv' : 'binary'); };

// ------------------------------------------------------------------ box filter / adaptive threshold / integral
// cv2.boxFilter(img, -1, (kx, ky), normalize, borderType=BORDER_REPLICATE), anchor at the centre.  uint8 -> uint8 (mean rounded half up,
// as cv2 measured), float -> float.  Separable: running column sums, then a sliding horizontal window over a replicate-padded row.
// (one copy per element type: JavaScriptCore de-optimises loops that see several typed-array kinds)
function boxFilter_U8(img, kx, normalize, ky) {
    var w = img.w, h = img.h, d = img.data;
    var out = mat(w, h, Uint8Array), o = out.data, ax = kx >> 1, ay = ky >> 1, scale = 1 / (kx * ky), x, y, t;
    var P = scratch('boxP', Float64Array, w + kx), cs = scratch('boxC', Float64Array, w);
    cs.fill(0);
    for (y = -ay; y <= ky - 1 - ay; y++) { var rb = (y < 0 ? 0 : y >= h ? h - 1 : y) * w; for (x = 0; x < w; x++) cs[x] += d[rb + x]; }
    for (y = 0; y < h; y++) {
        if (y > 0) {
            var nr = y + ky - 1 - ay, orow = y - 1 - ay, nb = (nr >= h ? h - 1 : nr) * w, ob = (orow < 0 ? 0 : orow) * w;
            for (x = 0; x < w; x++) cs[x] += d[nb + x] - d[ob + x];
        }
        var l = cs[0], r = cs[w - 1];
        for (t = 0; t < ax; t++) P[t] = l;
        for (x = 0; x < w; x++) P[ax + x] = cs[x];
        for (t = ax + w; t < w + kx - 1; t++) P[t] = r;
        var sum = 0, b = y * w;
        for (t = 0; t < kx; t++) sum += P[t];
        for (x = 0; x < w; x++) {
            if (x > 0) sum += P[x + kx - 1] - P[x - 1];
            o[b + x] = normalize ? sat8(Math.floor(sum * scale + 0.5)) : sat8(sum);
        }
    }
    return out;
}
function boxFilter_F32(img, kx, normalize, ky) {
    var w = img.w, h = img.h, d = img.data;
    var out = mat(w, h, Float32Array), o = out.data, ax = kx >> 1, ay = ky >> 1, scale = 1 / (kx * ky), x, y, t;
    var P = scratch('boxP', Float64Array, w + kx), cs = scratch('boxC', Float64Array, w);
    cs.fill(0);
    for (y = -ay; y <= ky - 1 - ay; y++) { var rb = (y < 0 ? 0 : y >= h ? h - 1 : y) * w; for (x = 0; x < w; x++) cs[x] += d[rb + x]; }
    for (y = 0; y < h; y++) {
        if (y > 0) {
            var nr = y + ky - 1 - ay, orow = y - 1 - ay, nb = (nr >= h ? h - 1 : nr) * w, ob = (orow < 0 ? 0 : orow) * w;
            for (x = 0; x < w; x++) cs[x] += d[nb + x] - d[ob + x];
        }
        var l = cs[0], r = cs[w - 1];
        for (t = 0; t < ax; t++) P[t] = l;
        for (x = 0; x < w; x++) P[ax + x] = cs[x];
        for (t = ax + w; t < w + kx - 1; t++) P[t] = r;
        var sum = 0, b = y * w;
        for (t = 0; t < kx; t++) sum += P[t];
        for (x = 0; x < w; x++) {
            if (x > 0) sum += P[x + kx - 1] - P[x - 1];
            o[b + x] = normalize ? sum * scale : sum;
        }
    }
    return out;
}
function boxFilter_F64(img, kx, normalize, ky) {
    var w = img.w, h = img.h, d = img.data;
    var out = mat(w, h, Float64Array), o = out.data, ax = kx >> 1, ay = ky >> 1, scale = 1 / (kx * ky), x, y, t;
    var P = scratch('boxP', Float64Array, w + kx), cs = scratch('boxC', Float64Array, w);
    cs.fill(0);
    for (y = -ay; y <= ky - 1 - ay; y++) { var rb = (y < 0 ? 0 : y >= h ? h - 1 : y) * w; for (x = 0; x < w; x++) cs[x] += d[rb + x]; }
    for (y = 0; y < h; y++) {
        if (y > 0) {
            var nr = y + ky - 1 - ay, orow = y - 1 - ay, nb = (nr >= h ? h - 1 : nr) * w, ob = (orow < 0 ? 0 : orow) * w;
            for (x = 0; x < w; x++) cs[x] += d[nb + x] - d[ob + x];
        }
        var l = cs[0], r = cs[w - 1];
        for (t = 0; t < ax; t++) P[t] = l;
        for (x = 0; x < w; x++) P[ax + x] = cs[x];
        for (t = ax + w; t < w + kx - 1; t++) P[t] = r;
        var sum = 0, b = y * w;
        for (t = 0; t < kx; t++) sum += P[t];
        for (x = 0; x < w; x++) {
            if (x > 0) sum += P[x + kx - 1] - P[x - 1];
            o[b + x] = normalize ? sum * scale : sum;
        }
    }
    return out;
}
cv.boxFilter = function cv_boxFilter(img, kx, normalize, ky) {
    if (normalize === undefined) normalize = true;
    if (ky === undefined) ky = kx;
    var c = img.data.constructor;
    return c === Uint8Array ? boxFilter_U8(img, kx, normalize, ky) : c === Float32Array ? boxFilter_F32(img, kx, normalize, ky) : boxFilter_F64(img, kx, normalize, ky);
};

// cv2.adaptiveThreshold(img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, win, C): 255 where src <= mean - ceil(C)
// (mean = rounded 8-bit box mean, replicate border).  Integer-only: for odd win the rounded mean never ties, so
// "src + idelta <= round(sum / k^2)"  <=>  2*sum >= (2*src + 2*idelta - 1) * k^2.
// Separable: running column sums (rows clamped = replicate), then a sliding horizontal window over them (columns padded = replicate).
// Writes into `out` (Uint8 image {w,h,data}, values 0/255) or, with `padded`, into a (w+2)x(h+2) Int8Array with a zero frame and
// foreground 1 (the layout findContours scans, so the 3 thresholds of the ArUco detector need no copy).
// (two copies - Uint8 output image / Int8 padded buffer - so every typed-array store site stays monomorphic)
function adaptiveThreshold_Plain(img, win, C, out, padded) {
    var w = img.w, h = img.h, d = img.data, k = win, a = k >> 1, k2 = k * k, idelta = Math.ceil(C);
    if ((k & 1) === 0) return adaptiveEven(img, win, C, out);
    var W = padded ? w + 2 : w, off = padded ? W + 1 : 0, fg = padded ? 1 : 255;
    var od, o = null;
    if (padded) { od = out; if (od.length < W * (h + 2)) throw new Error('padded buffer too small'); }
    else { o = out && out.data.length >= w * h ? out : mat(w, h); o.w = w; o.h = h; od = o.data; }
    var y, x;
    var c2 = (((2 * idelta - 1) * k2 + 1) / 2) | 0;       // integer: (2*idelta-1)*k2 is odd
    var P = scratch('atP', Int32Array, w + k);
    P.fill(0, 0, w + k);
    for (y = -a; y <= k - 1 - a; y++) {                   // initial column sums for output row 0
        var rb = (y < 0 ? 0 : y >= h ? h - 1 : y) * w;
        for (x = 0; x < w; x++) P[a + x] += d[rb + x];
    }
    for (y = 0; y < h; y++) {
        if (y > 0) {
            var nr = y + k - 1 - a, orow = y - 1 - a, nb = (nr >= h ? h - 1 : nr) * w, ob2 = (orow < 0 ? 0 : orow) * w;
            for (x = 0; x < w; x++) P[a + x] = (P[a + x] + d[nb + x] - d[ob2 + x]) | 0;
        }
        var l = P[a], r = P[a + w - 1], t;
        for (t = 0; t < a; t++) P[t] = l;
        for (t = a + w; t < w + k - 1; t++) P[t] = r;
        var sum = 0;
        for (t = 0; t < k; t++) sum += P[t];
        var b = y * w, o2 = off + y * W;
        od[o2] = (sum >= d[b] * k2 + c2) ? fg : 0;
        for (x = 1; x < w; x++) {
            sum = (sum + P[x + k - 1] - P[x - 1]) | 0;
            od[o2 + x] = (sum >= d[b + x] * k2 + c2) ? fg : 0;
        }
        if (padded) { od[o2 - 1] = 0; od[o2 + w] = 0; }
    }
    if (padded) { od.fill(0, 0, W); od.fill(0, W * (h + 1), W * (h + 2)); }
    return o;
}
// Same arithmetic with 32-bit word loads: four pixels per load (frame width and buffer offset multiples of 4).  Column sums are updated a
// word at a time, then the window slides over them four outputs per iteration.
function adaptiveWords(img, win, C, od) {
    var w = img.w, h = img.h, d = img.data, k = win, a = k >> 1, k2 = k * k, idelta = Math.ceil(C), W = w + 2;
    var c2 = (((2 * idelta - 1) * k2 + 1) / 2) | 0, y, x, m, t;
    var d32 = new Int32Array(d.buffer, d.byteOffset, d.length >> 2), w4 = w >> 2;
    var Pc = scratch('atPw', Int32Array, w + 2 * a + 8);
    Pc.fill(0, 0, w + 2 * a + 8);
    for (y = -a; y <= k - 1 - a; y++) { var rb = (y < 0 ? 0 : y >= h ? h - 1 : y) * w; for (x = 0; x < w; x++) Pc[a + x] += d[rb + x]; }
    for (y = 0; y < h; y++) {
        if (y > 0) {
            var nr = y + k - 1 - a, orow = y - 1 - a, nb4 = ((nr >= h ? h - 1 : nr) * w) >> 2, ob4 = ((orow < 0 ? 0 : orow) * w) >> 2;
            for (m = 0; m < w4; m++) {
                var dn = d32[nb4 + m], dd = d32[ob4 + m], q = a + 4 * m;
                Pc[q] = (Pc[q] + (dn & 255) - (dd & 255)) | 0;
                Pc[q + 1] = (Pc[q + 1] + ((dn >>> 8) & 255) - ((dd >>> 8) & 255)) | 0;
                Pc[q + 2] = (Pc[q + 2] + ((dn >>> 16) & 255) - ((dd >>> 16) & 255)) | 0;
                Pc[q + 3] = (Pc[q + 3] + (dn >>> 24) - (dd >>> 24)) | 0;
            }
        }
        var l = Pc[a], r = Pc[a + w - 1];
        for (t = 0; t < a; t++) Pc[t] = l;
        for (t = a + w; t < w + 2 * a; t++) Pc[t] = r;
        var sum = 0;
        for (t = 0; t < k; t++) sum += Pc[t];
        var o2 = W + 1 + y * W, b4 = (y * w) >> 2;
        for (m = 0; m < w4; m++) {
            var dw = d32[b4 + m], x4 = 4 * m, s0 = sum;
            var s1 = (s0 + Pc[x4 + k] - Pc[x4]) | 0, s2 = (s1 + Pc[x4 + k + 1] - Pc[x4 + 1]) | 0, s3 = (s2 + Pc[x4 + k + 2] - Pc[x4 + 2]) | 0;
            sum = (s3 + Pc[x4 + k + 3] - Pc[x4 + 3]) | 0;
            od[o2 + x4] = s0 >= (dw & 255) * k2 + c2 ? 1 : 0;
            od[o2 + x4 + 1] = s1 >= ((dw >>> 8) & 255) * k2 + c2 ? 1 : 0;
            od[o2 + x4 + 2] = s2 >= ((dw >>> 16) & 255) * k2 + c2 ? 1 : 0;
            od[o2 + x4 + 3] = s3 >= (dw >>> 24) * k2 + c2 ? 1 : 0;
        }
        od[o2 - 1] = 0; od[o2 + w] = 0;
    }
    od.fill(0, 0, W); od.fill(0, W * (h + 1), W * (h + 2));
}
// Fused single pass per row: the column sums are brought up to date just in time while the horizontal window slides over them
// (the value leaving the window was already updated earlier in the same row).  Same integer arithmetic as the two-pass version.
function adaptiveFused(img, win, C, od) {
    var w = img.w, h = img.h, d = img.data, k = win, a = k >> 1, k2 = k * k, idelta = Math.ceil(C), W = w + 2, fg = 1;
    var c2 = (((2 * idelta - 1) * k2 + 1) / 2) | 0, x, y, j, sum;
    var Pc = scratch('atPc', Int32Array, w);
    Pc.fill(0, 0, w);
    for (y = -a; y <= k - 1 - a; y++) { var rb = (y < 0 ? 0 : y >= h ? h - 1 : y) * w; for (x = 0; x < w; x++) Pc[x] += d[rb + x]; }
    var xB0 = a + 1, xB1 = w - a - 1;                 // loop B: x in [xB0, xB1)
    for (y = 0; y < h; y++) {
        var nb = 0, ob = 0;
        if (y > 0) { var nr = y + k - 1 - a, orow = y - 1 - a; nb = (nr >= h ? h - 1 : nr) * w; ob = (orow < 0 ? 0 : orow) * w; }
        for (x = 0; x <= a; x++) Pc[x] = (Pc[x] + d[nb + x] - d[ob + x]) | 0;
        sum = (a + 1) * Pc[0];
        for (x = 1; x <= a; x++) sum += Pc[x];
        var b = y * w, o2 = W + 1 + y * W, p0 = Pc[0];
        for (x = 0; x <= a; x++) {
            od[o2 + x] = sum >= d[b + x] * k2 + c2 ? fg : 0;
            j = x + a + 1;
            var pj = (Pc[j] + d[nb + j] - d[ob + j]) | 0;
            Pc[j] = pj;
            sum = (sum + pj - p0) | 0;
        }
        for (x = xB0; x < xB1; x++) {
            od[o2 + x] = sum >= d[b + x] * k2 + c2 ? fg : 0;
            j = x + a + 1;
            var pj2 = (Pc[j] + d[nb + j] - d[ob + j]) | 0;
            Pc[j] = pj2;
            sum = (sum + pj2 - Pc[x - a]) | 0;
        }
        var pl = Pc[w - 1];
        for (x = xB1; x < w; x++) {
            od[o2 + x] = sum >= d[b + x] * k2 + c2 ? fg : 0;
            sum = (sum + pl - Pc[x - a]) | 0;
        }
        od[o2 - 1] = 0; od[o2 + w] = 0;
    }
    od.fill(0, 0, W); od.fill(0, W * (h + 1), W * (h + 2));
}
function adaptiveThreshold_Padded(img, win, C, out, padded) {
    var w = img.w, h = img.h, d = img.data, k = win, a = k >> 1, k2 = k * k, idelta = Math.ceil(C);
    if ((k & 1) === 0) return adaptiveEven(img, win, C, out);
    var W = padded ? w + 2 : w, off = padded ? W + 1 : 0, fg = padded ? 1 : 255;
    var od, o = null;
    if (padded) { od = out; if (od.length < W * (h + 2)) throw new Error('padded buffer too small'); }
    else { o = out && out.data.length >= w * h ? out : mat(w, h); o.w = w; o.h = h; od = o.data; }
    var y, x;
    var c2 = (((2 * idelta - 1) * k2 + 1) / 2) | 0;       // integer: (2*idelta-1)*k2 is odd
    var P = scratch('atP', Int32Array, w + k);
    P.fill(0, 0, w + k);
    for (y = -a; y <= k - 1 - a; y++) {                   // initial column sums for output row 0
        var rb = (y < 0 ? 0 : y >= h ? h - 1 : y) * w;
        for (x = 0; x < w; x++) P[a + x] += d[rb + x];
    }
    for (y = 0; y < h; y++) {
        if (y > 0) {
            var nr = y + k - 1 - a, orow = y - 1 - a, nb = (nr >= h ? h - 1 : nr) * w, ob2 = (orow < 0 ? 0 : orow) * w;
            for (x = 0; x < w; x++) P[a + x] = (P[a + x] + d[nb + x] - d[ob2 + x]) | 0;
        }
        var l = P[a], r = P[a + w - 1], t;
        for (t = 0; t < a; t++) P[t] = l;
        for (t = a + w; t < w + k - 1; t++) P[t] = r;
        var sum = 0;
        for (t = 0; t < k; t++) sum += P[t];
        var b = y * w, o2 = off + y * W;
        od[o2] = (sum >= d[b] * k2 + c2) ? fg : 0;
        for (x = 1; x < w; x++) {
            sum = (sum + P[x + k - 1] - P[x - 1]) | 0;
            od[o2 + x] = (sum >= d[b + x] * k2 + c2) ? fg : 0;
        }
        if (padded) { od[o2 - 1] = 0; od[o2 + w] = 0; }
    }
    if (padded) { od.fill(0, 0, W); od.fill(0, W * (h + 1), W * (h + 2)); }
    return o;
}
cv.adaptiveThreshold = function cv_adaptiveThreshold(img, win, C, out, padded) {
    if (padded && (win & 1) === 1 && img.w >= win + 2 && out.length >= (img.w + 2) * (img.h + 2) && out instanceof Int8Array) { if (LITTLE_ENDIAN && (img.w & 3) === 0 && (img.data.byteOffset & 3) === 0 && img.data.constructor === Uint8Array) adaptiveWords(img, win, C, out); else adaptiveFused(img, win, C, out); return null; }
    return padded ? adaptiveThreshold_Padded(img, win, C, out, true) : adaptiveThreshold_Plain(img, win, C, out, false);
};
function adaptiveEven(img, win, C, out) {          // even window: half-up rounded mean (generic, slow path)
    var m = cv.boxFilter(img, win, true), w = img.w, h = img.h, o = out && !(out instanceof Int8Array) ? out : mat(w, h), idelta = Math.ceil(C);
    for (var i = 0; i < w * h; i++) o.data[i] = (img.data[i] - m.data[i] <= -idelta) ? 255 : 0;
    return o;
}

// cv2.integral(img, sdepth=CV_64F): (h+1) x (w+1) Float64
cv.integral = function cv_integral(img) {
    var w = img.w, h = img.h, d = img.data, W = w + 1, o = new Float64Array(W * (h + 1));
    for (var y = 0; y < h; y++) {
        var s = 0, b = (y + 1) * W, p = y * W;
        for (var x = 0; x < w; x++) { s += d[y * w + x]; o[b + x + 1] = o[p + x + 1] + s; }
    }
    return { w: W, h: h + 1, data: o };
};

// ------------------------------------------------------------------ contours
// cv2.findContours (Suzuki border following, OpenCV's icvFindNextContour / icvFetchContour ported literally).
// mode 'list' (RETR_LIST) | 'external' (RETR_EXTERNAL); method 'none' (CHAIN_APPROX_NONE) | 'simple' (CHAIN_APPROX_SIMPLE).
// Returns an array of Int32Array [x0,y0,x1,y1,...] in cv2's output order (last found first).
var DX = [1, 1, 0, -1, -1, -1, 0, 1], DY = [0, -1, -1, -1, 0, 1, 1, 1];
cv.findContours = function cv_findContours(bin, mode, method) {
    var w = bin.w, h = bin.h, src = bin.data, W = w + 2, H2 = h + 2;
    var img = scratch('fcImg', Int8Array, W * H2);
    var x, y;
    img.fill(0, 0, W);
    img.fill(0, W * (h + 1), W * H2);
    for (y = 0; y < h; y++) {
        var rb = (y + 1) * W;
        img[rb] = 0; img[rb + W - 1] = 0;
        var sb = y * w;
        for (x = 0; x < w; x++) img[rb + 1 + x] = src[sb + x] !== 0 ? 1 : 0;
    }
    return cv.findContoursPadded(img, w, h, mode, method);
};
// Same on a (w+2)x(h+2) Int8Array holding 0/1 with a zero frame (destroyed).  Contours with fewer than minPts / more than maxPts
// points are traced (they mark the image) but not returned.
cv.findContoursPadded = function cv_findContoursPadded(img, w, h, mode, method, minPts, maxPts) {
    var W = w + 2, mExternal = mode === 'external', mSimple = method === 'simple', x, y;
    minPts = minPts || 0; maxPts = maxPts || 1e9;
    var deltas = [1, -W + 1, -W, -W - 1, -1, W - 1, W, W + 1, 1, -W + 1, -W, -W - 1, -1, W - 1, W, W + 1];
    var out = [];
    var lnbdx = 0, lnbdy = 1, nbd = 2, NB = nbd | -128;
    var I32 = (img.byteOffset & 3) === 0 ? new Int32Array(img.buffer, img.byteOffset, img.length >> 2) : null;
    var buf = scratch('fcPts', Int32Array, 1024), np = 0;
    function put(px, py) {
        if (np + 2 > buf.length) { var nb = new Int32Array(buf.length * 2); nb.set(buf); buf = pool.fcPts = nb; }
        buf[np++] = px - 1; buf[np++] = py - 1;       // padded -> image coordinates
    }
    for (y = 1; y <= h; y++) {
        var prev = 0, p = 0, rowb = y * W;
        lnbdx = 0; lnbdy = y;                         // (set at row end in C: lnbd.x = 0; lnbd.y = y + 1)
        for (x = 1; x <= w; x++) {
            for (;;) {      // skip the run of pixels equal to prev; background is skipped a 32-bit word (4 pixels) at a time
                if (prev === 0 && I32 !== null && ((rowb + x) & 3) === 0) {
                    var wi = (rowb + x) >> 2;
                    while (x + 3 <= w && I32[wi] === 0) { wi++; x += 4; }
                }
                if (x > w) break;
                p = img[rowb + x];
                if (p !== prev) break;
                x++;
            }
            if (x > w) break;
            var isHole = 0, skip = false;
            if (!(prev === 0 && p === 1)) {
                if (p !== 0 || prev < 1) skip = true;
                else { if ((prev & -2) !== 0) lnbdx = x - 1; isHole = 1; }
            }
            if (!skip && mExternal && (isHole || img[lnbdy * W + lnbdx] > 0)) skip = true;
            if (!skip) {
                var ox = x - isHole, oy = y;
                lnbdx = x - isHole;
                // ---- icvFetchContour
                np = 0;
                var i0 = rowb + x - isHole, i1 = 0, i3, i4, s, sEnd, prevS, px = ox, py = oy;
                sEnd = s = isHole ? 0 : 4;
                do { s = (s - 1) & 7; i1 = i0 + deltas[s]; if (img[i1] !== 0) break; } while (s !== sEnd);
                if (s === sEnd) {
                    img[i0] = NB;
                    put(px, py);
                } else {
                    i3 = i0; prevS = s ^ 4;
                    for (;;) {
                        sEnd = s;
                        for (;;) { i4 = i3 + deltas[++s]; if (img[i4] !== 0) break; }
                        s &= 7;
                        if (((s - 1) >>> 0) < (sEnd >>> 0)) img[i3] = NB;
                        else if (img[i3] === 1) img[i3] = nbd;
                        if (s !== prevS || !mSimple) { put(px, py); prevS = s; }
                        px += DX[s]; py += DY[s];
                        if (i4 === i0 && i3 === i1) break;
                        i3 = i4; s = (s + 4) & 7;
                    }
                }
                if (np >= 2 * minPts && np <= 2 * maxPts) out.push(buf.slice(0, np));
                prev = img[rowb + x];                 // C re-reads img[x-1] on re-entry (x := x+1)
                continue;
            }
            prev = p;
            if ((prev & -2) !== 0) lnbdx = x;
        }
    }
    out.reverse();
    return out;
};
cv.findContoursExternal = function cv_findContoursExternal(bin, method) { return cv.findContours(bin, 'external', method || 'simple'); };

// cv2.arcLength(contour, closed)
cv.arcLength = function cv_arcLength(c, closed) {
    var n = c.length >> 1, s = 0;
    if (n <= 1) return 0;
    for (var i = 0; i < n - 1; i++) s += Math.hypot(c[2 * i + 2] - c[2 * i], c[2 * i + 3] - c[2 * i + 1]);
    if (closed) s += Math.hypot(c[0] - c[2 * n - 2], c[1] - c[2 * n - 1]);
    return s;
};
// cv2.contourArea(contour) (absolute value)
cv.contourArea = function cv_contourArea(c) {
    var n = c.length >> 1, a = 0;
    if (n < 3) return 0;
    var px = c[2 * n - 2], py = c[2 * n - 1];
    for (var i = 0; i < n; i++) { var x = c[2 * i], y = c[2 * i + 1]; a += px * y - x * py; px = x; py = y; }
    return Math.abs(a) * 0.5;
};
// cv2.isContourConvex (collinear triples count as not convex, as OpenCV)
cv.isContourConvex = function cv_isContourConvex(c) {
    var n = c.length >> 1;
    if (n < 3) return false;
    var px = c[2 * n - 2], py = c[2 * n - 1], cx = c[0], cy = c[1];
    var dx0 = cx - px, dy0 = cy - py, orient = 0;
    for (var i = 0; i < n; i++) {
        px = cx; py = cy; var j = (i + 1) % n; cx = c[2 * j]; cy = c[2 * j + 1];
        var dx = cx - px, dy = cy - py, a = dx * dy0, b = dy * dx0;
        orient |= (b > a) ? 1 : ((b < a) ? 2 : 3);
        if (orient === 3) return false;
        dx0 = dx; dy0 = dy;
    }
    return true;
};

// cv2.approxPolyDP (approx.cpp approxPolyDP_ ported literally).  c: flat [x,y,...] -> same kind of array
cv.approxPolyDP = function cv_approxPolyDP(c, eps, closed0) {
    var count = c.length >> 1;
    if (count === 0) return c.slice(0, 0);
    var dst = new c.constructor(c.length), stack = [];
    var initIters = 3, sliceS = 0, sliceE = 0, rS = 0, rE = 0, startX = -1000000, startY = -1000000, endX = 0, endY = 0, ptx = 0, pty = 0;
    var pos = 0, newCount = 0, closed = closed0, leEps = false, i, j;
    eps *= eps;
    if (!closed) {
        rS = count; endX = c[0]; endY = c[1]; startX = c[2 * count - 2]; startY = c[2 * count - 1];
        if (startX !== endX || startY !== endY) { sliceS = 0; sliceE = count - 1; stack.push(sliceS, sliceE); }
        else { closed = true; initIters = 1; }
    }
    if (closed) {
        rS = 0;
        for (i = 0; i < initIters; i++) {
            var maxDist = 0;
            pos = (pos + rS) % count;
            startX = c[2 * pos]; startY = c[2 * pos + 1]; if (++pos >= count) pos = 0;
            for (j = 1; j < count; j++) {
                ptx = c[2 * pos]; pty = c[2 * pos + 1]; if (++pos >= count) pos = 0;
                var dx = ptx - startX, dy = pty - startY, dist = dx * dx + dy * dy;
                if (dist > maxDist) { maxDist = dist; rS = j; }
            }
            leEps = maxDist <= eps;
        }
        if (!leEps) {
            rE = sliceS = pos % count;
            sliceE = rS = (rS + sliceS) % count;
            stack.push(rS, rE); stack.push(sliceS, sliceE);
        } else { dst[2 * newCount] = startX; dst[2 * newCount + 1] = startY; newCount++; }
    }
    while (stack.length) {
        sliceE = stack.pop(); sliceS = stack.pop();
        endX = c[2 * sliceE]; endY = c[2 * sliceE + 1];
        pos = sliceS;
        startX = c[2 * pos]; startY = c[2 * pos + 1]; if (++pos >= count) pos = 0;
        if (pos !== sliceE) {
            // cv2 5.0 measures the distance of a point to the SEGMENT start-end (the 4.x code used the distance to the infinite line)
            var md = 0, ddx = endX - startX, ddy = endY - startY, L2 = ddx * ddx + ddy * ddy;
            while (pos !== sliceE) {
                ptx = c[2 * pos]; pty = c[2 * pos + 1]; if (++pos >= count) pos = 0;
                var tpar = (ptx - startX) * ddx + (pty - startY) * ddy, d2;
                if (tpar <= 0) d2 = (ptx - startX) * (ptx - startX) + (pty - startY) * (pty - startY);
                else if (tpar >= L2) d2 = (ptx - endX) * (ptx - endX) + (pty - endY) * (pty - endY);
                else { var cr = (pty - startY) * ddx - (ptx - startX) * ddy; d2 = cr * cr / L2; }
                if (d2 > md) { md = d2; rS = (pos + count - 1) % count; }
            }
            leEps = md <= eps;
        } else { leEps = true; startX = c[2 * sliceS]; startY = c[2 * sliceS + 1]; }
        if (leEps) { dst[2 * newCount] = startX; dst[2 * newCount + 1] = startY; newCount++; }
        else { rE = sliceE; sliceE = rS; stack.push(rS, rE); stack.push(sliceS, sliceE); }
    }
    if (!closed) { dst[2 * newCount] = c[2 * count - 2]; dst[2 * newCount + 1] = c[2 * count - 1]; newCount++; }
    // final clean-up: drop points on [almost] straight lines
    closed = closed0;
    count = newCount;
    pos = closed ? count - 1 : 0;
    startX = dst[2 * pos]; startY = dst[2 * pos + 1]; if (++pos >= count) pos = 0;
    var wpos = pos;
    ptx = dst[2 * pos]; pty = dst[2 * pos + 1]; if (++pos >= count) pos = 0;
    for (i = closed ? 0 : 1; i < count - (closed ? 0 : 1) && newCount > 2; i++) {
        endX = dst[2 * pos]; endY = dst[2 * pos + 1]; if (++pos >= count) pos = 0;
        var ex = endX - startX, ey = endY - startY;
        var dd = Math.abs((ptx - startX) * ey - (pty - startY) * ex);
        var sip = (ptx - startX) * (endX - ptx) + (pty - startY) * (endY - pty);
        if (dd * dd <= 0.5 * eps * (ex * ex + ey * ey) && ex !== 0 && ey !== 0 && sip >= 0) {
            newCount--;
            dst[2 * wpos] = startX = endX; dst[2 * wpos + 1] = startY = endY;
            if (++wpos >= count) wpos = 0;
            ptx = dst[2 * pos]; pty = dst[2 * pos + 1]; if (++pos >= count) pos = 0;
            i++;
            continue;
        }
        dst[2 * wpos] = startX = ptx; dst[2 * wpos + 1] = startY = pty;
        if (++wpos >= count) wpos = 0;
        ptx = endX; pty = endY;
    }
    if (!closed) { dst[2 * wpos] = ptx; dst[2 * wpos + 1] = pty; }
    return dst.slice(0, 2 * newCount);
};

// ------------------------------------------------------------------ geometry
function solveLinear(A, b, n) {             // Gaussian elimination, partial pivoting (A row-major n x n, destroyed)
    for (var c = 0; c < n; c++) {
        var piv = c, best = Math.abs(A[c * n + c]);
        for (var r = c + 1; r < n; r++) if (Math.abs(A[r * n + c]) > best) { best = Math.abs(A[r * n + c]); piv = r; }
        if (best === 0) return null;
        if (piv !== c) {
            for (var k = 0; k < n; k++) { var t = A[c * n + k]; A[c * n + k] = A[piv * n + k]; A[piv * n + k] = t; }
            var tb = b[c]; b[c] = b[piv]; b[piv] = tb;
        }
        for (var r2 = c + 1; r2 < n; r2++) {
            var f = A[r2 * n + c] / A[c * n + c];
            if (f === 0) continue;
            for (var k2 = c; k2 < n; k2++) A[r2 * n + k2] -= f * A[c * n + k2];
            b[r2] -= f * b[c];
        }
    }
    var x = new Float64Array(n);
    for (var i = n - 1; i >= 0; i--) {
        var s = b[i];
        for (var j = i + 1; j < n; j++) s -= A[i * n + j] * x[j];
        x[i] = s / A[i * n + i];
    }
    return x;
}
cv.solveLinear = solveLinear;

// cv2.getPerspectiveTransform(src4, dst4): 4 points as flat [x0,y0,...x3,y3] -> Float64Array(9) row-major
cv.getPerspectiveTransform = function cv_getPerspectiveTransform(src, dst) {
    var A = new Float64Array(64), b = new Float64Array(8);
    for (var i = 0; i < 4; i++) {
        var sx = src[2 * i], sy = src[2 * i + 1], dx = dst[2 * i], dy = dst[2 * i + 1];
        var r = i * 8, r2 = (i + 4) * 8;
        A[r] = sx; A[r + 1] = sy; A[r + 2] = 1; A[r + 6] = -sx * dx; A[r + 7] = -sy * dx; b[i] = dx;
        A[r2 + 3] = sx; A[r2 + 4] = sy; A[r2 + 5] = 1; A[r2 + 6] = -sx * dy; A[r2 + 7] = -sy * dy; b[i + 4] = dy;
    }
    var x = solveLinear(A, b, 8), H = new Float64Array(9);
    if (!x) return H;
    for (i = 0; i < 8; i++) H[i] = x[i];
    H[8] = 1;
    return H;
};

cv.invert3x3 = function cv_invert3x3(M) {
    var a = M[0], b = M[1], c = M[2], d = M[3], e = M[4], f = M[5], g = M[6], h = M[7], i = M[8];
    var A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    var det = a * A + b * B + c * C, o = new Float64Array(9);
    if (det === 0) return o;
    var id = 1 / det;
    o[0] = A * id; o[1] = -(b * i - c * h) * id; o[2] = (b * f - c * e) * id;
    o[3] = B * id; o[4] = (a * i - c * g) * id; o[5] = -(a * f - c * d) * id;
    o[6] = C * id; o[7] = -(a * h - b * g) * id; o[8] = (a * e - b * d) * id;
    return o;
};

// cv2.perspectiveTransform on flat points -> new Float64Array
cv.perspectiveTransform = function cv_perspectiveTransform(pts, H, out) {
    var n = pts.length >> 1, o = out || new Float64Array(pts.length);
    for (var i = 0; i < n; i++) {
        var x = pts[2 * i], y = pts[2 * i + 1];
        var X = H[0] * x + H[1] * y + H[2], Y = H[3] * x + H[4] * y + H[5], Wd = H[6] * x + H[7] * y + H[8];
        Wd = Math.abs(Wd) > 2.220446049250313e-16 ? 1 / Wd : 0;
        o[2 * i] = X * Wd; o[2 * i + 1] = Y * Wd;
    }
    return o;
};

// cv2.findHomography(src, dst, 0): normalised DLT + 10 Levenberg-Marquardt iterations (npoints > 4), as OpenCV
cv.findHomography = function cv_findHomography(src, dst) {
    var n = src.length >> 1, i, j, k;
    if (n < 4) return null;
    var cxs = 0, cys = 0, cxd = 0, cyd = 0;
    for (i = 0; i < n; i++) { cxs += src[2 * i]; cys += src[2 * i + 1]; cxd += dst[2 * i]; cyd += dst[2 * i + 1]; }
    cxs /= n; cys /= n; cxd /= n; cyd /= n;
    var ss = 0, sd = 0;
    for (i = 0; i < n; i++) {
        ss += Math.hypot(src[2 * i] - cxs, src[2 * i + 1] - cys); sd += Math.hypot(dst[2 * i] - cxd, dst[2 * i + 1] - cyd);
    }
    ss = ss > 1e-12 ? Math.SQRT2 * n / ss : 1; sd = sd > 1e-12 ? Math.SQRT2 * n / sd : 1;
    var L = new Float64Array(81);
    for (i = 0; i < n; i++) {
        var x = (src[2 * i] - cxs) * ss, y = (src[2 * i + 1] - cys) * ss, X = (dst[2 * i] - cxd) * sd, Y = (dst[2 * i + 1] - cyd) * sd;
        var r1 = [0, 0, 0, -x, -y, -1, Y * x, Y * y, Y], r2 = [x, y, 1, 0, 0, 0, -X * x, -X * y, -X];
        for (j = 0; j < 9; j++) for (k = 0; k < 9; k++) L[j * 9 + k] += r1[j] * r1[k] + r2[j] * r2[k];
    }
    var v = smallestEigvec(L, 9);
    // H = Td^-1 * Hn * Ts
    var Hn = v, Ts = [ss, 0, -ss * cxs, 0, ss, -ss * cys, 0, 0, 1], Tdi = [1 / sd, 0, cxd, 0, 1 / sd, cyd, 0, 0, 1];
    var H = mul3(Tdi, mul3(Hn, Ts));
    var s = H[8] || 1; for (i = 0; i < 9; i++) H[i] /= s;
    if (n > 4) H = refineHomography(H, src, dst, n);
    return Float64Array.from(H);
};
function mul3(a, b) {
    var o = new Float64Array(9);
    for (var i = 0; i < 3; i++) for (var j = 0; j < 3; j++) { var s = 0; for (var k = 0; k < 3; k++) s += a[i * 3 + k] * b[k * 3 + j]; o[i * 3 + j] = s; }
    return o;
}
function smallestEigvec(A, n) {               // cyclic Jacobi on a symmetric matrix; returns eigenvector of the smallest eigenvalue
    var V = new Float64Array(n * n), i, j, k;
    for (i = 0; i < n; i++) V[i * n + i] = 1;
    A = Float64Array.from(A);
    for (var sweep = 0; sweep < 60; sweep++) {
        var off = 0;
        for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) off += A[i * n + j] * A[i * n + j];
        if (off < 1e-30) break;
        for (var p = 0; p < n - 1; p++) for (var q = p + 1; q < n; q++) {
            var apq = A[p * n + q];
            if (Math.abs(apq) < 1e-300) continue;
            var th = (A[q * n + q] - A[p * n + p]) / (2 * apq), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
            var c = 1 / Math.sqrt(t * t + 1), s = t * c;
            for (k = 0; k < n; k++) { var akp = A[k * n + p], akq = A[k * n + q]; A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq; }
            for (k = 0; k < n; k++) { var apk = A[p * n + k], aqk = A[q * n + k]; A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk; }
            for (k = 0; k < n; k++) { var vkp = V[k * n + p], vkq = V[k * n + q]; V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq; }
        }
    }
    var mi = 0;
    for (i = 1; i < n; i++) if (A[i * n + i] < A[mi * n + mi]) mi = i;
    var out = new Float64Array(n);
    for (k = 0; k < n; k++) out[k] = V[k * n + mi];
    return out;
}
function refineHomography(H, src, dst, n) {     // LM on the reprojection error (HomographyRefineCallback), 8 params, 10 iterations
    var h = Float64Array.from(H.slice(0, 8)), i, j, k, lambda = 1e-3;
    function resid(hh, J, r) {
        for (var q = 0; q < n; q++) {
            var x = src[2 * q], y = src[2 * q + 1];
            var w = hh[6] * x + hh[7] * y + 1, iw = Math.abs(w) > 2.2e-16 ? 1 / w : 0;
            var X = (hh[0] * x + hh[1] * y + hh[2]) * iw, Y = (hh[3] * x + hh[4] * y + hh[5]) * iw;
            if (r) { r[2 * q] = X - dst[2 * q]; r[2 * q + 1] = Y - dst[2 * q + 1]; }
            if (J) {
                var a = 2 * q * 8, b = (2 * q + 1) * 8;
                J[a] = x * iw; J[a + 1] = y * iw; J[a + 2] = iw; J[a + 3] = 0; J[a + 4] = 0; J[a + 5] = 0; J[a + 6] = -x * X * iw; J[a + 7] = -y * X * iw;
                J[b] = 0; J[b + 1] = 0; J[b + 2] = 0; J[b + 3] = x * iw; J[b + 4] = y * iw; J[b + 5] = iw; J[b + 6] = -x * Y * iw; J[b + 7] = -y * Y * iw;
            }
        }
    }
    var J = new Float64Array(2 * n * 8), r = new Float64Array(2 * n), r2 = new Float64Array(2 * n);
    function sse(v) { var s = 0; for (var q = 0; q < v.length; q++) s += v[q] * v[q]; return s; }
    resid(h, J, r);
    var err = sse(r);
    for (var it = 0; it < 10; it++) {
        var JtJ = new Float64Array(64), Jtr = new Float64Array(8);
        for (i = 0; i < 2 * n; i++) for (j = 0; j < 8; j++) {
            var jv = J[i * 8 + j]; if (jv === 0) continue;
            Jtr[j] += jv * r[i];
            for (k = 0; k < 8; k++) JtJ[j * 8 + k] += jv * J[i * 8 + k];
        }
        var ok = false;
        for (var tries = 0; tries < 12 && !ok; tries++) {
            var A = Float64Array.from(JtJ), bb = Float64Array.from(Jtr);
            for (j = 0; j < 8; j++) A[j * 9] *= 1 + lambda;
            var dlt = solveLinear(A, bb, 8);
            if (!dlt) { lambda *= 10; continue; }
            var h2 = new Float64Array(8);
            for (j = 0; j < 8; j++) h2[j] = h[j] - dlt[j];
            resid(h2, null, r2);
            var e2 = sse(r2);
            if (e2 < err) { err = e2; h = h2; lambda *= 0.1; resid(h, J, r); ok = true; var mx = 0; for (j = 0; j < 8; j++) mx = Math.max(mx, Math.abs(dlt[j])); if (mx < 1e-12) it = 99; }
            else lambda *= 10;
        }
        if (!ok) break;
    }
    var o = new Float64Array(9);
    for (j = 0; j < 8; j++) o[j] = h[j];
    o[8] = 1;
    return o;
}

// ---- bilinear sampling.  cv2 5.0.0 interpolates warpPerspective / remap in FLOAT at the exact source coordinate (measured by J3:
// 99.97% of pixels equal round(float bilinear); the 1/32 fixed-point model of 4.x only 92.5%).
function bilinearF(src, w, h, fx, fy, border, cval, isU8) {
    var x0 = Math.floor(fx), y0 = Math.floor(fy), ax = fx - x0, ay = fy - y0, a, b, c, d, x1 = x0 + 1, y1 = y0 + 1;
    if (x0 >= 0 && y0 >= 0 && x1 < w && y1 < h) {
        var p = y0 * w + x0; a = src[p]; b = src[p + 1]; c = src[p + w]; d = src[p + w + 1];
    } else if (border === 'constant') {
        if (x0 >= w || x1 < 0 || y0 >= h || y1 < 0) return cval;
        var okx0 = x0 >= 0 && x0 < w, okx1 = x1 >= 0 && x1 < w, oky0 = y0 >= 0 && y0 < h, oky1 = y1 >= 0 && y1 < h;
        a = okx0 && oky0 ? src[y0 * w + x0] : cval; b = okx1 && oky0 ? src[y0 * w + x1] : cval;
        c = okx0 && oky1 ? src[y1 * w + x0] : cval; d = okx1 && oky1 ? src[y1 * w + x1] : cval;
    } else {
        var xa = x0 < 0 ? 0 : x0 >= w ? w - 1 : x0, xb = x1 < 0 ? 0 : x1 >= w ? w - 1 : x1;
        var ya = y0 < 0 ? 0 : y0 >= h ? h - 1 : y0, yb = y1 < 0 ? 0 : y1 >= h ? h - 1 : y1;
        a = src[ya * w + xa]; b = src[ya * w + xb]; c = src[yb * w + xa]; d = src[yb * w + xb];
    }
    var v = (a * (1 - ax) + b * ax) * (1 - ay) + (c * (1 - ax) + d * ax) * ay;
    return isU8 ? sat8(rint(v)) : f32(v);
}

// cv2.warpPerspective(img, H, (w, h), flags=INTER_LINEAR|INTER_NEAREST, borderValue); H maps src -> dst (inverted inside)
// opts: {interp:'linear'|'nearest', border:'constant'|'replicate', value:0, inverse:false, out}
cv.warpPerspective = function cv_warpPerspective(img, H, w, h, opts) {
    opts = opts || {};
    var nearest = opts.interp === 'nearest', border = opts.border || 'constant', cval = opts.value || 0;
    var M = opts.inverse ? H : cv.invert3x3(H);
    var sw = img.w, sh = img.h, src = img.data, isU8 = src.constructor === Uint8Array;
    var out = opts.out && opts.out.data.length >= w * h ? opts.out : mat(w, h, src.constructor);
    out.w = w; out.h = h;
    var od = out.data, x, y, sw1 = sw - 1, sh1 = sh - 1, m0 = M[0], m3 = M[3], m6 = M[6];
    for (y = 0; y < h; y++) {
        var X0 = M[1] * y + M[2], Y0 = M[4] * y + M[5], W0 = M[7] * y + M[8], row = y * w;
        for (x = 0; x < w; x++) {
            var Wd = W0 + m6 * x; Wd = Wd ? 1 / Wd : 0;
            var fx = (X0 + m0 * x) * Wd, fy = (Y0 + m3 * x) * Wd;
            if (nearest) {
                var sx = rint(fx), sy = rint(fy);
                if (sx >= 0 && sx < sw && sy >= 0 && sy < sh) od[row + x] = src[sy * sw + sx];
                else if (border === 'constant') od[row + x] = cval;
                else od[row + x] = src[(sy < 0 ? 0 : sy >= sh ? sh - 1 : sy) * sw + (sx < 0 ? 0 : sx >= sw ? sw - 1 : sx)];
            } else if (isU8 && fx >= 0 && fy >= 0 && fx < sw1 && fy < sh1) {      // interior fast path of bilinearF (truncation = floor for fx, fy >= 0)
                var x0 = fx | 0, y0 = fy | 0, ax = fx - x0, ay = fy - y0, p = y0 * sw + x0, bx = 1 - ax;
                var v = (src[p] * bx + src[p + 1] * ax) * (1 - ay) + (src[p + sw] * bx + src[p + sw + 1] * ax) * ay;
                var vi = (v + 0.5) | 0;
                if (vi - v === 0.5 && (vi & 1)) vi--;
                od[row + x] = vi;
            } else od[row + x] = bilinearF(src, sw, sh, fx, fy, border, cval, isU8);
        }
    }
    return out;
};

// cv2.remap(img, mapx, mapy, INTER_LINEAR, borderMode) with float maps -> image of the map size.
// border 'replicate' | 'constant' (+ value)
cv.remap = function cv_remap(img, mapx, mapy, w, h, border, value) {
    var src = img.data, sw = img.w, sh = img.h, isU8 = src.constructor === Uint8Array, n = w * h;
    var out = mat(w, h, src.constructor), od = out.data, cval = value || 0;
    for (var i = 0; i < n; i++) {
        var fx = mapx[i], fy = mapy[i], x0 = Math.floor(fx), y0 = Math.floor(fy);
        if (isU8 && x0 >= 0 && y0 >= 0 && x0 < sw - 1 && y0 < sh - 1) {
            var ax = fx - x0, ay = fy - y0, p = y0 * sw + x0;
            var v = (src[p] * (1 - ax) + src[p + 1] * ax) * (1 - ay) + (src[p + sw] * (1 - ax) + src[p + sw + 1] * ax) * ay;
            var vi = Math.floor(v + 0.5);
            if (vi - v === 0.5 && (vi & 1)) vi--;
            od[i] = vi;
        } else od[i] = bilinearF(src, sw, sh, fx, fy, border, cval, isU8);
    }
    return out;
};
// samples at N points (xs, ys float32-valued) -> Float64Array (cv2.remap(...).astype(float64) in detect._sample)
cv.remapLinear = function cv_remapLinear(img, xs, ys, border, value) {
    var r = cv.remap(img, xs, ys, xs.length, 1, border || 'replicate', value);
    return Float64Array.from(r.data);
};

// cv2.cornerSubPix(gray uint8, corners (flat float32-valued [x,y,..], modified copy returned), win, zeroZone=(-1,-1), (maxIters, eps))
cv.cornerSubPix = function cv_cornerSubPix(img, corners, win, maxIters, eps) {
    var w = img.w, h = img.h, d = img.data, ww = win * 2 + 1, n = corners.length >> 1;
    var out = new Float64Array(corners.length);
    maxIters = Math.min(Math.max(maxIters, 1), 100); eps = eps * eps;
    var mask = new Float32Array(ww * ww), i, j, k;
    for (i = 0; i < ww; i++) {
        var yy = f32(f32(i - win) / win), vy = f32(Math.exp(-f32(yy * yy)));
        for (j = 0; j < ww; j++) { var xx = f32(f32(j - win) / win); mask[i * ww + j] = f32(vy * Math.exp(-f32(xx * xx))); }
    }
    var pw = ww + 2, buf = new Float32Array(pw * pw);
    for (var pi = 0; pi < n; pi++) {
        var cTx = f32(corners[2 * pi]), cTy = f32(corners[2 * pi + 1]), cIx = cTx, cIy = cTy, iter = 0, err = 0;
        do {
            // getRectSubPix(src, (pw, pw), cI) -> buf (float32, replicate border)
            var cx = f32(cIx - f32((pw - 1) * 0.5)), cy = f32(cIy - f32((pw - 1) * 0.5));
            var ipx = Math.floor(cx), ipy = Math.floor(cy), a = f32(cx - ipx), b = f32(cy - ipy);
            var a11 = f32(f32(1 - a) * f32(1 - b)), a12 = f32(a * f32(1 - b)), a21 = f32(f32(1 - a) * b), a22 = f32(a * b);
            var inside = ipx >= 0 && ipy >= 0 && ipx + pw < w && ipy + pw < h;
            for (i = 0; i < pw; i++) {
                var ry0 = ipy + i, ry1 = ry0 + 1;
                if (!inside) { ry0 = ry0 < 0 ? 0 : ry0 >= h ? h - 1 : ry0; ry1 = ry1 < 0 ? 0 : ry1 >= h ? h - 1 : ry1; }
                for (j = 0; j < pw; j++) {
                    var rx0 = ipx + j, rx1 = rx0 + 1;
                    if (!inside) { rx0 = rx0 < 0 ? 0 : rx0 >= w ? w - 1 : rx0; rx1 = rx1 < 0 ? 0 : rx1 >= w ? w - 1 : rx1; }
                    buf[i * pw + j] = d[ry0 * w + rx0] * a11 + d[ry0 * w + rx1] * a12 + d[ry1 * w + rx0] * a21 + d[ry1 * w + rx1] * a22;
                }
            }
            var A = 0, B = 0, C = 0, bb1 = 0, bb2 = 0;
            for (i = 0, k = 0; i < ww; i++) {
                var py = i - win, base = (i + 1) * pw + 1;
                for (j = 0; j < ww; j++, k++) {
                    var m = mask[k], tgx = buf[base + j + 1] - buf[base + j - 1], tgy = buf[base + j + pw] - buf[base + j - pw];
                    var gxx = tgx * tgx * m, gxy = tgx * tgy * m, gyy = tgy * tgy * m, px = j - win;
                    A += gxx; B += gxy; C += gyy; bb1 += gxx * px + gxy * py; bb2 += gxy * px + gyy * py;
                }
            }
            var det = A * C - B * B;
            if (Math.abs(det) <= 2.220446049250313e-16 * 2.220446049250313e-16) break;
            var scale = 1 / det;
            var nx = f32(cIx + C * scale * bb1 - B * scale * bb2), ny = f32(cIy - B * scale * bb1 + A * scale * bb2);
            err = (nx - cIx) * (nx - cIx) + (ny - cIy) * (ny - cIy);
            cIx = nx; cIy = ny;
            if (cIx < 0 || cIx >= w || cIy < 0 || cIy >= h) break;
        } while (++iter < maxIters && err > eps);
        if (Math.abs(cIx - cTx) > win || Math.abs(cIy - cTy) > win) { cIx = cTx; cIy = cTy; }
        out[2 * pi] = cIx; out[2 * pi + 1] = cIy;
    }
    return out;
};

// ------------------------------------------------------------------ resize
// cv2.resize.  mode 'area' | 'linear'.  (w, h) = destination size; invx / invy (optional) = the fx / fy the caller gave cv2
// (then cv2 derives the scale from them, not from the sizes: scale = 1/fx).  Use cv.resizeBy for the fx/fy form.
cv.resize = function cv_resize(img, w, h, mode, invx, invy) {
    var sw = img.w, sh = img.h;
    var sx = invx ? 1 / invx : sw / w, sy = invy ? 1 / invy : sh / h;
    if (mode === 'area') {
        if (sx >= 1 && sy >= 1) {
            var ix = rint(sx), iy = rint(sy);
            if (Math.abs(sx - ix) < 2.220446049250313e-16 && Math.abs(sy - iy) < 2.220446049250313e-16) return resizeAreaFast(img, w, h, ix, iy);
            return resizeAreaGeneral(img, w, h, sx, sy);
        }
    }
    return resizeLinear(img, w, h, sx, sy);
};
// cv2.resize(img, None, fx=, fy=, interpolation=...): destination size = cvRound(size * f)
cv.resizeBy = function cv_resizeBy(img, fx, fy, mode) {
    if (fy === undefined) fy = fx;
    return cv.resize(img, Math.max(rint(img.w * fx), 1), Math.max(rint(img.h * fy), 1), mode, fx, fy);
};

function resizeAreaFast_U8(img, w, h, kx, ky) {
    var sw = img.w, sh = img.h, d = img.data, out = mat(w, h, Uint8Array), o = out.data;
    var fullScale = f32(1 / (kx * ky)), x, y, i, j;
    for (y = 0; y < h; y++) {
        var ny = Math.min(ky, sh - y * ky);
        for (x = 0; x < w; x++) {
            var nx = Math.min(kx, sw - x * kx), s = 0, b = y * ky * sw + x * kx;     // at the edge cv2 averages the pixels that exist
            for (j = 0; j < ny; j++, b += sw) for (i = 0; i < nx; i++) s += d[b + i];
            o[y * w + x] = (kx === 2 && ky === 2 && nx === 2 && ny === 2) ? (s + 2) >> 2 : sat8(rint(f32(s * ((nx === kx && ny === ky) ? fullScale : f32(1 / (nx * ny))))));
        }
    }
    return out;
}
function resizeAreaFast_F(img, w, h, kx, ky) {
    var sw = img.w, sh = img.h, d = img.data, out = mat(w, h, Float32Array), o = out.data;
    var fullScale = f32(1 / (kx * ky)), x, y, i, j;
    for (y = 0; y < h; y++) {
        var ny = Math.min(ky, sh - y * ky);
        for (x = 0; x < w; x++) {
            var nx = Math.min(kx, sw - x * kx), s = 0, b = y * ky * sw + x * kx;     // at the edge cv2 averages the pixels that exist
            for (j = 0; j < ny; j++, b += sw) for (i = 0; i < nx; i++) s += d[b + i];
            o[y * w + x] = f32(s * ((nx === kx && ny === ky) ? fullScale : f32(1 / (nx * ny))));
        }
    }
    return out;
}
function resizeAreaFast(img, w, h, kx, ky) { return img.data.constructor === Uint8Array ? resizeAreaFast_U8(img, w, h, kx, ky) : resizeAreaFast_F(img, w, h, kx, ky); }

function areaTab(dsize, ssize, scale) {      // OpenCV's ResizeArea tables: for each destination index a list of (source index, weight)
    var di = [], si = [], al = [], count = new Int32Array(dsize + 1), k = 0;
    for (var dx = 0; dx < dsize; dx++) {
        var fs1 = dx * scale, fs2 = fs1 + scale, cell = Math.min(scale, ssize - fs1);
        var s1 = Math.ceil(fs1), s2 = Math.floor(fs2);
        s2 = Math.min(s2, ssize - 1); s1 = Math.min(s1, s2);
        count[dx] = k;
        if (s1 - fs1 > 1e-3) { si.push(s1 - 1); al.push(f32((s1 - fs1) / cell)); k++; }
        for (var sx = s1; sx < s2; sx++) { si.push(sx); al.push(f32(1 / cell)); k++; }
        if (fs2 - s2 > 1e-3) { si.push(s2); al.push(f32(Math.min(Math.min(fs2 - s2, 1), cell) / cell)); k++; }
    }
    count[dsize] = k;
    return { si: Int32Array.from(si), al: Float32Array.from(al), count: count };
}
function resizeAreaGeneral(img, w, h, sx, sy) {
    var sw = img.w, sh = img.h, d = img.data, isU8 = d.constructor === Uint8Array, out = mat(w, h, d.constructor), o = out.data;
    var xt = areaTab(w, sw, sx), yt = areaTab(h, sh, sy), x, y, k;
    var xsi = xt.si, xal = xt.al, xcnt = xt.count;
    var sum = new Float32Array(w), fixedT = true, maxT = 0, ti = new Int32Array(4 * w), tw = new Float32Array(4 * w);
    for (x = 0; x < w && fixedT; x++) {       // at most 4 taps per destination pixel: pad with (last index, weight 0)
        var nt = xcnt[x + 1] - xcnt[x];
        if (nt > 4 || nt < 1) { fixedT = false; break; }
        if (nt > maxT) maxT = nt;
        for (k = 0; k < 4; k++) { var kk = xcnt[x] + (k < nt ? k : nt - 1); ti[4 * x + k] = xsi[kk]; tw[4 * x + k] = k < nt ? xal[kk] : 0; }
    }
    // horizontal pass of a source row is computed once and kept while neighbouring destination rows share it (cache of 4 rows)
    var cacheRow = new Int32Array(4).fill(-1), cacheBuf = [new Float32Array(w), new Float32Array(w), new Float32Array(w), new Float32Array(w)], next = 0;
    for (y = 0; y < h; y++) {
        sum.fill(0);
        for (var yk = yt.count[y]; yk < yt.count[y + 1]; yk++) {
            var sr = yt.si[yk], srow = sr * sw, beta = yt.al[yk], hb = null, ci;
            for (ci = 0; ci < 4; ci++) if (cacheRow[ci] === sr) { hb = cacheBuf[ci]; break; }
            if (hb === null) {
                hb = cacheBuf[next]; cacheRow[next] = sr; next = (next + 1) & 3;
                if (fixedT && maxT <= 3) {
                    for (x = 0; x < w; x++) {
                        var o3 = x * 4, acc3 = f32(d[srow + ti[o3]] * tw[o3]);
                        acc3 = f32(acc3 + f32(d[srow + ti[o3 + 1]] * tw[o3 + 1]));
                        hb[x] = f32(acc3 + f32(d[srow + ti[o3 + 2]] * tw[o3 + 2]));
                    }
                } else if (fixedT) {             // taps padded to a fixed count with weight 0 (adding +0 changes nothing): unrolled
                    for (x = 0; x < w; x++) {
                        var o4 = x * 4, acc = f32(d[srow + ti[o4]] * tw[o4]);
                        acc = f32(acc + f32(d[srow + ti[o4 + 1]] * tw[o4 + 1]));
                        acc = f32(acc + f32(d[srow + ti[o4 + 2]] * tw[o4 + 2]));
                        acc = f32(acc + f32(d[srow + ti[o4 + 3]] * tw[o4 + 3]));
                        hb[x] = acc;
                    }
                } else for (x = 0; x < w; x++) {
                    var acc2 = 0;
                    for (k = xcnt[x]; k < xcnt[x + 1]; k++) acc2 = f32(acc2 + f32(d[srow + xsi[k]] * xal[k]));
                    hb[x] = acc2;
                }
            }
            for (x = 0; x < w; x++) sum[x] = f32(sum[x] + f32(hb[x] * beta));
        }
        if (isU8) for (x = 0; x < w; x++) o[y * w + x] = sat8(rint(sum[x]));
        else for (x = 0; x < w; x++) o[y * w + x] = sum[x];
    }
    return out;
}

// Lazily evaluated INTER_AREA downscale (general fractional path of resizeAreaGeneral) of an 8-bit image, optionally of its bitwise NOT,
// into a Float32 image holding the rounded 8-bit results.  ensure(x0, y0, x1, y1) computes (exactly the values cv.resize would give) the
// 32x32 output tiles that touch the half-open rectangle; reads outside the ensured area see 0.  Returns null when the scale is not the
// general fractional case (the caller then resizes eagerly).
cv.lazyAreaResize = function cv_lazyAreaResize(img, w, h, sx, sy, invert) {
    if (!(sx >= 1 && sy >= 1) || img.data.constructor !== Uint8Array) return null;
    var ix = rint(sx), iy = rint(sy);
    if (Math.abs(sx - ix) < 2.220446049250313e-16 && Math.abs(sy - iy) < 2.220446049250313e-16) return null;
    var sw = img.w, d = img.data, xt = areaTab(w, sw, sx), yt = areaTab(h, img.h, sy), xsi = xt.si, xal = xt.al, xcnt = xt.count;
    var TS = 32, tw = Math.ceil(w / TS), th = Math.ceil(h / TS), done = new Uint8Array(tw * th), data = new Float32Array(w * h);
    var sum = new Float32Array(TS), inv = invert ? 255 : 0, sgn = invert ? -1 : 1, x, k;
    var maxT = 0, ti = new Int32Array(4 * w), tww = new Float32Array(4 * w);         // taps padded to 4 per destination column (weight 0 adds nothing)
    for (x = 0; x < w; x++) {
        var nt = xcnt[x + 1] - xcnt[x];
        if (nt > 4 || nt < 1) return null;
        if (nt > maxT) maxT = nt;
        for (k = 0; k < 4; k++) { var kk = xcnt[x] + (k < nt ? k : nt - 1); ti[4 * x + k] = xsi[kk]; tww[4 * x + k] = k < nt ? xal[kk] : 0; }
    }
    var hbA = new Float32Array(TS), hbB = new Float32Array(TS), rowA = -1, rowB = -1;
    function hrow(sr, X0, nx, hb) {
        var srow = sr * sw, xx, o4;
        for (xx = 0; xx < nx; xx++) {
            o4 = (X0 + xx) * 4;
            var acc = f32((inv + sgn * d[srow + ti[o4]]) * tww[o4]);
            acc = f32(acc + f32((inv + sgn * d[srow + ti[o4 + 1]]) * tww[o4 + 1]));
            acc = f32(acc + f32((inv + sgn * d[srow + ti[o4 + 2]]) * tww[o4 + 2]));
            hb[xx] = f32(acc + f32((inv + sgn * d[srow + ti[o4 + 3]]) * tww[o4 + 3]));
        }
    }
    function tile(X0, Y0, X1, Y1) {
        var x, y, nx = X1 - X0;
        rowA = -1; rowB = -1;
        for (y = Y0; y < Y1; y++) {
            for (x = 0; x < nx; x++) sum[x] = 0;
            for (var yk = yt.count[y]; yk < yt.count[y + 1]; yk++) {
                var sr = yt.si[yk], beta = yt.al[yk], hb;
                if (sr === rowA) hb = hbA; else if (sr === rowB) hb = hbB;
                else { if (rowA < rowB) { hb = hbA; rowA = sr; } else { hb = hbB; rowB = sr; } hrow(sr, X0, nx, hb); }
                for (x = 0; x < nx; x++) sum[x] = f32(sum[x] + f32(hb[x] * beta));
            }
            for (x = 0; x < nx; x++) data[y * w + X0 + x] = sat8(rint(sum[x]));
        }
    }
    return {
        w: w, h: h, data: data,
        ensure: function (x0, y0, x1, y1) {
            x0 = x0 < 0 ? 0 : x0; y0 = y0 < 0 ? 0 : y0; x1 = x1 > w ? w : x1; y1 = y1 > h ? h : y1;
            if (x1 <= x0 || y1 <= y0) return;
            for (var ty = (y0 / TS) | 0; ty <= ((y1 - 1) / TS) | 0; ty++) for (var tx = (x0 / TS) | 0; tx <= ((x1 - 1) / TS) | 0; tx++) {
                if (done[ty * tw + tx]) continue;
                done[ty * tw + tx] = 1;
                tile(tx * TS, ty * TS, Math.min(tx * TS + TS, w), Math.min(ty * TS + TS, h));
            }
        }
    };
};

function resizeLinear(img, w, h, sx, sy) {
    var sw = img.w, sh = img.h, d = img.data, isU8 = d.constructor === Uint8Array, out = mat(w, h, d.constructor), o = out.data, x, y;
    var xo = new Int32Array(w), xa = new Float32Array(w), yo = new Int32Array(h), ya = new Float32Array(h);
    for (x = 0; x < w; x++) {
        var fx = f32(f32(f32(x + 0.5) * sx) - 0.5), ix = Math.floor(fx); fx = f32(fx - ix);
        if (ix < 0) { fx = 0; ix = 0; }
        if (ix >= sw - 1) { fx = 0; ix = sw - 1; }
        xo[x] = ix; xa[x] = fx;
    }
    for (y = 0; y < h; y++) {
        var fy = f32(f32(f32(y + 0.5) * sy) - 0.5), iy = Math.floor(fy); fy = f32(fy - iy);
        if (iy < 0) { fy = 0; iy = 0; }
        if (iy >= sh - 1) { fy = 0; iy = sh - 1; }
        yo[y] = iy; ya[y] = fy;
    }
    var xo1 = new Int32Array(w);
    for (x = 0; x < w; x++) xo1[x] = Math.min(xo[x] + 1, sw - 1);
    if (isU8) {
        var xa0 = new Int32Array(w), xa1 = new Int32Array(w), row0 = new Int32Array(w), row1 = new Int32Array(w), ca = -1, cb = -1;
        for (x = 0; x < w; x++) { xa0[x] = rint(f32(f32(1 - xa[x]) * 2048)); xa1[x] = rint(f32(xa[x] * 2048)); }
        for (y = 0; y < h; y++) {
            var r0 = yo[y], r1 = Math.min(r0 + 1, sh - 1), b1 = rint(f32(ya[y] * 2048)), b0 = rint(f32(f32(1 - ya[y]) * 2048)), t;
            if (r0 !== ca) {                                   // horizontal pass of row r0 (reuse the previous row1 when it is r0)
                if (r0 === cb) { t = row0; row0 = row1; row1 = t; } else { var s0 = r0 * sw; for (x = 0; x < w; x++) row0[x] = d[s0 + xo[x]] * xa0[x] + d[s0 + xo1[x]] * xa1[x]; }
                ca = r0;
            }
            if (r1 !== cb || r0 === cb) {
                var s1 = r1 * sw; for (x = 0; x < w; x++) row1[x] = d[s1 + xo[x]] * xa0[x] + d[s1 + xo1[x]] * xa1[x];
                cb = r1;
            }
            var ob = y * w;
            for (x = 0; x < w; x++) o[ob + x] = (((b0 * (row0[x] >> 4)) >> 16) + ((b1 * (row1[x] >> 4)) >> 16) + 2) >> 2;
        }
    } else {
        var row0f = new Float32Array(w), row1f = new Float32Array(w);
        for (y = 0; y < h; y++) {
            var q0 = yo[y], q1 = Math.min(q0 + 1, sh - 1), c1 = ya[y], c0 = f32(1 - c1), sa = q0 * sw, sb = q1 * sw;
            for (x = 0; x < w; x++) {
                var fa = xa[x], fb = f32(1 - fa);
                row0f[x] = f32(f32(d[sa + xo[x]] * fb) + f32(d[sa + xo1[x]] * fa));
                row1f[x] = f32(f32(d[sb + xo[x]] * fb) + f32(d[sb + xo1[x]] * fa));
            }
            var obf = y * w;
            for (x = 0; x < w; x++) o[obf + x] = f32(f32(c0 * row0f[x]) + f32(c1 * row1f[x]));
        }
    }
    return out;
}

// ------------------------------------------------------------------ Gaussian blur
function gaussKernelD(n, sigma) {                 // getGaussianKernel(n, sigma), double
    var k = new Float64Array(n), sum = 0, i;
    for (i = 0; i < n; i++) { var x = i - (n - 1) * 0.5; k[i] = Math.exp(-0.5 * x * x / (sigma * sigma)); sum += k[i]; }
    for (i = 0; i < n; i++) k[i] /= sum;
    return k;
}
// 8-bit images use OpenCV's bit-exact fixed-point path: kernel coefficients with 8 fractional bits, rounded by sequential error
// diffusion (measured against cv2 by impulse responses), exact integer sums, final (sum + 2^15) >> 16.
function gaussKernelFixed(n, sigma) {
    var k = gaussKernelD(n, sigma), c = new Int32Array(n), err = 0;
    for (var i = 0; i < n; i++) { var v = k[i] * 256 + err; c[i] = rint(v); err = v - c[i]; }
    return c;
}
// cv2.GaussianBlur(img, (ksize||0, ksize||0), sigma); ksize derived from sigma as cv2 does (uint8: round(6 sigma + 1)|1, float: round(8 sigma + 1)|1)
// Separable, one contiguous pass per symmetric tap pair, border reflect-101.  One copy per element type: JavaScriptCore
// de-optimises a loop that sees both Uint8Array and Float32Array (measured 5-10x slower), so the type dispatch is outside.
// (The 8-bit path is OpenCV's bit-exact fixed-point one: see gaussKernelFixed.)

// ---- 11-tap (sigma <= 2) Gaussian, specialised: horizontal pass computes four outputs from 14 loaded values, vertical pass four rows at a time
// from 14 loaded rows (the passes are load-bound; the sums are the same left-to-right expressions as the generic code, so results are identical).
function g11h_U8(d, b, w, tmp, P, c0, k0, k1, k2, k3, k4) {          // one row, Uint8 source -> Uint16 tmp (values <= 255 * 257)
    var x, q;
    for (x = 0; x < 5; x++) tmp[b + x] = c0 * P[x + 5] + k0 * (P[x] + P[x + 10]) + k1 * (P[x + 1] + P[x + 9]) + k2 * (P[x + 2] + P[x + 8]) + k3 * (P[x + 3] + P[x + 7]) + k4 * (P[x + 4] + P[x + 6]);
    for (x = 5; x + 3 < w - 5; x += 4) {
        q = b + x;
        var u0 = d[q - 5], u1 = d[q - 4], u2 = d[q - 3], u3 = d[q - 2], u4 = d[q - 1], u5 = d[q], u6 = d[q + 1], u7 = d[q + 2], u8 = d[q + 3], u9 = d[q + 4], u10 = d[q + 5], u11 = d[q + 6], u12 = d[q + 7], u13 = d[q + 8];
        tmp[q] = c0 * u5 + k0 * (u0 + u10) + k1 * (u1 + u9) + k2 * (u2 + u8) + k3 * (u3 + u7) + k4 * (u4 + u6);
        tmp[q + 1] = c0 * u6 + k0 * (u1 + u11) + k1 * (u2 + u10) + k2 * (u3 + u9) + k3 * (u4 + u8) + k4 * (u5 + u7);
        tmp[q + 2] = c0 * u7 + k0 * (u2 + u12) + k1 * (u3 + u11) + k2 * (u4 + u10) + k3 * (u5 + u9) + k4 * (u6 + u8);
        tmp[q + 3] = c0 * u8 + k0 * (u3 + u13) + k1 * (u4 + u12) + k2 * (u5 + u11) + k3 * (u6 + u10) + k4 * (u7 + u9);
    }
    for (; x < w - 5; x++) { q = b + x; tmp[q] = c0 * d[q] + k0 * (d[q - 5] + d[q + 5]) + k1 * (d[q - 4] + d[q + 4]) + k2 * (d[q - 3] + d[q + 3]) + k3 * (d[q - 2] + d[q + 2]) + k4 * (d[q - 1] + d[q + 1]); }
    for (; x < w; x++) tmp[b + x] = c0 * P[x + 5] + k0 * (P[x] + P[x + 10]) + k1 * (P[x + 1] + P[x + 9]) + k2 * (P[x + 2] + P[x + 8]) + k3 * (P[x + 3] + P[x + 7]) + k4 * (P[x + 4] + P[x + 6]);
}
function g11v1_U8(tmp, o, w, h, y, c0, k0, k1, k2, k3, k4) {
    var rc = y * w, q0 = reflect101(y - 5, h) * w, q1 = reflect101(y - 4, h) * w, q2 = reflect101(y - 3, h) * w, q3 = reflect101(y - 2, h) * w, q4 = reflect101(y - 1, h) * w;
    var u4 = reflect101(y + 1, h) * w, u3 = reflect101(y + 2, h) * w, u2 = reflect101(y + 3, h) * w, u1 = reflect101(y + 4, h) * w, u0 = reflect101(y + 5, h) * w;
    for (var x = 0; x < w; x++) {
        var v = (c0 * tmp[rc + x] + k0 * (tmp[q0 + x] + tmp[u0 + x]) + k1 * (tmp[q1 + x] + tmp[u1 + x]) + k2 * (tmp[q2 + x] + tmp[u2 + x]) + k3 * (tmp[q3 + x] + tmp[u3 + x]) + k4 * (tmp[q4 + x] + tmp[u4 + x]) + 32768) >> 16;
        o[rc + x] = v > 255 ? 255 : v;
    }
}
function g11v4_U8(tmp, o, w, y, c0, k0, k1, k2, k3, k4) {          // rows y..y+3, all 14 source rows inside the image
    var r0 = (y - 5) * w, oa = y * w, x, q;
    for (x = 0; x < w; x++) {
        q = r0 + x;
        var t0 = tmp[q], t1 = tmp[q + w], t2 = tmp[q + 2 * w], t3 = tmp[q + 3 * w], t4 = tmp[q + 4 * w], t5 = tmp[q + 5 * w], t6 = tmp[q + 6 * w], t7 = tmp[q + 7 * w], t8 = tmp[q + 8 * w], t9 = tmp[q + 9 * w], t10 = tmp[q + 10 * w], t11 = tmp[q + 11 * w], t12 = tmp[q + 12 * w], t13 = tmp[q + 13 * w];
        var va = (c0 * t5 + k0 * (t0 + t10) + k1 * (t1 + t9) + k2 * (t2 + t8) + k3 * (t3 + t7) + k4 * (t4 + t6) + 32768) >> 16;
        var vb = (c0 * t6 + k0 * (t1 + t11) + k1 * (t2 + t10) + k2 * (t3 + t9) + k3 * (t4 + t8) + k4 * (t5 + t7) + 32768) >> 16;
        var vc = (c0 * t7 + k0 * (t2 + t12) + k1 * (t3 + t11) + k2 * (t4 + t10) + k3 * (t5 + t9) + k4 * (t6 + t8) + 32768) >> 16;
        var vd = (c0 * t8 + k0 * (t3 + t13) + k1 * (t4 + t12) + k2 * (t5 + t11) + k3 * (t6 + t10) + k4 * (t7 + t9) + 32768) >> 16;
        var oo = oa + x;
        o[oo] = va > 255 ? 255 : va; o[oo + w] = vb > 255 ? 255 : vb; o[oo + 2 * w] = vc > 255 ? 255 : vc; o[oo + 3 * w] = vd > 255 ? 255 : vd;
    }
}
function g11_U8(img, c) {
    var w = img.w, h = img.h, d = img.data, out = mat(w, h, Uint8Array), o = out.data, x, y;
    var c0 = c[5], k0 = c[0], k1 = c[1], k2 = c[2], k3 = c[3], k4 = c[4];
    var tmp = scratch('g11T', Uint16Array, w * h), P = scratch('g11P', Int32Array, w + 10);
    for (y = 0; y < h; y++) {
        var b = y * w;
        for (x = 0; x < 15; x++) P[x] = d[b + reflect101(x - 5, w)];                     // pixels -5..9
        for (x = w - 5; x < w + 10; x++) P[x] = d[b + reflect101(x - 5, w)];             // pixels w-10..w+4
        g11h_U8(d, b, w, tmp, P, c0, k0, k1, k2, k3, k4);
    }
    for (y = 0; y < 5; y++) g11v1_U8(tmp, o, w, h, y, c0, k0, k1, k2, k3, k4);
    for (y = 5; y + 3 <= h - 6; y += 4) g11v4_U8(tmp, o, w, y, c0, k0, k1, k2, k3, k4);
    for (; y < h; y++) g11v1_U8(tmp, o, w, h, y, c0, k0, k1, k2, k3, k4);
    return out;
}
function g11h_F(P, tmp, b, w, c0, h0, h1, h2, h3, h4) {                 // P: padded row (Float64, w + 10); tmp: Float64 holding float32-rounded values
    var x;
    for (x = 0; x + 3 < w; x += 4) {
        var u0 = P[x], u1 = P[x + 1], u2 = P[x + 2], u3 = P[x + 3], u4 = P[x + 4], u5 = P[x + 5], u6 = P[x + 6], u7 = P[x + 7], u8 = P[x + 8], u9 = P[x + 9], u10 = P[x + 10], u11 = P[x + 11], u12 = P[x + 12], u13 = P[x + 13];
        var q = b + x;
        tmp[q] = f32(c0 * u5 + h0 * (u0 + u10) + h1 * (u1 + u9) + h2 * (u2 + u8) + h3 * (u3 + u7) + h4 * (u4 + u6));
        tmp[q + 1] = f32(c0 * u6 + h0 * (u1 + u11) + h1 * (u2 + u10) + h2 * (u3 + u9) + h3 * (u4 + u8) + h4 * (u5 + u7));
        tmp[q + 2] = f32(c0 * u7 + h0 * (u2 + u12) + h1 * (u3 + u11) + h2 * (u4 + u10) + h3 * (u5 + u9) + h4 * (u6 + u8));
        tmp[q + 3] = f32(c0 * u8 + h0 * (u3 + u13) + h1 * (u4 + u12) + h2 * (u5 + u11) + h3 * (u6 + u10) + h4 * (u7 + u9));
    }
    for (; x < w; x++) tmp[b + x] = f32(c0 * P[x + 5] + h0 * (P[x] + P[x + 10]) + h1 * (P[x + 1] + P[x + 9]) + h2 * (P[x + 2] + P[x + 8]) + h3 * (P[x + 3] + P[x + 7]) + h4 * (P[x + 4] + P[x + 6]));
}
function g11v1_F(tmp, w, h, y, c0, h0, h1, h2, h3, h4, x) {            // one output value at column x of row y (reflect101 rows)
    var rc = y * w + x;
    return f32(c0 * tmp[rc] + h0 * (tmp[reflect101(y - 5, h) * w + x] + tmp[reflect101(y + 5, h) * w + x]) + h1 * (tmp[reflect101(y - 4, h) * w + x] + tmp[reflect101(y + 4, h) * w + x])
        + h2 * (tmp[reflect101(y - 3, h) * w + x] + tmp[reflect101(y + 3, h) * w + x]) + h3 * (tmp[reflect101(y - 2, h) * w + x] + tmp[reflect101(y + 2, h) * w + x]) + h4 * (tmp[reflect101(y - 1, h) * w + x] + tmp[reflect101(y + 1, h) * w + x]));
}
function g11v4_F32(tmp, o, w, y, c0, h0, h1, h2, h3, h4) {             // float32 output, rows y..y+3 (source rows y-5..y+8 inside the image)
    var r0 = (y - 5) * w, oa = y * w, x, q;
    for (x = 0; x < w; x++) {
        q = r0 + x;
        var t0 = tmp[q], t1 = tmp[q + w], t2 = tmp[q + 2 * w], t3 = tmp[q + 3 * w], t4 = tmp[q + 4 * w], t5 = tmp[q + 5 * w], t6 = tmp[q + 6 * w], t7 = tmp[q + 7 * w], t8 = tmp[q + 8 * w], t9 = tmp[q + 9 * w], t10 = tmp[q + 10 * w], t11 = tmp[q + 11 * w], t12 = tmp[q + 12 * w], t13 = tmp[q + 13 * w];
        var oo = oa + x;
        o[oo] = c0 * t5 + h0 * (t0 + t10) + h1 * (t1 + t9) + h2 * (t2 + t8) + h3 * (t3 + t7) + h4 * (t4 + t6);
        o[oo + w] = c0 * t6 + h0 * (t1 + t11) + h1 * (t2 + t10) + h2 * (t3 + t9) + h3 * (t4 + t8) + h4 * (t5 + t7);
        o[oo + 2 * w] = c0 * t7 + h0 * (t2 + t12) + h1 * (t3 + t11) + h2 * (t4 + t10) + h3 * (t5 + t9) + h4 * (t6 + t8);
        o[oo + 3 * w] = c0 * t8 + h0 * (t3 + t13) + h1 * (t4 + t12) + h2 * (t5 + t11) + h3 * (t6 + t10) + h4 * (t7 + t9);
    }
}
function g11v4_flat(tmp, o, w, y, c0, h0, h1, h2, h3, h4) {            // same, rounded to float32, clipped to 0..255 and truncated into a Uint8 image
    var r0 = (y - 5) * w, oa = y * w, x, q;
    for (x = 0; x < w; x++) {
        q = r0 + x;
        var t0 = tmp[q], t1 = tmp[q + w], t2 = tmp[q + 2 * w], t3 = tmp[q + 3 * w], t4 = tmp[q + 4 * w], t5 = tmp[q + 5 * w], t6 = tmp[q + 6 * w], t7 = tmp[q + 7 * w], t8 = tmp[q + 8 * w], t9 = tmp[q + 9 * w], t10 = tmp[q + 10 * w], t11 = tmp[q + 11 * w], t12 = tmp[q + 12 * w], t13 = tmp[q + 13 * w];
        var va = f32(c0 * t5 + h0 * (t0 + t10) + h1 * (t1 + t9) + h2 * (t2 + t8) + h3 * (t3 + t7) + h4 * (t4 + t6));
        var vb = f32(c0 * t6 + h0 * (t1 + t11) + h1 * (t2 + t10) + h2 * (t3 + t9) + h3 * (t4 + t8) + h4 * (t5 + t7));
        var vc = f32(c0 * t7 + h0 * (t2 + t12) + h1 * (t3 + t11) + h2 * (t4 + t10) + h3 * (t5 + t9) + h4 * (t6 + t8));
        var vd = f32(c0 * t8 + h0 * (t3 + t13) + h1 * (t4 + t12) + h2 * (t5 + t11) + h3 * (t6 + t10) + h4 * (t7 + t9));
        var oo = oa + x;
        o[oo] = va < 0 ? 0 : va > 255 ? 255 : va; o[oo + w] = vb < 0 ? 0 : vb > 255 ? 255 : vb; o[oo + 2 * w] = vc < 0 ? 0 : vc > 255 ? 255 : vc; o[oo + 3 * w] = vd < 0 ? 0 : vd > 255 ? 255 : vd;
    }
}
function g11_F32(img, c) {
    var w = img.w, h = img.h, d = img.data, out = mat(w, h, Float32Array), o = out.data, x, y;
    var c0 = c[5], h0 = c[0], h1 = c[1], h2 = c[2], h3 = c[3], h4 = c[4];
    var tmp = scratch('g11TF', Float64Array, w * h), P = scratch('g11PF', Float64Array, w + 10);
    for (y = 0; y < h; y++) {
        var b = y * w;
        for (x = 0; x < 5; x++) P[x] = d[b + reflect101(x - 5, w)];
        for (x = 0; x < w; x++) P[5 + x] = d[b + x];
        for (x = 0; x < 5; x++) P[5 + w + x] = d[b + reflect101(w + x, w)];
        g11h_F(P, tmp, b, w, c0, h0, h1, h2, h3, h4);
    }
    for (y = 0; y < 5; y++) for (x = 0; x < w; x++) o[y * w + x] = g11v1_F(tmp, w, h, y, c0, h0, h1, h2, h3, h4, x);
    for (y = 5; y + 3 <= h - 6; y += 4) g11v4_F32(tmp, o, w, y, c0, h0, h1, h2, h3, h4);
    for (; y < h; y++) for (x = 0; x < w; x++) o[y * w + x] = g11v1_F(tmp, w, h, y, c0, h0, h1, h2, h3, h4, x);
    return out;
}
function gaussBlur_U8(img, sigma, n) {
    var w = img.w, h = img.h, d = img.data, out = mat(w, h, Uint8Array), o = out.data, r = n >> 1, x, y, j;
    var c = gaussKernelFixed(n, sigma);
    if (r === 5 && w >= 24 && h >= 24) { var cs = 0; for (j = 0; j < n; j++) cs += c[j]; if (cs <= 257) return g11_U8(img, c); }
    var P = scratch('gbPU8', Int32Array, w + 2 * r), acc = scratch('gbAU8', Int32Array, w);
    var tmp = scratch('gbTU8', Int32Array, w * h);
    if (r === 5) { var csum = 0; for (j = 0; j < n; j++) csum += c[j]; if (csum <= 257) tmp = scratch('gbTU8s', Uint16Array, w * h); }   // 255 * 257 < 2^16: halves the memory traffic of the vertical pass
    for (y = 0; y < h; y++) {                       // horizontal: padded row, then one pass per tap pair
        var b = y * w;
        for (x = 0; x < r; x++) P[x] = d[b + reflect101(x - r, w)];
        if (r === 5 && w > 20) { for (x = 0; x < 10; x++) P[r + x] = d[b + x]; for (x = w - 10; x < w; x++) P[r + x] = d[b + x]; }
        else for (x = 0; x < w; x++) P[r + x] = d[b + x];
        for (x = 0; x < r; x++) P[r + w + x] = d[b + reflect101(w + x, w)];
        var c0 = c[r];
        if (r === 5) {                              // sigma 1.5 (the canvas smoothing): one fused pass, all 11 taps
            var k0 = c[0], k1 = c[1], k2 = c[2], k3 = c[3], k4 = c[4];
            for (x = 0; x < 5 && x < w; x++) tmp[b + x] = c0 * P[x + 5] + k0 * (P[x] + P[x + 10]) + k1 * (P[x + 1] + P[x + 9]) + k2 * (P[x + 2] + P[x + 8]) + k3 * (P[x + 3] + P[x + 7]) + k4 * (P[x + 4] + P[x + 6]);
            for (; x < w - 5; x++) { var q = b + x; tmp[q] = c0 * d[q] + k0 * (d[q - 5] + d[q + 5]) + k1 * (d[q - 4] + d[q + 4]) + k2 * (d[q - 3] + d[q + 3]) + k3 * (d[q - 2] + d[q + 2]) + k4 * (d[q - 1] + d[q + 1]); }
            for (; x < w; x++) tmp[b + x] = c0 * P[x + 5] + k0 * (P[x] + P[x + 10]) + k1 * (P[x + 1] + P[x + 9]) + k2 * (P[x + 2] + P[x + 8]) + k3 * (P[x + 3] + P[x + 7]) + k4 * (P[x + 4] + P[x + 6]);
            continue;
        }
        for (x = 0; x < w; x++) acc[x] = c0 * P[x + r];
        for (j = 1; j <= r; j++) { var cj = c[r + j], lo = r - j, hi = r + j; for (x = 0; x < w; x++) acc[x] += cj * (P[x + lo] + P[x + hi]); }
        for (x = 0; x < w; x++) tmp[b + x] = acc[x];
    }
    for (y = 0; y < h; y++) {                       // vertical: centre row * c0 + sum over taps of (row above + row below) * cj
        var rc = y * w, c1 = c[r], ob = y * w;
        if (r === 5) {
            var q0 = reflect101(y - 5, h) * w, q1 = reflect101(y - 4, h) * w, q2 = reflect101(y - 3, h) * w, q3 = reflect101(y - 2, h) * w, q4 = reflect101(y - 1, h) * w;
            var u4 = reflect101(y + 1, h) * w, u3 = reflect101(y + 2, h) * w, u2 = reflect101(y + 3, h) * w, u1 = reflect101(y + 4, h) * w, u0 = reflect101(y + 5, h) * w;
            var e0 = c[0], e1 = c[1], e2 = c[2], e3 = c[3], e4 = c[4];
            for (x = 0; x < w; x++) {
                var v5 = (c1 * tmp[rc + x] + e0 * (tmp[q0 + x] + tmp[u0 + x]) + e1 * (tmp[q1 + x] + tmp[u1 + x]) + e2 * (tmp[q2 + x] + tmp[u2 + x]) + e3 * (tmp[q3 + x] + tmp[u3 + x]) + e4 * (tmp[q4 + x] + tmp[u4 + x]) + 32768) >> 16;
                o[ob + x] = v5 > 255 ? 255 : v5;
            }
            continue;
        }
        for (x = 0; x < w; x++) acc[x] = c1 * tmp[rc + x];
        for (j = 1; j <= r; j++) {
            var cj2 = c[r + j], ra = reflect101(y - j, h) * w, rb = reflect101(y + j, h) * w;
            for (x = 0; x < w; x++) acc[x] += cj2 * (tmp[ra + x] + tmp[rb + x]);
        }
        for (x = 0; x < w; x++) { var v = (acc[x] + 32768) >> 16; o[ob + x] = v > 255 ? 255 : v; }
    }
    return out;
}
function gaussBlur_F32(img, sigma, n) {
    var w = img.w, h = img.h, d = img.data, out = mat(w, h, Float32Array), o = out.data, r = n >> 1, x, y, j;
    var c = gaussKernelD(n, sigma);
    for (j = 0; j < n; j++) c[j] = f32(c[j]);
    if (r === 5 && w >= 24 && h >= 24) return g11_F32(img, c);
    var P = scratch('gbPF32', Float64Array, w + 2 * r), acc = scratch('gbAF32', Float64Array, w);
    var tmp = scratch('gbTF32', Float32Array, w * h);
    for (y = 0; y < h; y++) {                       // horizontal: padded row, then one pass per tap pair
        var b = y * w;
        for (x = 0; x < r; x++) P[x] = d[b + reflect101(x - r, w)];
        for (x = 0; x < w; x++) P[r + x] = d[b + x];
        for (x = 0; x < r; x++) P[r + w + x] = d[b + reflect101(w + x, w)];
        var c0 = c[r];
        if (r === 5) {      // fused: the same left-to-right sum as the per-tap passes below
            var h0 = c[0], h1 = c[1], h2 = c[2], h3 = c[3], h4 = c[4];
            for (x = 0; x < w; x++) tmp[b + x] = c0 * P[x + 5] + h0 * (P[x] + P[x + 10]) + h1 * (P[x + 1] + P[x + 9]) + h2 * (P[x + 2] + P[x + 8]) + h3 * (P[x + 3] + P[x + 7]) + h4 * (P[x + 4] + P[x + 6]);
            continue;
        }
        for (x = 0; x < w; x++) acc[x] = c0 * P[x + r];
        for (j = 1; j <= r; j++) { var cj = c[r + j], lo = r - j, hi = r + j; for (x = 0; x < w; x++) acc[x] += cj * (P[x + lo] + P[x + hi]); }
        for (x = 0; x < w; x++) tmp[b + x] = acc[x];
    }
    for (y = 0; y < h; y++) {                       // vertical: centre row * c0 + sum over taps of (row above + row below) * cj
        var rc = y * w, c1 = c[r];
        if (r === 5) {
            var q0 = reflect101(y - 5, h) * w, q1 = reflect101(y - 4, h) * w, q2 = reflect101(y - 3, h) * w, q3 = reflect101(y - 2, h) * w, q4 = reflect101(y - 1, h) * w;
            var u4 = reflect101(y + 1, h) * w, u3 = reflect101(y + 2, h) * w, u2 = reflect101(y + 3, h) * w, u1 = reflect101(y + 4, h) * w, u0 = reflect101(y + 5, h) * w;
            var e0 = c[0], e1 = c[1], e2 = c[2], e3 = c[3], e4 = c[4], ob5 = y * w;
            for (x = 0; x < w; x++) o[ob5 + x] = c1 * tmp[rc + x] + e0 * (tmp[q0 + x] + tmp[u0 + x]) + e1 * (tmp[q1 + x] + tmp[u1 + x]) + e2 * (tmp[q2 + x] + tmp[u2 + x]) + e3 * (tmp[q3 + x] + tmp[u3 + x]) + e4 * (tmp[q4 + x] + tmp[u4 + x]);
            continue;
        }
        for (x = 0; x < w; x++) acc[x] = c1 * tmp[rc + x];
        for (j = 1; j <= r; j++) {
            var cj2 = c[r + j], ra = reflect101(y - j, h) * w, rb = reflect101(y + j, h) * w;
            for (x = 0; x < w; x++) acc[x] += cj2 * (tmp[ra + x] + tmp[rb + x]);
        }
        var ob = y * w;
        for (x = 0; x < w; x++) o[ob + x] = acc[x];
    }
    return out;
}
function gaussBlur_F64(img, sigma, n) {
    var w = img.w, h = img.h, d = img.data, out = mat(w, h, Float64Array), o = out.data, r = n >> 1, x, y, j;
    var c = gaussKernelD(n, sigma);
    
    var P = scratch('gbPF64', Float64Array, w + 2 * r), acc = scratch('gbAF64', Float64Array, w);
    var tmp = scratch('gbTF64', Float64Array, w * h);
    for (y = 0; y < h; y++) {                       // horizontal: padded row, then one pass per tap pair
        var b = y * w;
        for (x = 0; x < r; x++) P[x] = d[b + reflect101(x - r, w)];
        for (x = 0; x < w; x++) P[r + x] = d[b + x];
        for (x = 0; x < r; x++) P[r + w + x] = d[b + reflect101(w + x, w)];
        var c0 = c[r];
        for (x = 0; x < w; x++) acc[x] = c0 * P[x + r];
        for (j = 1; j <= r; j++) { var cj = c[r + j], lo = r - j, hi = r + j; for (x = 0; x < w; x++) acc[x] += cj * (P[x + lo] + P[x + hi]); }
        for (x = 0; x < w; x++) tmp[b + x] = acc[x];
    }
    for (y = 0; y < h; y++) {                       // vertical: centre row * c0 + sum over taps of (row above + row below) * cj
        var rc = y * w, c1 = c[r];
        for (x = 0; x < w; x++) acc[x] = c1 * tmp[rc + x];
        for (j = 1; j <= r; j++) {
            var cj2 = c[r + j], ra = reflect101(y - j, h) * w, rb = reflect101(y + j, h) * w;
            for (x = 0; x < w; x++) acc[x] += cj2 * (tmp[ra + x] + tmp[rb + x]);
        }
        var ob = y * w;
        for (x = 0; x < w; x++) o[ob + x] = acc[x];
    }
    return out;
}
// detect.py:flatten_lighting tail, fused: max(bg, 8) -> divide(gray, bg, scale) -> GaussianBlur(float32, sigma) -> clip to 0..255 and truncate
// to uint8, bit-identical to the separate steps (sigma whose kernel is 11 taps; anything else takes the separate steps).
cv.flattenDivideBlur = function cv_flattenDivideBlur(gray, bg, scale, sigma) {
    var w = gray.w, h = gray.h, A = gray.data, B = bg.data, n = rint(sigma * 4 * 2 + 1) | 1, x, y, i;
    if (n !== 11 || w < 12 || h < 12) {
        var bb = new Uint8Array(B.length);
        for (i = 0; i < B.length; i++) bb[i] = B[i] < 8 ? 8 : B[i];
        var f = cv.gaussianBlur(cv.divide(gray, { w: w, h: h, data: bb }, scale), sigma), od = new Uint8Array(f.data.length);
        for (i = 0; i < od.length; i++) { var v = f.data[i]; od[i] = v < 0 ? 0 : v > 255 ? 255 : v; }
        return { w: w, h: h, data: od };
    }
    var c = gaussKernelD(n, sigma), r = 5;
    for (i = 0; i < n; i++) c[i] = f32(c[i]);
    var c0 = c[5], h0 = c[0], h1 = c[1], h2 = c[2], h3 = c[3], h4 = c[4];
    var P = scratch('fdP', Float64Array, w + 10), tmp = scratch('g11TF', Float64Array, w * h), out = new Uint8Array(w * h);
    for (y = 0; y < h; y++) {
        var b = y * w;
        for (x = 0; x < w; x++) { var bv2 = B[b + x]; if (bv2 < 8) bv2 = 8; P[r + x] = f32(f32(scale * A[b + x]) / bv2); }
        for (x = 0; x < r; x++) P[x] = P[r + reflect101(x - r, w)];
        for (x = 0; x < r; x++) P[r + w + x] = P[r + reflect101(w + x, w)];
        g11h_F(P, tmp, b, w, c0, h0, h1, h2, h3, h4);
    }
    var v1 = function (yy) { for (var xx = 0; xx < w; xx++) { var vv = g11v1_F(tmp, w, h, yy, c0, h0, h1, h2, h3, h4, xx); out[yy * w + xx] = vv < 0 ? 0 : vv > 255 ? 255 : vv; } };
    for (y = 0; y < 5; y++) v1(y);
    for (y = 5; y + 3 <= h - 6; y += 4) g11v4_flat(tmp, out, w, y, c0, h0, h1, h2, h3, h4);
    for (; y < h; y++) v1(y);
    return { w: w, h: h, data: out };
};
cv.gaussianBlur = function cv_gaussianBlur(img, sigma, ksize) {
    var d = img.data, isU8 = d.constructor === Uint8Array;
    var n = ksize || (rint(sigma * (isU8 ? 3 : 4) * 2 + 1) | 1);
    if (n === 1) { var o1 = mat(img.w, img.h, d.constructor); o1.data.set(d); return o1; }
    return isU8 ? gaussBlur_U8(img, sigma, n) : d.constructor === Float32Array ? gaussBlur_F32(img, sigma, n) : gaussBlur_F64(img, sigma, n);
};

// ------------------------------------------------------------------ morphology (rectangular k x k kernel of ones, anchor k >> 1,
// out-of-image ignored = cv2's default border).  Window max/min by doubling: M_2s[i] = op(M_s[i], M_s[i+s]) in place, then
// out[i] = op(M_s[i], M_s[i + k - s]) with s the largest power of two <= k: log2(k)+1 contiguous passes per direction.
// (one copy per element type - see the note at gaussianBlur)
// 8-bit path: SWAR (four bytes per 32-bit word).  Horizontal: four image rows are packed into one word per column, so the window
// operation along x is a plain word-wise min/max (no unaligned access); the result is unpacked into a vertically padded image.  Vertical:
// the same doubling on whole rows, word-wise.  Bytewise unsigned min/max on packed bytes: t = (a | 0x80808080) - (b & 0x7f7f7f7f)
// has bit 7 of every byte set iff the low 7 bits of a >= those of b (no borrow crosses a byte); combined with the top bits that gives
// a >= b per byte, expanded to a full-byte mask by an imul.
var SW_H = 0x80808080 | 0, SW_L = 0x7f7f7f7f;
function swMin(a, b) {
    var t = ((a | SW_H) - (b & SW_L)) | 0, ge = ((a & ~b) | (~(a ^ b) & t)) & SW_H, m = Math.imul(ge >>> 7, 255);
    return (b & m) | (a & ~m);
}
function swMax(a, b) {
    var t = ((a | SW_H) - (b & SW_L)) | 0, ge = ((a & ~b) | (~(a ^ b) & t)) & SW_H, m = Math.imul(ge >>> 7, 255);
    return (a & m) | (b & ~m);
}
function dblPassMin(V, n, step) { for (var i = 0, lim = n - step; i < lim; i++) V[i] = swMin(V[i], V[i + step]); }
function dblPassMax(V, n, step) { for (var i = 0, lim = n - step; i < lim; i++) V[i] = swMax(V[i], V[i + step]); }
function finalMin(V, off, o, n) { for (var i = 0; i < n; i++) o[i] = swMin(V[i], V[i + off]); }
function finalMax(V, off, o, n) { for (var i = 0; i < n; i++) o[i] = swMax(V[i], V[i + off]); }
function packRows(img, y0, h, w, BW, a, ident) {      // BW[a + x] = rows y0..y0+3 of column x packed (rows >= h = identity bytes)
    var d = img.data, r0 = y0 * w, r1 = r0 + w, r2 = r1 + w, r3 = r2 + w, id8 = ident & 255, x;
    if (y0 + 3 < h) { for (x = 0; x < w; x++) BW[a + x] = d[r0 + x] | (d[r1 + x] << 8) | (d[r2 + x] << 16) | (d[r3 + x] << 24); }
    else for (x = 0; x < w; x++) BW[a + x] = d[r0 + x] | ((y0 + 1 < h ? d[r1 + x] : id8) << 8) | ((y0 + 2 < h ? d[r2 + x] : id8) << 16) | ((y0 + 3 < h ? d[r3 + x] : id8) << 24);
}
function unpackRows(BW, T, y0, h, w, S, rowOff) {       // rows into the vertically padded image T (stride S bytes, first data row rowOff)
    var b0 = (rowOff + y0) * S, b1 = b0 + S, b2 = b1 + S, b3 = b2 + S, x;
    if (y0 + 3 < h) { for (x = 0; x < w; x++) { var v = BW[x]; T[b0 + x] = v; T[b1 + x] = v >>> 8; T[b2 + x] = v >>> 16; T[b3 + x] = v >>> 24; } }
    else for (x = 0; x < w; x++) { var v2 = BW[x]; T[b0 + x] = v2; if (y0 + 1 < h) T[b1 + x] = v2 >>> 8; if (y0 + 2 < h) T[b2 + x] = v2 >>> 16; if (y0 + 3 < h) T[b3 + x] = v2 >>> 24; }
}
function vhFilter_U8(img, k, isMax, into) {
    var w = img.w, h = img.h, a = k >> 1, ident = isMax ? 0 : -1, out = into && into.data.length >= w * h ? into : mat(w, h, Uint8Array), o = out.data, x, y, s;
    out.w = w; out.h = h;
    var big = 1; while (big * 2 <= k) big *= 2;
    var S = (w + 3) & ~3, Wp = w + k - 1, Hp = h + k - 1, S4 = S >> 2;
    var BW = scratch('mvBW', Int32Array, Wp + 4), BWo = scratch('mvBWo', Int32Array, w + 4);
    var T = scratch('mvTu', Uint8Array, Hp * S + 4), T32 = new Int32Array(T.buffer, T.byteOffset, T.length >> 2);
    // vertical padding rows (identity)
    var idb = ident & 255;
    if (idb) { T.fill(idb, 0, a * S); T.fill(idb, (a + h) * S, Hp * S); } else { T.fill(0, 0, a * S); T.fill(0, (a + h) * S, Hp * S); }
    var off = k - big;
    for (y = 0; y < h; y += 4) {
        for (x = 0; x < a; x++) BW[x] = ident;                // (the doubling passes overwrite the padding)
        for (x = a + w; x < Wp; x++) BW[x] = ident;
        packRows(img, y, h, w, BW, a, ident);
        if (isMax) { for (s = 1; s < big; s *= 2) dblPassMax(BW, Wp, s); for (x = 0; x < w; x++) BWo[x] = swMax(BW[x], BW[x + off]); }
        else { for (s = 1; s < big; s *= 2) dblPassMin(BW, Wp, s); for (x = 0; x < w; x++) BWo[x] = swMin(BW[x], BW[x + off]); }
        unpackRows(BWo, T, y, h, w, S, a);
    }
    // vertical doubling over whole rows
    var N4 = Hp * S4;
    for (s = 1; s < big; s *= 2) { if (isMax) dblPassMax(T32, N4, s * S4); else dblPassMin(T32, N4, s * S4); }
    var voff = (k - big) * S4, rowBuf = scratch('mvRow', Int32Array, S4 + 1), rowB8 = new Uint8Array(rowBuf.buffer, rowBuf.byteOffset, rowBuf.length * 4);
    for (y = 0; y < h; y++) {
        var V = y * S4;
        if (isMax) { for (x = 0; x < S4; x++) rowBuf[x] = swMax(T32[V + x], T32[V + voff + x]); }
        else { for (x = 0; x < S4; x++) rowBuf[x] = swMin(T32[V + x], T32[V + voff + x]); }
        o.set(rowB8.subarray(0, w), y * w);
    }
    return out;
}
function vhFilter_F32(img, k, isMax, into) {
    var w = img.w, h = img.h, d = img.data, T = Float32Array, a = k >> 1;
    var ident = isMax ? -Infinity : Infinity, out = into && into.data.length >= w * h ? into : mat(w, h, T), o = out.data, x, y, i, s;
    out.w = w; out.h = h;
    var big = 1; while (big * 2 <= k) big *= 2;
    // horizontal: rows padded to Wp = w + k - 1 (a identity on the left, k-1-a on the right)
    var Wp = w + k - 1, N = Wp * h, M = scratch('mvM', T, N), R = scratch('mvR', T, N);
    for (y = 0; y < h; y++) {
        var rb = y * Wp, sb = y * w;
        for (x = 0; x < a; x++) M[rb + x] = ident;
        for (x = 0; x < w; x++) M[rb + a + x] = d[sb + x];
        for (x = a + w; x < Wp; x++) M[rb + x] = ident;
    }
    for (s = 1; s < big; s *= 2) {
        var lim = N - s;
        if (isMax) for (i = 0; i < lim; i++) { var p = M[i], q = M[i + s]; M[i] = p > q ? p : q; }
        else for (i = 0; i < lim; i++) { var p2 = M[i], q2 = M[i + s]; M[i] = p2 < q2 ? p2 : q2; }
    }
    var off = k - big, lim2 = N - off;
    if (isMax) for (i = 0; i < lim2; i++) { var p3 = M[i], q3 = M[i + off]; R[i] = p3 > q3 ? p3 : q3; }
    else for (i = 0; i < lim2; i++) { var p4 = M[i], q4 = M[i + off]; R[i] = p4 < q4 ? p4 : q4; }
    // vertical: gather the valid w columns of R into rows padded with a identity rows on top and k-1-a at the bottom
    var Hp = h + k - 1, V = scratch('mvV', T, Hp * w), V2 = scratch('mvV2f', T, 1);
    for (i = 0; i < a * w; i++) V[i] = ident;
    for (y = 0; y < h; y++) { var rr = y * Wp, vb = (a + y) * w; for (x = 0; x < w; x++) V[vb + x] = R[rr + x]; }
    for (i = (a + h) * w; i < Hp * w; i++) V[i] = ident;
    var N2 = Hp * w;
    for (s = 1; s < big; s *= 2) {
        var step = s * w, l3 = N2 - step;
        if (isMax) for (i = 0; i < l3; i++) { var p5 = V[i], q5 = V[i + step]; V[i] = p5 > q5 ? p5 : q5; }
        else for (i = 0; i < l3; i++) { var p6 = V[i], q6 = V[i + step]; V[i] = p6 < q6 ? p6 : q6; }
    }
    var off2 = (k - big) * w, l4 = h * w;
    if (isMax) for (i = 0; i < l4; i++) { var p7 = V[i], q7 = V[i + off2]; o[i] = p7 > q7 ? p7 : q7; }
    else for (i = 0; i < l4; i++) { var p8 = V[i], q8 = V[i + off2]; o[i] = p8 < q8 ? p8 : q8; }
    return out;
}
function vhFilter(img, k, isMax, into) { return img.data.constructor === Uint8Array ? vhFilter_U8(img, k, isMax, into) : vhFilter_F32(img, k, isMax, into); }
// optional `into` = an image to write the result into (no allocation)
cv.dilate = function cv_dilate(img, k, into) { return k <= 1 ? { w: img.w, h: img.h, data: img.data.slice() } : vhFilter(img, k, true, into); };
cv.erode = function cv_erode(img, k, into) { return k <= 1 ? { w: img.w, h: img.h, data: img.data.slice() } : vhFilter(img, k, false, into); };
cv.morphOpen = function cv_morphOpen(img, k, into) {
    if (k <= 1) return { w: img.w, h: img.h, data: img.data.slice() };
    var T = img.data.constructor, e = scratch('mopen', T, img.w * img.h), tmp = { w: img.w, h: img.h, data: e };
    vhFilter(img, k, false, tmp);
    return vhFilter(tmp, k, true, into);
};

// ------------------------------------------------------------------ Laplacian / template matching
// cv2.Laplacian(img, CV_32F) (ksize 1: [0 1 0; 1 -4 1; 0 1 0], reflect-101 border)
cv.laplacian = function cv_laplacian(img) {
    var w = img.w, h = img.h, d = img.data, out = mat(w, h, Float32Array), o = out.data, x, y;
    for (y = 0; y < h; y++) {
        var up = reflect101(y - 1, h) * w, dn = reflect101(y + 1, h) * w, b = y * w;
        for (x = 0; x < w; x++) {
            var xl = x > 0 ? x - 1 : reflect101(x - 1, w), xr = x < w - 1 ? x + 1 : reflect101(x + 1, w);
            o[b + x] = d[up + x] + d[dn + x] + d[b + xl] + d[b + xr] - 4 * d[b + x];
        }
    }
    return out;
};
// variance of the Laplacian (population variance, as numpy .var())
cv.laplacianVar = function cv_laplacianVar(img) {
    var l = cv.laplacian(img).data, n = l.length, s = 0, s2 = 0;
    for (var i = 0; i < n; i++) { s += l[i]; s2 += l[i] * l[i]; }
    var m = s / n;
    return s2 / n - m * m;
};
function mtCorr1(d, W, tz, w, h, base) {
    var num = 0;
    for (var j = 0; j < h; j++) { var rb = base + j * W, tb = j * w; for (var i = 0; i < w; i++) num += d[rb + i] * tz[tb + i]; }
    return num;
}
function mtCorr4(d, W, tz, w, h, base, out) {      // correlation of the template with the windows at base .. base+3 (each its own running sum, rows then columns)
    var a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (var j = 0; j < h; j++) {
        var rb = base + j * W, tb = j * w, d0 = d[rb], d1 = d[rb + 1], d2 = d[rb + 2];
        for (var i = 0; i < w; i++) {
            var d3 = d[rb + i + 3], t = tz[tb + i];
            a0 += d0 * t; a1 += d1 * t; a2 += d2 * t; a3 += d3 * t;
            d0 = d1; d1 = d2; d2 = d3;
        }
    }
    out[0] = a0; out[1] = a1; out[2] = a2; out[3] = a3;
}
// cv2.matchTemplate(img, tpl, TM_CCOEFF_NORMED) -> Float32 image (W-w+1) x (H-h+1)
cv.matchTemplateCcoeffNormed = function cv_matchTemplateCcoeffNormed(img, tpl) {
    var W = img.w, H = img.h, w = tpl.w, h = tpl.h, d = img.data, t = tpl.data, ow = W - w + 1, oh = H - h + 1, n = w * h;
    var out = mat(ow, oh, Float32Array), o = out.data, x, y, i, j;
    var tm = 0; for (i = 0; i < n; i++) tm += t[i]; tm /= n;
    var tz = new Float64Array(n), tn = 0;
    for (i = 0; i < n; i++) { tz[i] = t[i] - tm; tn += tz[i] * tz[i]; }
    tn = Math.sqrt(tn);
    var I = cv.integral(img), sq = new Float64Array((W + 1) * (H + 1)), S = W + 1;
    for (y = 0; y < H; y++) { var rs = 0; for (x = 0; x < W; x++) { rs += d[y * W + x] * d[y * W + x]; sq[(y + 1) * S + x + 1] = sq[y * S + x + 1] + rs; } }
    var id = I.data;
    var nums = new Float64Array(4);
    for (y = 0; y < oh; y++) for (x = 0; x < ow; x++) {
        var num;
        if ((x & 3) === 0 && x + 3 < ow) mtCorr4(d, W, tz, w, h, y * W + x, nums);       // four neighbouring windows at once (same per-window summation order)
        if (x + 3 - (x & 3) < ow) num = nums[x & 3]; else num = mtCorr1(d, W, tz, w, h, y * W + x);
        var s1 = id[(y + h) * S + x + w] - id[y * S + x + w] - id[(y + h) * S + x] + id[y * S + x];
        var s2 = sq[(y + h) * S + x + w] - sq[y * S + x + w] - sq[(y + h) * S + x] + sq[y * S + x];
        var v = s2 - s1 * s1 / n, tt = Math.sqrt(v > 0 ? v : 0) * tn, res;
        if (tt < 1e-9 && Math.abs(num) < 1e-9) res = 1;                  // flat window and flat template: cv2 reports a perfect match
        else if (Math.abs(num) < tt) res = num / tt;
        else if (Math.abs(num) < tt * 1.125) res = num > 0 ? 1 : -1;
        else res = 0;
        o[y * ow + x] = res;
    }
    return out;
};

O.cv = cv;
})(globalThis.OMR = globalThis.OMR || {});
