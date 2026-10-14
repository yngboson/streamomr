(function (O) {
'use strict';
// tracks.js - omr/tracks.py: timing tracks.  Measure where the ticks along the four edges really are and bend the page flat again.
// Track = {side, nominal: [[x, y] mm], d: Float64Array(2n) displacement mm (NaN where not found), resp: Float64Array(n)}
var f32 = Math.fround;
var cv = O.cv, cvx = O.cvx, lay = O.layout;

var RING_MM = 2.6, DETECT_SCALE = 0.5, MIN_RESP = 0.22, NMS_MM = 3.0, WIN_ALONG_MM = 7.5;
var WIN_ACROSS_MM = { L: 8.0, R: 8.0, T: 5.5, B: 5.5 };
var EXT_ACROSS_MM = { L: 14.0, R: 14.0, T: 7.0, B: 9.0 };
var EXT_WIN_MM = 2.2, EXT_DAMP = 0.9, EXT_MISSES = 2, QUAD = 0.08, LAMBDA = 0.25, MISS = 0.3, PRIOR = 0.002;
var ANCHOR_MM = 2.0, ANCHOR_SLOPE = 0.12, ANCHOR_COST = 1.0, ROUGH_MAX_MM = 0.5, MAX_GAP = 12, PRESENT_FRAC = 0.3, MIN_FRAC = 0.5,
    FLAT_MM = 0.5, MAX_DISP_MM = 14.0, OUTLIER_MM = 1.2;

function Track(side, nominal, d, resp) { return { side: side, nominal: nominal, d: d, resp: resp }; }
function found(t) { var c = 0; for (var i = 0; i < t.resp.length; i++) if (isFinite(t.d[2 * i])) c++; return c; }
function frac(t) { return found(t) / Math.max(t.nominal.length, 1); }
function foundIdx(d) { var idx = []; for (var i = 0; i < d.length / 2; i++) if (isFinite(d[2 * i])) idx.push(i); return idx; }

// omr/tracks.py:67  band: {w,h,data:Float32Array} -> Float32Array response
function response(band, ppm, size) {
    var iw = Math.max(rint(size[0] * ppm), 3), ih = Math.max(rint(size[1] * ppm), 3);
    var ow = rint((size[0] + RING_MM) * ppm), oh = rint((size[1] + RING_MM) * ppm);
    var inner = cv.boxFilter(band, iw, true, ih).data, outer = cv.boxFilter(band, ow, true, oh).data;
    var aIn = iw * ih, aOut = ow * oh, den = Math.max(aOut - aIn, 1), n = inner.length, out = new Float32Array(n);
    for (var i = 0; i < n; i++) {
        var ring = f32(f32(f32(outer[i] * aOut) - f32(inner[i] * aIn)) / den);
        var q = f32(f32(ring - inner[i]) / Math.max(ring, 24.0));
        out[i] = q < -1.0 ? -1.0 : q > 1.5 ? 1.5 : q;
    }
    return { w: band.w, h: band.h, data: out };
}
function rint(x) { return cv.rint(x); }

// omr/tracks.py:79  -> [[fy, fx, value]]
function peaks(resp, ppm) {
    var k = rint(NMS_MM * ppm) | 1, mx = cv.dilate(resp, k).data, d = resp.data, W = resp.w, H = resp.h, out = [];
    for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
        var i = y * W + x;
        if (!(d[i] >= mx[i] && d[i] >= f32(MIN_RESP))) continue;
        var fy = y, fx = x;
        if (y > 0 && y < H - 1) { var a = d[i - W], b = d[i], c = d[i + W]; fy = f32(fy + f32(f32(0.5 * f32(a - c)) / f32(f32(f32(a - f32(2 * b)) + c) - f32(1e-9)))); }
        if (x > 0 && x < W - 1) { var a2 = d[i - 1], b2 = d[i], c2 = d[i + 1]; fx = f32(fx + f32(f32(0.5 * f32(a2 - c2)) / f32(f32(f32(a2 - f32(2 * b2)) + c2) - f32(1e-9)))); }
        out.push([fy, fx, d[i]]);
    }
    return out;
}

// omr/tracks.py:96  d: Float64Array(2m) interleaved displacements, anchor [ax, ay] or null, dist: scalar or Float64Array(m)
function anchorCost(d, m, anchor, dist) {
    var out = new Float64Array(m);
    if (!anchor) return out;
    for (var i = 0; i < m; i++) {
        var di = typeof dist === 'number' ? dist : dist[i];
        out[i] = ANCHOR_COST * Math.max(0.0, Math.hypot(d[2 * i] - anchor[0], d[2 * i + 1] - anchor[1]) - ANCHOR_MM - ANCHOR_SLOPE * di);
    }
    return out;
}

// omr/tracks.py:104  allc: {X,Y,R: Float64Array(m)}, T: Int32Array(m) (non-decreasing) -> Map tick -> [x, y, resp]
function chain(allc, T, n, a0, a1, dist0, dist1) {
    var tot = T.length, i, k, j;
    if (tot === 0) return new Map();
    var start = new Int32Array(n + 1);
    for (k = 0, i = 0; k <= n; k++) { while (i < tot && T[i] < k) i++; start[k] = i; }       // np.searchsorted(T, arange(n + 1)) (left)
    var X = allc.X, Y = allc.Y, R = allc.R, unary = new Float64Array(tot), xy = new Float64Array(2 * tot);
    for (i = 0; i < tot; i++) { unary[i] = R[i] - PRIOR * Math.hypot(X[i], Y[i]); xy[2 * i] = X[i]; xy[2 * i + 1] = Y[i]; }
    var d0 = new Float64Array(tot);
    for (i = 0; i < tot; i++) d0[i] = (dist0 || 0) + T[i] * lay.TRACK_PITCH;
    var ac0 = anchorCost(xy, tot, a0, d0), first = new Float64Array(tot), S = new Float64Array(tot), Sg = new Float64Array(tot), prev = new Int32Array(tot).fill(-1);
    for (i = 0; i < tot; i++) { first[i] = unary[i] - MISS * T[i] - ac0[i]; S[i] = first[i]; Sg[i] = S[i] + MISS * T[i]; }
    for (k = 0; k < n; k++) {
        var a = start[k], b = start[k + 1], p0 = start[Math.max(0, k - MAX_GAP - 1)];
        if (a === b || p0 === a) continue;
        var newS = new Float64Array(b - a), newPrev = new Int32Array(b - a);
        for (i = a; i < b; i++) {
            var bestv = -Infinity, arg = 0;
            for (j = p0; j < a; j++) {
                var dd = Math.hypot(X[i] - X[j], Y[i] - Y[j]), tr = Sg[j] - LAMBDA * dd - QUAD * dd * dd;
                if (tr > bestv) { bestv = tr; arg = j; }
            }
            var best = bestv + unary[i] - MISS * (k - 1);
            if (best > first[i]) { newS[i - a] = best; newPrev[i - a] = arg; } else { newS[i - a] = first[i]; newPrev[i - a] = -1; }
        }
        for (i = a; i < b; i++) { S[i] = newS[i - a]; Sg[i] = S[i] + MISS * k; prev[i] = newPrev[i - a]; }
    }
    var ends = new Float64Array(tot);
    for (i = 0; i < tot; i++) ends[i] = S[i] - MISS * (n - 1 - T[i]);
    if (a1) {
        var d1 = new Float64Array(tot);
        for (i = 0; i < tot; i++) d1[i] = (dist1 || 0) + (n - 1 - T[i]) * lay.TRACK_PITCH;
        var ac1 = anchorCost(xy, tot, a1, d1);
        for (i = 0; i < tot; i++) ends[i] = ends[i] - ac1[i];
    }
    var cur = 0;
    for (i = 1; i < tot; i++) if (ends[i] > ends[cur]) cur = i;
    var out = new Map();
    while (cur >= 0) { out.set(T[cur], [X[cur], Y[cur], R[cur]]); cur = prev[cur]; }
    return out;
}

var ENDS = { L: ['TL', 'BL'], R: ['TR', 'BR'], T: ['TL', 'TR'], B: ['BL', 'BR'] };
var _mc = null;
function markerCentres() {
    if (!_mc) {
        _mc = {};
        var rs = lay.markerRects(0);
        for (var c in rs) _mc[c] = [rs[c][0] + rs[c][2] / 2, rs[c][1] + rs[c][3] / 2];
    }
    return _mc;
}

// omr/tracks.py:154  canvas: uint8 grey {w,h,data}; centers [[x, y] mm]; anchors {corner: [dx, dy]}
function detectSide(canvas, ppm, side, centers, anchors) {
    var H = canvas.h, W = canvas.w, horiz = side === 'T' || side === 'B', WA = WIN_ACROSS_MM[side], size, across0, WB, lo, hi, r0, r1, c0, c1;
    var n = centers.length, k, i;
    if (horiz) {
        size = [lay.TRACK_LEN - 0.4, lay.TRACK_TB_H - 0.3]; across0 = centers[0][1]; WB = Math.max(WA, EXT_ACROSS_MM[side]);
        lo = across0 - WB - 2.5; hi = across0 + WB + 2.5;
        r0 = Math.floor(Math.max(lo * ppm, 0)); r1 = Math.floor(Math.min(hi * ppm, H));
        c0 = Math.floor(Math.max((centers[0][0] - WIN_ALONG_MM - 3) * ppm, 0)); c1 = Math.floor(Math.min((centers[n - 1][0] + WIN_ALONG_MM + 3) * ppm, W));
    } else {
        size = [lay.TRACK_LR_W - 0.4, lay.TRACK_LEN]; across0 = centers[0][0]; WB = Math.max(WA, EXT_ACROSS_MM[side]);
        lo = across0 - WB - 2.5; hi = across0 + WB + 2.5;
        c0 = Math.floor(Math.max(lo * ppm, 0)); c1 = Math.floor(Math.min(hi * ppm, W));
        r0 = Math.floor(Math.max((centers[0][1] - WIN_ALONG_MM - 3) * ppm, 0)); r1 = Math.floor(Math.min((centers[n - 1][1] + WIN_ALONG_MM + 3) * ppm, H));
    }
    var d = new Float64Array(2 * n).fill(NaN), rsp = new Float64Array(n);
    if (r1 - r0 < 8 || c1 - c0 < 8) return Track(side, centers, d, rsp);
    var sub = cvx.mk(c1 - c0, r1 - r0), sw = c1 - c0;
    for (var y = r0; y < r1; y++) sub.data.set(canvas.data.subarray(y * W + c0, y * W + c1), (y - r0) * sw);
    var bu = cv.resize(sub, rint(sub.w * DETECT_SCALE), rint(sub.h * DETECT_SCALE), 'area', DETECT_SCALE, DETECT_SCALE);
    var band = { w: bu.w, h: bu.h, data: Float32Array.from(bu.data) };
    var pk = peaks(response(band, ppm * DETECT_SCALE, size), ppm * DETECT_SCALE);
    if (!pk.length) return Track(side, centers, d, rsp);
    var np = pk.length, ptsmm = new Float64Array(2 * np), val = new Float64Array(np);
    for (i = 0; i < np; i++) {
        ptsmm[2 * i] = ((pk[i][1] + 0.5) / DETECT_SCALE - 0.5 + c0) / ppm;
        ptsmm[2 * i + 1] = ((pk[i][0] + 0.5) / DETECT_SCALE - 0.5 + r0) / ppm;
        val[i] = pk[i][2];
    }
    var ia = horiz ? 0 : 1, ic = horiz ? 1 : 0, acx = [], acy = [], acr = [], kk = [];
    for (k = 0; k < n; k++) for (i = 0; i < np; i++) {
        var rx = ptsmm[2 * i] - centers[k][0], ry = ptsmm[2 * i + 1] - centers[k][1], ra = ia === 0 ? rx : ry, rc = ic === 0 ? rx : ry;
        if (Math.abs(ra) <= WIN_ALONG_MM && Math.abs(rc) <= WA) { kk.push(k); acx.push(rx); acy.push(ry); acr.push(val[i]); }
    }
    anchors = anchors || {};
    var ends = ENDS[side], mc = markerCentres(), ca = ends[0], cb = ends[1];
    var a0 = anchors[ca] || null, a1 = anchors[cb] || null;
    var res = chain({ X: Float64Array.from(acx), Y: Float64Array.from(acy), R: Float64Array.from(acr) }, Int32Array.from(kk), n, a0, a1,
        Math.hypot(centers[0][0] - mc[ca][0], centers[0][1] - mc[ca][1]), Math.hypot(centers[n - 1][0] - mc[cb][0], centers[n - 1][1] - mc[cb][1]));
    res.forEach(function (c, t) { d[2 * t] = c[0]; d[2 * t + 1] = c[1]; rsp[t] = c[2]; });
    extend(d, rsp, centers, ptsmm, val);
    dropOutliers(d, rsp);
    if (roughness(d) > ROUGH_MAX_MM) { d.fill(NaN); rsp.fill(0); }
    return Track(side, centers, d, rsp);
}

// omr/tracks.py:208
function roughness(d) {
    var idx = foundIdx(d);
    if (idx.length < 5) return 0.0;
    var dev = [];
    for (var q = 1; q < idx.length - 1; q++) {
        var a = idx[q - 1], m = idx[q], b = idx[q + 1], t = (m - a) / (b - a);
        var ex = d[2 * m] - ((1 - t) * d[2 * a] + t * d[2 * b]), ey = d[2 * m + 1] - ((1 - t) * d[2 * a + 1] + t * d[2 * b + 1]);
        dev.push(Math.hypot(ex, ey) / (1.0 + 0.15 * (b - a - 2)));
    }
    return cvx.median(dev);
}

// omr/tracks.py:219
function extend(d, rsp, centers, ptsmm, val) {
    var idx = foundIdx(d), np = val.length;
    if (idx.length < 3 || !np) return;
    var n = d.length / 2, dirs = [[-1, idx[0], idx[1]], [1, idx[idx.length - 1], idx[idx.length - 2]]];
    for (var di = 0; di < 2; di++) {
        var step = dirs[di][0], k0 = dirs[di][1], k1 = dirs[di][2];
        var cx = d[2 * k0], cy = d[2 * k0 + 1], sx = (d[2 * k0] - d[2 * k1]) / Math.abs(k0 - k1), sy = (d[2 * k0 + 1] - d[2 * k1 + 1]) / Math.abs(k0 - k1);
        var misses = 0, k = k0 + step;
        while (k >= 0 && k < n && misses < EXT_MISSES) {
            cx += sx; cy += sy; sx *= EXT_DAMP; sy *= EXT_DAMP;
            var bestj = -1;
            for (var j = 0; j < np; j++) {
                var rx = ptsmm[2 * j] - centers[k][0] - cx, ry = ptsmm[2 * j + 1] - centers[k][1] - cy;
                if (Math.hypot(rx, ry) <= EXT_WIN_MM && (bestj < 0 || val[j] > val[bestj])) bestj = j;
            }
            if (bestj >= 0) {
                if (!isFinite(d[2 * k]) || val[bestj] > rsp[k]) {
                    d[2 * k] = ptsmm[2 * bestj] - centers[k][0]; d[2 * k + 1] = ptsmm[2 * bestj + 1] - centers[k][1]; rsp[k] = val[bestj];
                }
                cx = d[2 * k]; cy = d[2 * k + 1]; misses = 0;
            } else misses++;
            k += step;
        }
    }
}

// omr/tracks.py:244
function dropOutliers(d, rsp) {
    for (var it = 0; it < 6; it++) {
        var idx = foundIdx(d), L = idx.length, q;
        if (L < 4) return;
        var pred = new Float64Array(2 * L), lim = new Float64Array(L);
        for (q = 1; q < L - 1; q++) {
            var a = idx[q - 1], m = idx[q], b = idx[q + 1], t = (m - a) / (b - a);
            pred[2 * q] = (1 - t) * d[2 * a] + t * d[2 * b]; pred[2 * q + 1] = (1 - t) * d[2 * a + 1] + t * d[2 * b + 1];
            lim[q] = OUTLIER_MM + 0.15 * (b - a);
        }
        var ends = [[0, idx[0], idx[1], idx[2]], [L - 1, idx[L - 1], idx[L - 2], idx[L - 3]]];
        for (var e = 0; e < 2; e++) {
            var pos = ends[e][0], i0 = ends[e][1], i1 = ends[e][2], i2 = ends[e][3], f = (i0 - i1) / (i2 - i1);
            pred[2 * pos] = d[2 * i1] + (d[2 * i2] - d[2 * i1]) * f; pred[2 * pos + 1] = d[2 * i1 + 1] + (d[2 * i2 + 1] - d[2 * i1 + 1]) * f;
            lim[pos] = 2.0 * OUTLIER_MM;
        }
        var worst = -1, wv = -1, any = false;
        for (q = 0; q < L; q++) {
            var nr = Math.hypot(d[2 * idx[q]] - pred[2 * q], d[2 * idx[q] + 1] - pred[2 * q + 1]);
            if (nr > lim[q]) { any = true; var ratio = nr / lim[q]; if (ratio > wv) { wv = ratio; worst = q; } }
        }
        if (!any) return;
        d[2 * idx[worst]] = NaN; d[2 * idx[worst] + 1] = NaN; rsp[idx[worst]] = 0.0;
    }
}

// =============================================================================== the field
var XL = lay.TRACK_EDGE_LR + lay.TRACK_LR_W / 2, XR = lay.PAGE_W - XL, YT = lay.TRACK_EDGE_TB + lay.TRACK_TB_H / 2, YB = lay.PAGE_H - YT;

function interp(x, xs, ys) {               // np.interp with clamping, xs increasing
    var n = xs.length;
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    var lo = 0, hi = n - 1;
    while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
    var dx = xs[hi] - xs[lo];
    if (dx === 0) return ys[hi];
    return ys[lo] + (x - xs[lo]) * ((ys[hi] - ys[lo]) / dx);
}

// omr/tracks.py:285  Field: displacement d(x, y) (mm) of the printed page in the canvas, a Coons patch of the four edges' tick displacements
function Field(tracks) {
    var self = { tracks: tracks, sidesOk: {}, c: {}, fn: {}, maxDisp: null };
    var end = {}, s, i;
    for (s in tracks) self.sidesOk[s] = frac(tracks[s]) >= MIN_FRAC;
    for (s in tracks) {
        var idx = foundIdx(tracks[s].d);
        end[s] = idx.length && self.sidesOk[s] ? [[tracks[s].d[2 * idx[0]], tracks[s].d[2 * idx[0] + 1]], [tracks[s].d[2 * idx[idx.length - 1]], tracks[s].d[2 * idx[idx.length - 1] + 1]]] : [null, null];
    }
    function corner(a, b) {
        var v = [a, b].filter(function (x) { return x !== null; });
        if (!v.length) return [0, 0];
        var sx = 0, sy = 0;
        v.forEach(function (p) { sx += p[0]; sy += p[1]; });
        return [sx / v.length, sy / v.length];
    }
    self.c = { TL: corner(end.L[0], end.T[0]), TR: corner(end.R[0], end.T[1]), BL: corner(end.L[1], end.B[0]), BR: corner(end.R[1], end.B[1]) };
    var defs = { L: [YT, YB, 'TL', 'BL'], R: [YT, YB, 'TR', 'BR'], T: [XL, XR, 'TL', 'TR'], B: [XL, XR, 'BL', 'BR'] };
    for (s in defs) {
        var t = tracks[s], lo = defs[s][0], hi = defs[s][1], clo = self.c[defs[s][2]], chi = self.c[defs[s][3]];
        var ix = self.sidesOk[s] ? foundIdx(t.d) : [], ax = (s === 'L' || s === 'R') ? 1 : 0;
        var u = [lo], vx = [clo[0]], vy = [clo[1]];
        ix.forEach(function (k) { u.push(t.nominal[k][ax]); vx.push(t.d[2 * k]); vy.push(t.d[2 * k + 1]); });
        u.push(hi); vx.push(chi[0]); vy.push(chi[1]);
        self.fn[s] = { u: u, vx: vx, vy: vy };
    }
    return self;
}
function edge(f, s, u) { var e = f.fn[s]; return [interp(u, e.u, e.vx), interp(u, e.u, e.vy)]; }
function clip(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

// Field.__call__ for one point: nominal page position (mm) -> [dx, dy] mm
function fieldAt(f, x, y) {
    var s = clip((x - XL) / (XR - XL), 0, 1), t = clip((y - YT) / (YB - YT), 0, 1), xc = clip(x, XL, XR), yc = clip(y, YT, YB), c = f.c;
    var eT = edge(f, 'T', xc), eB = edge(f, 'B', xc), eL = edge(f, 'L', yc), eR = edge(f, 'R', yc), out = [0, 0];
    for (var k = 0; k < 2; k++)
        out[k] = ((1 - t) * eT[k] + t * eB[k] + (1 - s) * eL[k] + s * eR[k]
            - ((1 - s) * (1 - t) * c.TL[k] + s * (1 - t) * c.TR[k] + (1 - s) * t * c.BL[k] + s * t * c.BR[k]));
    return out;
}
// Field.grid: -> Float64Array(len(py) * len(px) * 2)
function fieldGrid(f, px, py) {
    var nx = px.length, ny = py.length, out = new Float64Array(nx * ny * 2), c = f.c, i, j, k;
    var s = new Float64Array(nx), eT = [], eB = [], xc = new Float64Array(nx);
    for (i = 0; i < nx; i++) { s[i] = clip((px[i] - XL) / (XR - XL), 0, 1); xc[i] = clip(px[i], XL, XR); eT.push(edge(f, 'T', xc[i])); eB.push(edge(f, 'B', xc[i])); }
    for (j = 0; j < ny; j++) {
        var t = clip((py[j] - YT) / (YB - YT), 0, 1), yc = clip(py[j], YT, YB), eL = edge(f, 'L', yc), eR = edge(f, 'R', yc);
        for (i = 0; i < nx; i++) for (k = 0; k < 2; k++)
            out[(j * nx + i) * 2 + k] = ((1 - t) * eT[i][k] + t * eB[i][k] + (1 - s[i]) * eL[k] + s[i] * eR[k]
                - ((1 - s[i]) * (1 - t) * c.TL[k] + s[i] * (1 - t) * c.TR[k] + (1 - s[i]) * t * c.BL[k] + s[i] * t * c.BR[k]));
    }
    return out;
}
function usable(f) { var ok = f.sidesOk; return (ok.L && ok.R) || (ok.T && ok.B); }
function maxDisp(f) {
    if (f.maxDisp !== null) return f.maxDisp;
    var m = 0;
    for (var j = 0; j < 20; j++) for (var i = 0; i < 15; i++) {
        var d = fieldAt(f, XL + (XR - XL) * i / 14, YT + (YB - YT) * j / 19);
        m = Math.max(m, Math.hypot(d[0], d[1]));
    }
    return f.maxDisp = m;
}
function info(f) {
    var t = f.tracks, sum = 0, nt = 0, sides = 0;
    for (var s in t) { sum += frac(t[s]); nt++; if (f.sidesOk[s]) sides++; }
    return { tracks_found: sum / nt, tracks_sides: sides, tracks_max_mm: maxDisp(f) };
}

// omr/tracks.py:370  Field of a sheet's tracks in the corner-marker warp, or null when they cannot be trusted
function measure(canvas, ppm, anchors) {
    var cen = lay.trackTickCenters(), tr = {};
    ['B', 'T'].forEach(function (s) { tr[s] = detectSide(canvas, ppm, s, cen[s], anchors); });
    if (Math.max(frac(tr.B), frac(tr.T)) < PRESENT_FRAC) return null;
    ['L', 'R'].forEach(function (s) { tr[s] = detectSide(canvas, ppm, s, cen[s], anchors); });
    var f = Field(tr);
    if (!usable(f) || maxDisp(f) > MAX_DISP_MM) return null;
    return f;
}

// omr/tracks.py:384  canvas resampled so that every nominal page position (mm * ppm) holds what the field says is printed there
function rectify(canvas, field, ppm, border) {
    if (border === undefined) border = 255;
    var H = canvas.h, W = canvas.w, ds = Math.max(rint(1.5 * ppm), 2), nW = Math.ceil(W / ds), nH = Math.ceil(H / ds), i, j;
    var px = new Float64Array(nW), py = new Float64Array(nH);
    for (i = 0; i < nW; i++) px[i] = ((i + 0.5) * ds - 0.5) / ppm;
    for (j = 0; j < nH; j++) py[j] = ((j + 0.5) * ds - 0.5) / ppm;
    var d = fieldGrid(field, px, py), mx = cvx.mk(nW, nH, Float32Array), my = cvx.mk(nW, nH, Float32Array);
    for (j = 0; j < nH; j++) for (i = 0; i < nW; i++) {
        mx.data[j * nW + i] = (px[i] + d[(j * nW + i) * 2]) * ppm;
        my.data[j * nW + i] = (py[j] + d[(j * nW + i) * 2 + 1]) * ppm;
    }
    // cv2.resize(map, (nW*ds, nH*ds), INTER_LINEAR) then cv2.remap(canvas, ...) fused: the coarse maps are interpolated per output pixel
    // (same half-pixel-centre weights, float32 rounding of the interpolated map), so no full-size float maps are built
    var sc = 1 / ds, xo = new Int32Array(W), xo1 = new Int32Array(W), xf = new Float32Array(W), xg = new Float32Array(W), x;
    for (x = 0; x < W; x++) {
        var fx = (x + 0.5) * sc - 0.5, k = Math.floor(fx);
        fx -= k;
        if (k < 0) { fx = 0; k = 0; }
        if (k >= nW - 1) { fx = 0; k = nW - 1; }
        xo[x] = k; xo1[x] = Math.min(k + 1, nW - 1); xf[x] = f32(fx); xg[x] = f32(1 - xf[x]);
    }
    var nRuns = 0, runStart = new Int32Array(W + 1), runEnd = new Int32Array(W + 1), runK = new Int32Array(W + 1), runK2 = new Int32Array(W + 1);
    for (x = 0; x < W; x++) {
        if (nRuns === 0 || xo[x] !== runK[nRuns - 1] || xo1[x] !== runK2[nRuns - 1]) { runStart[nRuns] = x; runK[nRuns] = xo[x]; runK2[nRuns] = xo1[x]; nRuns++; }
        runEnd[nRuns - 1] = x + 1;
    }
    var W1 = W - 1, H1 = H - 1, out = cvx.mk(W, H), od = out.data, src = canvas.data, rowx = new Float32Array(nW), rowy = new Float32Array(nW);
    for (j = 0; j < H; j++) {
        var fy = (j + 0.5) * sc - 0.5, ky = Math.floor(fy);
        fy -= ky;
        if (ky < 0) { fy = 0; ky = 0; }
        if (ky >= nH - 1) { fy = 0; ky = nH - 1; }
        var k1 = Math.min(ky + 1, nH - 1), wy1 = f32(fy), wy0 = f32(1 - wy1), b0 = ky * nW, b1 = k1 * nW;
        for (i = 0; i < nW; i++) {
            rowx[i] = f32(f32(mx.data[b0 + i] * wy0) + f32(mx.data[b1 + i] * wy1));
            rowy[i] = f32(f32(my.data[b0 + i] * wy0) + f32(my.data[b1 + i] * wy1));
        }
        var ob = j * W;
        for (var run = 0; run < nRuns; run++) {                  // pixels of one run share the two coarse columns: their map values are hoisted
            var kx = runK[run], k2 = runK2[run], rx0 = rowx[kx], rx1 = rowx[k2], ry0 = rowy[kx], ry1 = rowy[k2], xe = runEnd[run];
            for (x = runStart[run]; x < xe; x++) {
                var wx1 = xf[x], wx0 = xg[x];
                var sx = f32(f32(rx0 * wx0) + f32(rx1 * wx1)), sy = f32(f32(ry0 * wx0) + f32(ry1 * wx1));
                if (sx >= 0 && sy >= 0 && sx < W1 && sy < H1) {        // interior (truncation = floor for sx, sy >= 0)
                    var ix = sx | 0, iy = sy | 0, ax = sx - ix, ay = sy - iy, pp = iy * W + ix, bx = 1 - ax, by = 1 - ay;
                    od[ob + x] = Math.floor((src[pp] * bx + src[pp + 1] * ax) * by + (src[pp + W] * bx + src[pp + W + 1] * ax) * ay + 0.5);
                } else if (sx < -1 || sx >= W || sy < -1 || sy >= H) od[ob + x] = border;       // all four neighbours outside (what cvx.sample returns then)
                else od[ob + x] = cvx.sample(src, W, H, sx, sy, 'constant', border, true);
            }
        }
    }
    return out;
}

var FLAT = { flat: true };      // "measured, and the sheet is flat": recognize keeps the plain warp

O.tracks = { Track: Track, measure: measure, rectify: rectify, FLAT: FLAT, FLAT_MM: FLAT_MM, MIN_FRAC: MIN_FRAC, detectSide: detectSide,
    Field: Field, fieldAt: fieldAt, maxDisp: maxDisp, info: info, usable: usable };
})(globalThis.OMR = globalThis.OMR || {});
