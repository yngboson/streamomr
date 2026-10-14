(function (O) {
'use strict';
// illum.js - omr/illum.py: lighting-robust bubble darkness (method "E" = "final", the shipped one; the benchmark alternatives
// B / D10 / D20 / F are not ported).  Layout-guided paper sampling in a thin annulus around every bubble, per-pixel plane / cluster
// paper model, edge flag, local contrast (glare) correction.
// Dtypes follow numpy: pixel gathers and darkness are float32, the paper model is float64 after the halo lift (float32 x float64).
var f32 = Math.fround;
var cvx = O.cvx, pctRows = O.fastops.pctRows;

var INNER_DISC_FRAC = 0.72, ANN_IN_MM = 0.25, ANN_OUT_MM = 0.75, N_SECT = 8, SECT_PCT = 97, OUTLIER_TOL = 0.08, PLANE_TOL = 0.07,
    EDGE_TOL = 0.05, CLUSTER_TOL = 0.06, SIGMA_MM = 1.3, PAPER_GAIN = 1.06, MIN_PAPER = 20.0, NEIGHBOUR_MM = 9.0, HOT_DARKNESS = 0.2,
    HALO_MAX = 1.12, HALO_NEIGHBOURS = 4, HALO_REACH_MM = 12.0, C_REF_PCT = 60, C_NEIGHBOURS = 14, C_APPLY = 0.88,
    C_NOISE_SIGMAS = 3.0, C_FLAG = 0.55;
var TWO_PI = 2 * Math.PI;
function pmod(a, b) { var r = a % b; return r < 0 ? r + b : r; }       // Python / numpy float modulo
function rint(x) { return O.cv.rint(x); }

var _cache = {};

// omr/recognize.py:_disc  (dy, dx) offsets of a disc, row-major (y outer, x inner) like np.mgrid + boolean mask
function disc(radiusPx) {
    var key = 'd' + (Math.round(radiusPx * 100) / 100).toFixed(2);       // round(radius_px, 2)
    var c = _cache[key];
    if (!c) {
        var r = Math.ceil(radiusPx), dy = [], dx = [], r2 = radiusPx * radiusPx;
        for (var y = -r; y <= r; y++) for (var x = -r; x <= r; x++) if (x * x + y * y <= r2) { dy.push(y); dx.push(x); }
        c = _cache[key] = { dy: Int32Array.from(dy), dx: Int32Array.from(dx), n: dy.length };
    }
    return c;
}

function r3(x) { return (Math.round(x * 1000) / 1000).toFixed(3); }

// omr/illum.py:100
function annulus(Rmm, ppm) {
    var key = 'a' + r3(Rmm) + '_' + r3(ppm), c = _cache[key];
    if (!c) {
        var rOut = (Rmm + ANN_OUT_MM) * ppm, h = Math.ceil(rOut) + 1, ys = [], xs = [], sec = [];
        for (var y = -h; y <= h; y++) for (var x = -h; x <= h; x++) {
            var rr = Math.hypot(x, y) / ppm;
            if (rr >= Rmm + ANN_IN_MM && rr <= Rmm + ANN_OUT_MM) {
                var ang = pmod(Math.atan2(y, x), TWO_PI);
                ys.push(y); xs.push(x); sec.push(Math.floor(ang / (TWO_PI / N_SECT)) % N_SECT);
            }
        }
        var groups = [];
        for (var s = 0; s < N_SECT; s++) { var g = []; for (var i = 0; i < sec.length; i++) if (sec[i] === s) g.push(i); groups.push(g); }
        c = _cache[key] = { y: Int32Array.from(ys), x: Int32Array.from(xs), groups: groups };
    }
    return c;
}

// omr/illum.py:118  offsets of the printed ring line
function ring(Rmm, ppm) {
    var key = 'r' + r3(Rmm) + '_' + r3(ppm), c = _cache[key];
    if (!c) {
        var h = Math.ceil((Rmm + 0.6) * ppm), ys = [], xs = [];
        for (var y = -h; y <= h; y++) for (var x = -h; x <= h; x++) if (Math.abs(Math.hypot(x, y) / ppm - Rmm) <= 0.22) { ys.push(y); xs.push(x); }
        c = _cache[key] = { y: Int32Array.from(ys), x: Int32Array.from(xs) };
    }
    return c;
}

// omr/illum.py:128  W[k][p] (K x m, float32) weights of sector sample k over disc pixel p
function sectorGeometry(Rmm, dy, dx, ppm) {
    var key = 'W' + r3(Rmm) + '_' + r3(ppm) + '_' + dy.length, c = _cache[key];
    if (!c) {
        var m = dy.length, W = new Float32Array(N_SECT * m), rs = Rmm + 0.5 * (ANN_IN_MM + ANN_OUT_MM);
        for (var k = 0; k < N_SECT; k++) {
            var th = (k + 0.5) * TWO_PI / N_SECT, sx = rs * Math.cos(th), sy = rs * Math.sin(th);
            for (var p = 0; p < m; p++) {
                var qx = dx[p] / ppm, qy = dy[p] / ppm, d2 = (qx - sx) * (qx - sx) + (qy - sy) * (qy - sy);
                W[k * m + p] = Math.exp(-d2 / (2 * SIGMA_MM * SIGMA_MM));
            }
        }
        c = _cache[key] = W;
    }
    return c;
}

// omr/illum.py:139  img: uint8 {w,h,data}; px, py: Int32Array centres; -> Float32Array(n * dy.length), border replicated
function gather(img, px, py, dy, dx) {
    var n = px.length, m = dy.length, W = img.w, H = img.h, d = img.data, out = new Float32Array(n * m), i, j;
    for (i = 0; i < n; i++) {
        var cx = px[i], cy = py[i], o = i * m;
        for (j = 0; j < m; j++) {
            var y = cy + dy[j], x = cx + dx[j];
            y = y < 0 ? 0 : y >= H ? H - 1 : y; x = x < 0 ? 0 : x >= W ? W - 1 : x;
            out[o + j] = d[y * W + x];
        }
    }
    return out;
}

function centrePx(centers, ppm) {            // np.round(c * ppm).astype(int)
    var n = centers.length, px = new Int32Array(n), py = new Int32Array(n);
    for (var i = 0; i < n; i++) { px[i] = rint(centers[i][0] * ppm); py[i] = rint(centers[i][1] * ppm); }
    return { px: px, py: py };
}

// omr/illum.py:156  -> Float32Array(n * N_SECT): gray level of blank paper in the annulus around every bubble, outliers repaired
function paperSamples(img, centers, diameterMm, ppm) {
    var c = centrePx(centers, ppm), n = centers.length, a = annulus(diameterMm / 2, ppm);
    var pix = gather(img, c.px, c.py, a.y, a.x), M = a.y.length, v = new Float32Array(n * N_SECT), k, i, j;
    for (k = 0; k < N_SECT; k++) {
        var g = a.groups[k], gl = g.length, sub = new Float32Array(n * gl);
        for (i = 0; i < n; i++) for (j = 0; j < gl; j++) sub[i * gl + j] = pix[i * M + g[j]];
        var p = pctRows(sub, n, gl, SECT_PCT);
        for (i = 0; i < n; i++) v[i * N_SECT + k] = p[i];
    }
    var out = new Float32Array(n * N_SECT);
    var tol = f32(OUTLIER_TOL), tol2 = f32(0.5 * OUTLIER_TOL);
    for (i = 0; i < n; i++) for (k = 0; k < N_SECT; k++) {
        var vv = v[i * N_SECT + k], prv = v[i * N_SECT + (k + N_SECT - 1) % N_SECT], nxt = v[i * N_SECT + (k + 1) % N_SECT];
        var mm = f32(f32(0.5) * f32(prv + nxt)), mx = Math.max(mm, 1.0);
        var odd = Math.abs(f32(vv - mm)) > f32(tol * mx) && Math.abs(f32(prv - nxt)) < f32(tol2 * mx);
        out[i * N_SECT + k] = odd ? mm : vv;
    }
    return out;
}

var _planeBasis = {};
function planeBasis(Rmm) {                   // omr/illum.py:184: A (K x 3) rows [1, sx, sy]
    var key = r3(Rmm), c = _planeBasis[key];
    if (!c) {
        var rs = Rmm + 0.5 * (ANN_IN_MM + ANN_OUT_MM), A = [];
        for (var k = 0; k < N_SECT; k++) { var th = (k + 0.5) * TWO_PI / N_SECT; A.push([1, rs * Math.cos(th), rs * Math.sin(th)]); }
        c = _planeBasis[key] = A;
    }
    return c;
}

function solve3(M, b) {                      // 3x3 Gaussian elimination with partial pivoting
    var a = [M.slice(0, 3), M.slice(3, 6), M.slice(6, 9)], x = [b[0], b[1], b[2]], c, r, k;
    for (c = 0; c < 3; c++) {
        var p = c;
        for (r = c + 1; r < 3; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
        if (p !== c) { var t = a[c]; a[c] = a[p]; a[p] = t; var tb = x[c]; x[c] = x[p]; x[p] = tb; }
        for (r = c + 1; r < 3; r++) {
            var f = a[r][c] / a[c][c];
            for (k = c; k < 3; k++) a[r][k] -= f * a[c][k];
            x[r] -= f * x[c];
        }
    }
    for (r = 2; r >= 0; r--) { var s = x[r]; for (k = r + 1; k < 3; k++) s -= a[r][k] * x[k]; x[r] = s / a[r][r]; }
    return x;
}
function wplane(v, w, A, i, coef) {          // omr/illum.py:196 weighted LS plane through the samples of bubble i -> coef[3i..3i+2]
    var M = new Array(9).fill(0), b = [0, 0, 0], k, a, c;
    for (k = 0; k < N_SECT; k++) {
        var wk = w[i * N_SECT + k], vk = v[i * N_SECT + k];
        for (a = 0; a < 3; a++) { b[a] += wk * A[k][a] * vk; for (c = 0; c < 3; c++) M[a * 3 + c] += wk * A[k][a] * A[k][c]; }
    }
    M[0] += 1e-6; M[4] += 1e-6; M[8] += 1e-6;
    var s = solve3(M, b);
    coef[3 * i] = s[0]; coef[3 * i + 1] = s[1]; coef[3 * i + 2] = s[2];
}

// omr/illum.py:204.  v: Float64Array(n*8), W: Float32Array(8*m), explained: Uint8Array(n*8) -> {P: Float64Array(n*m), consistent: Uint8Array(n)}
function paperModel(v, n, Rmm, dy, dx, ppm, W, explained) {
    var m = dy.length, A = planeBasis(Rmm), i, k, p, j;
    var scale = new Float64Array(n), w = new Float64Array(n * N_SECT), coef = new Float64Array(3 * n);
    for (i = 0; i < n; i++) {
        var s = 0; for (k = 0; k < N_SECT; k++) s += v[i * N_SECT + k];
        scale[i] = Math.max(s / N_SECT, 1.0);
        var ws = 0;
        for (k = 0; k < N_SECT; k++) { w[i * N_SECT + k] = explained[i * N_SECT + k] ? 0 : 1; ws += w[i * N_SECT + k]; }
        if (ws < 4) for (k = 0; k < N_SECT; k++) w[i * N_SECT + k] = 1;
        wplane(v, w, A, i, coef);
    }
    var resid = new Float64Array(N_SECT);
    for (var it = 0; it < 2; it++) {
        var anyBad = false;
        for (i = 0; i < n; i++) {
            var mx = 0, worst = 0;
            for (k = 0; k < N_SECT; k++) {
                var pr = coef[3 * i] + coef[3 * i + 1] * A[k][1] + coef[3 * i + 2] * A[k][2];
                resid[k] = Math.abs(v[i * N_SECT + k] - pr) * w[i * N_SECT + k];
                if (resid[k] > mx) { mx = resid[k]; worst = k; }
            }
            if (mx > PLANE_TOL * scale[i]) { w[i * N_SECT + worst] = 0; wplane(v, w, A, i, coef); anyBad = true; }
        }
        if (!anyBad) break;
    }
    var P = new Float64Array(n * m), consistent = new Uint8Array(n).fill(1), stepIdx = [];
    for (i = 0; i < n; i++) {
        var mxr = 0, adj = false, out = new Uint8Array(N_SECT), signed = new Float64Array(N_SECT);
        for (k = 0; k < N_SECT; k++) {
            signed[k] = v[i * N_SECT + k] - (coef[3 * i] + coef[3 * i + 1] * A[k][1] + coef[3 * i + 2] * A[k][2]);
            var r = Math.abs(signed[k]) * w[i * N_SECT + k];
            if (r > mxr) mxr = r;
            out[k] = (w[i * N_SECT + k] === 0 && !explained[i * N_SECT + k]) ? 1 : 0;
        }
        // an adjacent pair of ignored samples is what a shadow edge cutting the annulus looks like - unless all ignored are DARKER than the fit
        var anyAdj = false, allDark = true;
        for (k = 0; k < N_SECT; k++) {
            if (out[k] && out[(k + N_SECT - 1) % N_SECT]) anyAdj = true;
            if (out[k] && !(signed[k] < 0)) allDark = false;
        }
        adj = anyAdj && !allDark;
        var planeOk = mxr <= PLANE_TOL * scale[i] && !adj;
        if (planeOk) {
            for (j = 0; j < m; j++) P[i * m + j] = coef[3 * i] + coef[3 * i + 1] * (dx[j] / ppm) + coef[3 * i + 2] * (dy[j] / ppm);
        } else stepIdx.push(i);
    }
    var C = new Float64Array(N_SECT * N_SECT), Sx = new Float64Array(N_SECT);
    for (var si = 0; si < stepIdx.length; si++) {
        i = stepIdx[si];
        for (k = 0; k < N_SECT; k++) {
            var refk = Math.max(v[i * N_SECT + k], 1.0);
            for (j = 0; j < N_SECT; j++) C[k * N_SECT + j] = Math.abs(v[i * N_SECT + j] - v[i * N_SECT + k]) <= CLUSTER_TOL * refk ? 1 : 0;
        }
        var pmax = -Infinity, pmin = Infinity;
        for (p = 0; p < m; p++) {
            var kbest = 0, sbest = -1;
            for (k = 0; k < N_SECT; k++) {
                var sk = 0;
                for (j = 0; j < N_SECT; j++) sk += C[k * N_SECT + j] * W[j * m + p];
                sk = f32(sk);
                if (sk > sbest) { sbest = sk; kbest = k; }
            }
            var num = 0, den = 0;
            for (j = 0; j < N_SECT; j++) { var wm = C[kbest * N_SECT + j] * W[j * m + p]; num += wm * v[i * N_SECT + j]; den += wm; }
            var pc = num / Math.max(den, 1e-6);
            P[i * m + p] = pc;
            if (pc > pmax) pmax = pc; if (pc < pmin) pmin = pc;
        }
        consistent[i] = (pmax - pmin) <= EDGE_TOL * Math.max(pmax, 1.0) ? 1 : 0;
    }
    for (i = 0; i < n * m; i++) if (P[i] < 1.0) P[i] = 1.0;
    return { P: P, consistent: consistent };
}

// omr/illum.py:259  -> Uint8Array(n * 8)
function explainedByNeighbours(centers, hot, v, ref) {
    var n = centers.length, out = new Uint8Array(n * N_SECT), hi = [], i, h, k;
    for (i = 0; i < n; i++) if (hot[i]) hi.push(i);
    if (!hi.length) return out;
    var anyhit = new Uint8Array(n * N_SECT);
    for (i = 0; i < n; i++) {
        for (var q = 0; q < hi.length; q++) {
            h = hi[q];
            var dxm = centers[h][0] - centers[i][0], dym = centers[h][1] - centers[i][1], dist = Math.hypot(dxm, dym);
            if (!(dist > 1e-6 && dist < NEIGHBOUR_MM)) continue;
            var ang = pmod(Math.atan2(dym, dxm), TWO_PI), half = Math.asin(Math.min(Math.max(3.6 / Math.max(dist, 3.7), 0), 1));
            for (k = 0; k < N_SECT; k++) {
                var centre = (k + 0.5) * TWO_PI / N_SECT, dang = Math.abs(pmod(ang - centre + Math.PI, TWO_PI) - Math.PI);
                if (dang <= half + Math.PI / N_SECT) anyhit[i * N_SECT + k] = 1;
            }
        }
    }
    for (i = 0; i < n; i++) for (k = 0; k < N_SECT; k++) out[i * N_SECT + k] = anyhit[i * N_SECT + k] && v[i * N_SECT + k] < 0.93 * ref[i] ? 1 : 0;
    return out;
}

function medianOf(a) {                       // np.median of a small array (float64 arithmetic)
    var s = Float64Array.from(a).sort(), n = s.length;
    return n & 1 ? s[n >> 1] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

// omr/illum.py:289  -> Float64Array(n) factor that undoes a mark's own blur halo in the paper samples of an inked bubble.
// level: Float32Array(n) (median of the float32 samples)
function haloLift(centers, hot, level) {
    var n = centers.length, lift = new Float64Array(n).fill(1), hi = [], blank = [], i, q, b;
    for (i = 0; i < n; i++) (hot[i] ? hi : blank).push(i);
    if (!hi.length || !blank.length) return lift;
    var kk = Math.min(HALO_NEIGHBOURS, blank.length);
    for (q = 0; q < hi.length; q++) {
        var h = hi[q], ds = [];
        for (b = 0; b < blank.length; b++) {
            var d = Math.hypot(centers[h][0] - centers[blank[b]][0], centers[h][1] - centers[blank[b]][1]);
            ds.push([d <= HALO_REACH_MM ? d : Infinity, blank[b]]);
        }
        ds.sort(function (x, y) { return x[0] - y[0]; });
        var vals = [];
        for (b = 0; b < kk; b++) if (isFinite(ds[b][0])) vals.push(level[ds[b][1]]);
        var f = 1.0;
        if (vals.length) {
            vals.sort(function (x, y) { return x - y; });
            var nb = vals.length & 1 ? vals[vals.length >> 1] : f32(f32(vals[vals.length / 2 - 1] + vals[vals.length / 2]) * 0.5);
            f = f32(nb / Math.max(level[h], 1.0));
        }
        lift[h] = Math.min(Math.max(f, 1.0), f32(HALO_MAX));
    }
    return lift;
}

// omr/illum.py:316  -> {c: Float32Array(n), ref}
function contrastField(img, centers, diameterMm, ppm, paper) {
    var n = centers.length, cp = centrePx(centers, ppm), rg = ring(diameterMm / 2, ppm), i, j;
    var pix = gather(img, cp.px, cp.py, rg.y, rg.x), m = rg.y.length;
    var p25 = pctRows(pix, n, m, 25), dark = new Float64Array(n);
    for (i = 0; i < n; i++) dark[i] = 1.0 - p25[i] / Math.max(paper[i], 1.0);
    var ref = cvx.percentile(dark, C_REF_PCT);
    if (ref < 0.03) return { c: new Float32Array(n).fill(1), ref: ref };
    var rel = new Float64Array(n);
    for (i = 0; i < n; i++) rel[i] = dark[i] / ref;
    var k = Math.min(C_NEIGHBOURS, n), out = new Float32Array(n);
    if (n <= k) return { c: out.fill(medianOf(rel)), ref: ref };
    // the k nearest bubbles of every bubble (ties by index, as the stable sort of the Python): insertion into a sorted k-list
    var cx = new Float64Array(n), cy = new Float64Array(n), tmp = new Float64Array(k), bd = new Float64Array(k), bi = new Int32Array(k);
    for (i = 0; i < n; i++) { cx[i] = centers[i][0]; cy[i] = centers[i][1]; }
    for (i = 0; i < n; i++) {
        var cnt = 0, xi = cx[i], yi = cy[i];
        for (j = 0; j < n; j++) {
            var dx = xi - cx[j], dy = yi - cy[j], dd = dx * dx + dy * dy;
            if (cnt === k && dd >= bd[k - 1]) continue;       // later index loses ties
            var pos = cnt < k ? cnt++ : k - 1;
            while (pos > 0 && bd[pos - 1] > dd) { bd[pos] = bd[pos - 1]; bi[pos] = bi[pos - 1]; pos--; }
            bd[pos] = dd; bi[pos] = j;
        }
        for (j = 0; j < k; j++) tmp[j] = rel[bi[j]];
        out[i] = medianOf(tmp);
    }
    return { c: out, ref: ref };
}

// omr/illum.py:392  smooth: uint8 grey; centers: [[x, y] mm]; -> {D: Float32Array(n * npix), n, m, reliable: Uint8Array(n)}
function bubbleDarkness(smooth, centers, diameterMm, ppm) {
    var n = centers.length, cp = centrePx(centers, ppm), dsc = disc(INNER_DISC_FRAC * diameterMm / 2 * ppm), dy = dsc.dy, dx = dsc.dx, m = dsc.n;
    var pix = gather(smooth, cp.px, cp.py, dy, dx), i, k, j;
    var v32 = paperSamples(smooth, centers, diameterMm, ppm);
    var W = sectorGeometry(diameterMm / 2, dy, dx, ppm);
    var ref32 = new Float32Array(n), pmean = new Float32Array(n), hot = new Uint8Array(n);
    for (i = 0; i < n; i++) {
        var row = Array.prototype.slice.call(v32, i * N_SECT, (i + 1) * N_SECT).sort(function (a, b) { return a - b; });
        ref32[i] = f32(f32(row[3] + row[4]) * 0.5);
        var s = 0; for (j = 0; j < m; j++) s += pix[i * m + j];
        pmean[i] = s / m;
        hot[i] = f32(1.0 - f32(pmean[i] / Math.max(ref32[i], 1.0))) > f32(HOT_DARKNESS) ? 1 : 0;
    }
    var lift = haloLift(centers, hot, ref32), v = new Float64Array(n * N_SECT), ref = new Float64Array(n), tmp = new Float64Array(N_SECT);
    for (i = 0; i < n; i++) {
        for (k = 0; k < N_SECT; k++) { v[i * N_SECT + k] = v32[i * N_SECT + k] * lift[i]; tmp[k] = v[i * N_SECT + k]; }
        ref[i] = medianOf(tmp);
    }
    var expl = explainedByNeighbours(centers, hot, v, ref);
    for (i = 0; i < n; i++) {
        var cnt = 0;
        for (k = 0; k < N_SECT; k++) {
            if (hot[i] && v[i * N_SECT + k] < 0.93 * ref[i]) expl[i * N_SECT + k] = 1;
            cnt += expl[i * N_SECT + k];
        }
        if (cnt > 3) for (k = 0; k < N_SECT; k++) expl[i * N_SECT + k] = 0;
    }
    var pm = paperModel(v, n, diameterMm / 2, dy, dx, ppm, W, expl);
    var reliable = new Uint8Array(n), P = pm.P;
    for (i = 0; i < n; i++) {
        var vmax = 0; for (k = 0; k < N_SECT; k++) vmax = Math.max(vmax, v[i * N_SECT + k]);
        reliable[i] = pm.consistent[i] && vmax >= MIN_PAPER ? 1 : 0;
    }
    var med = new Float64Array(n);
    for (i = 0; i < n; i++) { for (k = 0; k < N_SECT; k++) tmp[k] = v[i * N_SECT + k]; med[i] = medianOf(tmp); }
    var cf = contrastField(smooth, centers, diameterMm, ppm, med), cLoc = cf.c;
    // The correction is one-sided (it only raises darkness), so it is applied only to what stands out from the page's own scatter
    var cmed = cvx.percentile(cLoc, 50, true), dev = new Float32Array(n);
    for (i = 0; i < n; i++) dev[i] = Math.abs(f32(cLoc[i] - cmed));
    var sigma = 1.4826 * cvx.percentile(dev, 50, true);
    var cApply = Math.min(C_APPLY, cmed - C_NOISE_SIGMAS * sigma), cApply32 = f32(cApply), cFlag32 = f32(C_FLAG);
    var D = new Float32Array(n * m);
    for (i = 0; i < n; i++) {
        var gain = 1.0;
        if (cLoc[i] < cApply32) gain = f32(1.0 / Math.min(Math.max(cLoc[i], cFlag32), 1.0));
        if (!(cLoc[i] >= cFlag32)) reliable[i] = 0;
        if (!reliable[i]) continue;
        for (j = 0; j < m; j++) D[i * m + j] = (1.0 - pix[i * m + j] / (P[i * m + j] * PAPER_GAIN)) * gain;
    }
    return { D: D, n: n, m: m, reliable: reliable };
}

O.illum = { disc: disc, paperSamples: paperSamples, paperModel: paperModel, contrastField: contrastField, haloLift: haloLift, bubbleDarkness: bubbleDarkness,
    INNER_DISC_FRAC: INNER_DISC_FRAC };
})(globalThis.OMR = globalThis.OMR || {});
