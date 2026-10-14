(function (O) {
'use strict';
// detect.js - omr/detect.py: corner-marker detection that survives print defects (see the docstring there for the reasoning).
// Opened-ladder passes, AprilTag-style re-decoding of rejected candidates, lighting-flattened probe, missing-corner window search.
// Corners are flat Float64Array(8) [x0, y0, ..., x3, y3] (TL, TR, BR, BL); found = Map id -> corners (insertion order = Python dict order).
var f32 = Math.fround;
var cv = O.cv, cvx = O.cvx, aruco = O.aruco, lay = O.layout, mk = O.markers;
var rint = cv.rint;
var trace = null;     // debug hook: function (tag, img, result)
// Pass policy (J3 optimisation, measured on 553 frames: video 424, synthetic 120, photos 9 - see web/test/pipe/STATUS.md).  Defaults are the
// optimised policy; setPolicy({flatOpen: true, hintOnly: false, candScale: 0}) gives omr/detect.py's exact pass sequence (parity runs).
//   flatOpen   : the opened pass on the lighting-flattened copy (last resort).  Never added a marker on any recorded frame -> off.
//   hintOnly   : with a decoded marker only ITS scale is opened (python also tried the two nearest ladder scales).
//   candScale  : no marker decoded but marker-like rejected candidates exist: open at their scale instead of walking the ladder (n steps).
//   coarse     : frames with a long side > COARSE_MIN are first searched at half size (one raw ArUco pass, 4x fewer pixels); when that already gives a
//                solid set (>= 3 markers of one page, none smaller than COARSE_MIN_SIDE px at half size) the corners are mapped back and sub-pixel
//                refined on the full frame (the path photos already take) and only the missing-corner window search runs at full size.  Otherwise
//                the normal full-size search runs from scratch.
var policy = { flatOpen: false, hintOnly: true, candScale: 1, coarse: false };
var COARSE_MIN = 1100, COARSE_MIN_SIDE = 24;
function detectIn(img, tag) { var r = aruco.detect(img); if (trace) trace(tag, img, r); return r; }

var MODULE_MM = lay.MARKER_MM / 6.0, MAX_SHEET_PPM_OF_FRAME = 1.1 / 210.0, MIN_PPM = 2.0, KERNEL_OF_MODULE = 0.85, LADDER_RATIO = 1.3,
    DECODE_MIN_CONTRAST = 0.25, RING_MODULES = 0.6, MIN_FOR_STOP = 3, SAME_PLACE = 0.35;
var COMPLETE_ROI = 1.3, COMPLETE_MAX_OFF = 0.9, COMPLETE_WIDE = [2.6, 2.2], COMPLETE_KERNELS = [0.85, 0.55, 0.35], COMPLETE_SIDE = [0.6, 1.6];
var FLAT_BG_FRAC = 1 / 10, FLAT_SCALE = 8, FLAT_DENOISE_SIGMA = 1.2;
var DETECT_LONG_SIDE = 1920;

// omr/detect.py:67
function ladder(h, w) {
    var short = Math.min(h, w), hi = MODULE_MM * MAX_SHEET_PPM_OF_FRAME * short, lo = MODULE_MM * MIN_PPM, out = [hi];
    while (out[out.length - 1] / LADDER_RATIO >= lo * 1.1) out.push(out[out.length - 1] / LADDER_RATIO);
    return out;
}
function kernelSize(modulePx, frac) { return Math.max(3, rint(frac * modulePx) | 1); }      // odd, at least 3
function side(c) {                                                                       // mean edge length of a quad (flat 8)
    var s = 0;
    for (var i = 0; i < 4; i++) { var j = (i + 1) & 3; s += Math.hypot(c[2 * j] - c[2 * i], c[2 * j + 1] - c[2 * i + 1]); }
    return s / 4;
}
function centre(c) { return [(c[0] + c[2] + c[4] + c[6]) / 4, (c[1] + c[3] + c[5] + c[7]) / 4]; }

// ---------------------------------------------------------------- merge by id
function Merge() { this.found = new Map(); this.conflict = new Set(); this.dup = false; }
Merge.prototype.add = function (corners, ids) {
    if (!ids) return;
    var seen = new Set();
    for (var n = 0; n < ids.length; n++) {
        var i = ids[n], c = corners[n];
        if (i >= 4 * mk.MAX_PAGES) continue;
        if (this.conflict.has(i)) continue;
        if (this.found.has(i)) {
            var old = this.found.get(i), co = centre(old), cn = centre(c);
            if (Math.hypot(co[0] - cn[0], co[1] - cn[1]) > SAME_PLACE * Math.max(side(old), side(c))) {
                this.dup = this.dup || seen.has(i);
                this.conflict.add(i);
                this.found.delete(i);
            }
            seen.add(i);
            continue;
        }
        this.found.set(i, Float64Array.from(c));
        seen.add(i);
    }
};
Merge.prototype.solid = function () {
    var groups = {}, best = 0;
    this.found.forEach(function (c, i) { var g = Math.floor(i / 4); groups[g] = (groups[g] || 0) + 1; if (groups[g] > best) best = groups[g]; });
    return best >= MIN_FOR_STOP;
};
Merge.prototype.modulePx = function () {
    if (!this.found.size) return null;
    var s = []; this.found.forEach(function (c) { s.push(side(c)); });
    return cvx.median(s) / 6.0;
};
function groupsOf(m) {
    var groups = {};
    m.found.forEach(function (c, i) { var g = Math.floor(i / 4); (groups[g] = groups[g] || []).push(i); });
    return groups;
}

// ---------------------------------------------------------------- AprilTag-style re-decoding of rejected candidates
var CELL_UV = [], WHITE_UV = [], BORDER = [], GRID_UV = [];
(function () {
    var G = [-0.25, 0, 0.25], r, c, i, j, t;
    for (r = 0; r < 6; r++) for (c = 0; c < 6; c++) for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) CELL_UV.push(c + 0.5 + G[j], r + 0.5 + G[i]);
    var R0 = -RING_MODULES, R1 = 6 + RING_MODULES;
    for (t = 0; t < 6; t++) WHITE_UV.push([t + 0.5, R0]);
    for (t = 0; t < 6; t++) WHITE_UV.push([t + 0.5, R1]);
    for (t = 0; t < 6; t++) WHITE_UV.push([R0, t + 0.5]);
    for (t = 0; t < 6; t++) WHITE_UV.push([R1, t + 0.5]);
    for (r = 0; r < 6; r++) for (c = 0; c < 6; c++) BORDER.push(r === 0 || r === 5 || c === 0 || c === 5);
})();
var UNIT = [0, 0, 6, 0, 6, 6, 0, 6];
// fixed least-squares pseudo-inverses: model (1, u, v) fitted to the 20 border cells / the 24 ring samples
function pinvOf(A) {
    var n = A.length, M = new Float64Array(9), i, j, k;
    for (i = 0; i < n; i++) for (j = 0; j < 3; j++) for (k = 0; k < 3; k++) M[j * 3 + k] += A[i][j] * A[i][k];
    var inv = new Float64Array(9), cols = [];
    for (j = 0; j < 3; j++) {
        var e = [0, 0, 0]; e[j] = 1;
        cols.push(cv.solveLinear(Float64Array.from(M), e, 3));
    }
    for (j = 0; j < 3; j++) for (k = 0; k < 3; k++) inv[j * 3 + k] = cols[k][j];
    var P = []; // 3 x n
    for (j = 0; j < 3; j++) { var row = new Float64Array(n); for (i = 0; i < n; i++) for (k = 0; k < 3; k++) row[i] += inv[j * 3 + k] * A[i][k]; P.push(row); }
    return P;
}
var A_BORDER = [], A_WHITE = [], A_GRID = [], PINV_BORDER, PINV_WHITE, BORDER_IDX = [];
(function () {
    for (var r = 0; r < 6; r++) for (var c = 0; c < 6; c++) {
        A_GRID.push([1, c + 0.5, r + 0.5]);
        if (BORDER[r * 6 + c]) { A_BORDER.push([1, c + 0.5, r + 0.5]); BORDER_IDX.push(r * 6 + c); }
    }
    WHITE_UV.forEach(function (uv) { A_WHITE.push([1, uv[0], uv[1]]); });
    PINV_BORDER = pinvOf(A_BORDER); PINV_WHITE = pinvOf(A_WHITE);
})();
function applyPinv(P, y) { var o = [0, 0, 0]; for (var j = 0; j < 3; j++) { var s = 0; for (var i = 0; i < y.length; i++) s += P[j][i] * y[i]; o[j] = s; } return o; }

function sample(img, H, uv) {                         // omr/detect.py:_sample  uv: flat (u, v) pairs -> Float64Array of grey levels
    var p = cv.perspectiveTransform(uv, H), n = uv.length >> 1, xs = new Float32Array(n), ys = new Float32Array(n);
    for (var i = 0; i < n; i++) { xs[i] = p[2 * i]; ys[i] = p[2 * i + 1]; }
    return cv.remapLinear(img, xs, ys, 'replicate');
}
var WHITE_FLAT = (function () { var a = []; WHITE_UV.forEach(function (u) { a.push(u[0], u[1]); }); return a; })();

// omr/detect.py:_model_fit -> {q (flat 8 clockwise), rel (36), ringRel (24)} or null
function modelFit(img, quad) {
    var q = Float64Array.from(quad);
    var ax = q[2] - q[0], ay = q[3] - q[1], bx = q[4] - q[2], by = q[5] - q[3];
    if (ax * by - ay * bx < 0) q = Float64Array.of(q[6], q[7], q[4], q[5], q[2], q[3], q[0], q[1]);
    var qf = new Float64Array(8);
    for (var i = 0; i < 8; i++) qf[i] = f32(q[i]);
    var H = cv.getPerspectiveTransform(UNIT, qf);
    var s = sample(img, H, CELL_UV), cells = new Float64Array(36), k, j;
    for (k = 0; k < 36; k++) { var m = 0; for (j = 0; j < 9; j++) m += s[k * 9 + j]; cells[k] = m / 9; }
    var ring = sample(img, H, WHITE_FLAT);
    var bc = applyPinv(PINV_BORDER, BORDER_IDX.map(function (ix) { return cells[ix]; })), wc = applyPinv(PINV_WHITE, ring);
    var blk = new Float64Array(36), wht = new Float64Array(36), conMin = Infinity, conSum = 0, whtSum = 0;
    for (k = 0; k < 36; k++) {
        blk[k] = A_GRID[k][0] * bc[0] + A_GRID[k][1] * bc[1] + A_GRID[k][2] * bc[2];
        wht[k] = A_GRID[k][0] * wc[0] + A_GRID[k][1] * wc[1] + A_GRID[k][2] * wc[2];
        var con = wht[k] - blk[k];
        if (con < conMin) conMin = con;
        conSum += con; whtSum += wht[k];
    }
    if (conMin <= 0 || conSum / 36 < DECODE_MIN_CONTRAST * (whtSum / 36)) return null;
    var rel = new Float64Array(36);
    for (k = 0; k < 36; k++) {
        rel[k] = (cells[k] - blk[k]) / (wht[k] - blk[k]);
        if (BORDER[k] && rel[k] > 0.5) return null;
    }
    var ringRel = new Float64Array(24);
    for (k = 0; k < 24; k++) {
        var a = A_WHITE[k], pb = a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2], pw = a[0] * wc[0] + a[1] * wc[1] + a[2] * wc[2];
        ringRel[k] = (ring[k] - pb) / (pw - pb);
        if (ringRel[k] < 0.5) return null;
    }
    return { q: q, rel: rel, ringRel: ringRel };
}

// omr/detect.py:model_decode -> {id, corners} or null
function modelDecode(img, quad) {
    var fit = modelFit(img, quad);
    if (!fit) return null;
    var bits = new Uint8Array(16), r, c;
    for (r = 0; r < 4; r++) for (c = 0; c < 4; c++) bits[r * 4 + c] = fit.rel[(r + 1) * 6 + c + 1] > 0.5 ? 1 : 0;
    var id = aruco.identify(bits, 0.0);
    if (!id) return null;
    var q = fit.q, out = new Float64Array(8);
    for (var k = 0; k < 4; k++) { var s = (k - id.rotation + 4) & 3; out[2 * k] = q[2 * s]; out[2 * k + 1] = q[2 * s + 1]; }   // np.roll(q, rot, axis=0)
    return { id: id.id, corners: out };
}

function markerlike(gray, rejected, minSide) {
    var n = 0;
    for (var i = 0; i < rejected.length; i++) if (side(rejected[i]) >= minSide && modelFit(gray, rejected[i]) !== null) n++;
    return n;
}

function refine(gray, q, win) {              // cornerSubPix(gray, q (float32), (win, win), (-1, -1), (30, 0.1)) -> flat Float64Array(8)
    var qf = new Float64Array(8);
    for (var i = 0; i < 8; i++) qf[i] = f32(q[i]);
    return cv.cornerSubPix(gray, qf, win, 30, 0.1);
}
function redecode(m, gray, imgs, rejected, minSide) {
    for (var n = 0; n < rejected.length; n++) {
        var r = rejected[n];
        if (side(r) < minSide) continue;
        for (var k = 0; k < imgs.length; k++) {
            var out = modelDecode(imgs[k], r);
            if (out === null) continue;
            if (m.found.has(out.id) || out.id >= 4 * mk.MAX_PAGES) break;
            var win = Math.max(2, rint(0.3 * side(out.corners) / 6));          // 0.3 module, like ArUco's relative window
            m.add([refine(gray, out.corners, win)], [out.id]);
            break;
        }
    }
}

// omr/detect.py:flatten_lighting
function flattenLighting(gray) {
    var h = gray.h, w = gray.w;
    var small = cv.resize(gray, Math.max(Math.floor(w / FLAT_SCALE), 1), Math.max(Math.floor(h / FLAT_SCALE), 1), 'area');
    var k = Math.max(5, Math.floor(Math.min(small.w, small.h) * FLAT_BG_FRAC) | 1);
    var bg = cv.gaussianBlur(cv.dilate(small, k), k / 3.0);
    bg = cv.resize(bg, w, h, 'linear');
    return cv.flattenDivideBlur(gray, bg, 235.0, FLAT_DENOISE_SIGMA);
}

// omr/detect.py:_blob_quads  -> array of flat-8 quads
function blobQuads(img, predSide, predC, maxOff) {
    var b = cv.threshold(img, 'otsu', 255, 'binary_inv').bin, cs = cv.findContoursExternal(b), out = [];
    for (var n = 0; n < cs.length; n++) {
        var c = cs[n];
        if (cv.contourArea(c) < Math.pow(COMPLETE_SIDE[0] * predSide, 2) * 0.5) continue;
        var ap = cv.approxPolyDP(c, 0.04 * cv.arcLength(c, true), true);
        if (ap.length !== 8 || !cv.isContourConvex(ap)) continue;
        var q = Float64Array.from(ap), cc = centre(q);
        if (COMPLETE_SIDE[0] * predSide <= side(q) && side(q) <= COMPLETE_SIDE[1] * predSide &&
            Math.hypot(cc[0] - predC[0], cc[1] - predC[1]) < maxOff * predSide) out.push(q);
    }
    return out;
}

function markerRectCorners(id) {             // marker's layout quad (mm) TL, TR, BR, BL for marker id
    var page = Math.floor(id / 4), corner = mk.CORNERS[id % 4], r = lay.markerRects(page)[corner], x = r[0], y = r[1], w = r[2], h = r[3];
    return [x, y, x + w, y, x + w, y + h, x, y + h];
}
function pageQuads(page) {
    var out = {};
    for (var c = 0; c < 4; c++) out[mk.markerId(page, mk.CORNERS[c])] = markerRectCorners(4 * page + c);
    return out;
}

// omr/detect.py:_search_window -> {q, runs}
function searchWindow(gray, miss, sd, ctr, roiF, maxOff) {
    var h = gray.h, w = gray.w;
    var x0 = Math.max(Math.trunc(ctr[0] - roiF * sd), 0), y0 = Math.max(Math.trunc(ctr[1] - roiF * sd), 0);
    var x1 = Math.min(Math.trunc(ctr[0] + roiF * sd), w), y1 = Math.min(Math.trunc(ctr[1] + roiF * sd), h);
    if (x1 - x0 < sd * 0.5 || y1 - y0 < sd * 0.5) return { q: null, runs: 0 };            // the predicted marker lies outside the frame
    var rw = x1 - x0, roi = cvx.mk(rw, y1 - y0);
    for (var y = y0; y < y1; y++) roi.data.set(gray.data.subarray(y * w + x0, y * w + x1), (y - y0) * rw);
    var local = new Merge(), runs = 0, fr = [0.0].concat(COMPLETE_KERNELS), cOff = [ctr[0] - x0, ctr[1] - y0];
    for (var fi = 0; fi < fr.length; fi++) {
        var img = fr[fi] === 0.0 ? roi : cv.morphOpen(roi, kernelSize(sd / 6.0, fr[fi]));
        var det = detectIn(img, 'window');
        runs++;
        local.add(det.corners, det.ids);
        if (!local.found.has(miss)) redecode(local, roi, [roi, img], det.rejected, sd * COMPLETE_SIDE[0]);
        if (!local.found.has(miss)) {
            var bq = blobQuads(img, sd, cOff, maxOff);
            for (var b = 0; b < bq.length; b++) {
                var out = modelDecode(img, bq[b]);
                if (out !== null && out.id === miss) { local.add([out.corners], [miss]); break; }
            }
        }
        if (local.found.has(miss)) {
            var q = local.found.get(miss), qq = new Float64Array(8);
            for (var i = 0; i < 4; i++) { qq[2 * i] = q[2 * i] + x0; qq[2 * i + 1] = q[2 * i + 1] + y0; }
            var cc = centre(qq);
            if (Math.hypot(cc[0] - ctr[0], cc[1] - ctr[1]) < maxOff * sd) return { q: qq, runs: runs };
            local.found.delete(miss);
        }
    }
    return { q: null, runs: runs };
}

function affineFit(src, dst) {               // lstsq([src, 1], dst).T -> [a b c; d e f] as 6 numbers (flat pts)
    var n = src.length >> 1, A = [], bx = [], by = [];
    for (var i = 0; i < n; i++) { A.push([src[2 * i], src[2 * i + 1], 1]); bx.push(dst[2 * i]); by.push(dst[2 * i + 1]); }
    var sx = cvx.lstsq(A, bx), sy = cvx.lstsq(A, by);
    return [sx[0], sx[1], sx[2], sy[0], sy[1], sy[2]];
}
function finiteAll(a) { for (var i = 0; i < a.length; i++) if (!isFinite(a[i])) return false; return true; }
function concat(list) { var n = 0, i; list.forEach(function (a) { n += a.length; }); var o = new Float64Array(n), p = 0; list.forEach(function (a) { for (i = 0; i < a.length; i++) o[p++] = a[i]; }); return o; }

// omr/detect.py:_complete  -> detector runs used
function complete(m, gray) {
    var runs = 0;
    for (var rep = 0; rep < 2; rep++) {                                   // 2 markers -> 3 -> 4
        var groups = groupsOf(m), gk = Object.keys(groups), g, i;
        if (gk.some(function (k) { return groups[k].length >= 4; })) break;
        var cand = gk.filter(function (k) { return groups[k].length === 3; });
        if (!cand.length) cand = gk.filter(function (k) { return groups[k].length === 2; });
        if (!cand.length || (cand.length > 1 && groups[cand[0]].length === 2)) break;      // two pages with two markers each: nothing to go by
        var page = +cand[0], ids = groups[page].slice().sort(function (a, b) { return a - b; }), mm = pageQuads(page);
        var src = concat(ids.map(function (i) { return mm[i]; })), dst = concat(ids.map(function (i) { return m.found.get(i); })), H;
        if (ids.length === 3) H = cv.findHomography(src, dst);
        else { var A = affineFit(src, dst); H = Float64Array.of(A[0], A[1], A[2], A[3], A[4], A[5], 0, 0, 1); }
        if (!H || !finiteAll(H)) break;
        var got = false;
        for (var miss = 4 * page; miss < 4 * page + 4; miss++) {
            if (m.found.has(miss)) continue;
            var pq = cv.perspectiveTransform(Float64Array.from(mm[miss]), H), sd = side(pq), ctr = centre(pq);
            if (!(finiteAll(pq) && sd > 0)) continue;
            var stages = ids.length === 3 ? [[COMPLETE_ROI, COMPLETE_MAX_OFF], COMPLETE_WIDE] : [COMPLETE_WIDE];
            for (var st = 0; st < stages.length; st++) {
                var r = searchWindow(gray, miss, sd, ctr, stages[st][0], stages[st][1]);
                runs += r.runs;
                if (r.q !== null) { m.add([r.q], [miss]); got = true; break; }
            }
            if (got) break;                                              // re-predict the rest from one more marker
        }
        if (!got) break;
    }
    return runs;
}

// omr/detect.py:_off_frame
function offFrame(m, h, w) {
    var groups = groupsOf(m), gk = Object.keys(groups);
    if (gk.length !== 1) return false;
    var page = +gk[0], ids = groups[page];
    if (!(ids.length >= 1 && ids.length <= 2)) return false;
    ids = ids.slice().sort(function (a, b) { return a - b; });
    var mm = pageQuads(page), src = concat(ids.map(function (i) { return mm[i]; })), dst = concat(ids.map(function (i) { return m.found.get(i); }));
    var A = affineFit(src, dst);
    if (!finiteAll(A)) return false;
    var here = [0, 0], n = dst.length >> 1, i;
    for (i = 0; i < n; i++) { here[0] += dst[2 * i]; here[1] += dst[2 * i + 1]; }
    here[0] /= n; here[1] /= n;
    for (var miss = 4 * page; miss < 4 * page + 4; miss++) {
        if (m.found.has(miss)) continue;
        var q = mm[miss], pq = new Float64Array(8);
        for (i = 0; i < 4; i++) { pq[2 * i] = q[2 * i] * A[0] + q[2 * i + 1] * A[1] + A[2]; pq[2 * i + 1] = q[2 * i] * A[3] + q[2 * i + 1] * A[4] + A[5]; }
        var ctr = centre(pq), sd = side(pq), margin = COMPLETE_WIDE[0] * sd + 0.15 * Math.hypot(ctr[0] - here[0], ctr[1] - here[1]);
        if (-margin <= ctr[0] && ctr[0] <= w + margin && -margin <= ctr[1] && ctr[1] <= h + margin) return false;
    }
    return true;
}

// omr/detect.py:find_markers  gray: uint8 {w,h,data} -> {found: Map, dup, info}
function findMarkers(gray) {
    var m = new Merge(), det = detectIn(gray, 'raw'), rej = det.rejected;
    m.add(det.corners, det.ids);
    var passes = 1, used = 0, minSide = lay.MARKER_MM * MIN_PPM, flat = null, probed = false;
    function info() { return { detect_passes: passes, detect_kernel: used }; }
    if (!m.solid()) redecode(m, gray, [gray], rej, minSide);
    if (!m.solid()) {
        if (offFrame(m, gray.h, gray.w)) return { found: m.found, dup: m.dup, info: info() };
        if (!(m.found.size || markerlike(gray, rej, minSide))) {
            // nothing marker-like on the raw frame: one probe on the lighting-flattened copy
            flat = flattenLighting(gray);
            var d3 = detectIn(flat, 'flat-probe');
            m.add(d3.corners, d3.ids); rej = d3.rejected;
            passes++;
            if (!m.solid()) redecode(m, gray, [gray, flat], rej, minSide);
            if (!m.solid() && !(m.found.size || markerlike(flat, rej, minSide))) return { found: m.found, dup: m.dup, info: info() };
            probed = true;
        }
        var steps = ladder(gray.h, gray.w), hint = probed ? null : m.modulePx();     // (after the flat probe the scale is not trusted yet)
        if (hint !== null) {
            steps = steps.slice().sort(function (a, b) { return Math.abs(Math.log(a / hint)) - Math.abs(Math.log(b / hint)); });
            steps = policy.hintOnly ? [hint] : [hint].concat(steps.slice(0, 2));
        } else {
            var cs = [];
            if (policy.candScale) {
                var cimg = probed && flat !== null ? flat : gray;
                for (var ri = 0; ri < rej.length; ri++) if (side(rej[ri]) >= minSide && modelFit(cimg, rej[ri]) !== null) cs.push(side(rej[ri]) / 6.0);
            }
            if (cs.length) {
                var h2 = cvx.median(cs);
                steps = steps.slice().sort(function (a, b) { return Math.abs(Math.log(a / h2)) - Math.abs(Math.log(b / h2)); }).slice(0, policy.candScale);
            } else steps = steps.length > 2 ? steps.slice(1).concat(steps.slice(0, 1)) : steps;
        }
        var doneK = new Set();
        for (var si = 0; si < steps.length; si++) {
            var k = kernelSize(steps[si], KERNEL_OF_MODULE);
            if (doneK.has(k)) continue;
            doneK.add(k);
            var opened = cv.morphOpen(gray, k), d2 = detectIn(opened, 'opened' + k);
            m.add(d2.corners, d2.ids); rej = d2.rejected;
            passes++;
            if (!m.solid()) redecode(m, gray, [gray, opened], rej, minSide);
            if (m.solid()) { used = k; break; }
        }
        if (!m.solid() && m.found.size) {                // last resort on a lighting-flattened copy (dim / shadowed / gradient-lit frame)
            if (!probed) {
                flat = flattenLighting(gray);
                var d4 = detectIn(flat, 'flat-last');
                m.add(d4.corners, d4.ids); rej = d4.rejected;
                passes++;
                if (!m.solid()) redecode(m, gray, [gray, flat], rej, minSide);
            }
            if (policy.flatOpen && m.found.size && !m.solid()) {
                var kk = kernelSize(m.modulePx(), KERNEL_OF_MODULE), d5 = detectIn(cv.morphOpen(flat, kk), 'flat-opened');
                m.add(d5.corners, d5.ids);
                passes++;
            }
            if (m.solid()) used = -1;
        }
    }
    passes += complete(m, gray);
    return { found: m.found, dup: m.dup, info: info() };
}

// omr/detect.py:find_markers_scaled: frames larger than DETECT_LONG_SIDE are searched on an INTER_AREA copy, corners refined at full size
function findMarkersScaled(gray) {
    var h = gray.h, w = gray.w, s = DETECT_LONG_SIDE / Math.max(h, w);
    if (s >= 1.0) {
        if (policy.coarse && Math.max(h, w) > COARSE_MIN) { var c = findCoarse(gray); if (c !== null) return c; }
        return findMarkers(gray);
    }
    var small = cv.resize(gray, Math.max(rint(w * s), 1), Math.max(rint(h * s), 1), 'area');
    var r = findMarkers(small), sx = w / small.w, sy = h / small.h, out = new Map();
    r.found.forEach(function (c, id) {
        var p = new Float64Array(8);
        for (var i = 0; i < 4; i++) { p[2 * i] = f32((c[2 * i] + 0.5) * sx - 0.5); p[2 * i + 1] = f32((c[2 * i + 1] + 0.5) * sy - 0.5); }
        var win = Math.max(2, rint(0.3 * side(p) / 6));
        out.set(id, refine(gray, p, win));
    });
    r.info.detect_scale = s;
    return { found: out, dup: r.dup, info: r.info };
}

// coarse-to-fine search (policy.coarse) -> {found, dup, info} or null (then the full-size search runs)
function findCoarse(gray) {
    var w = gray.w, h = gray.h, small = cv.resize(gray, rint(w / 2), rint(h / 2), 'area'), d = detectIn(small, 'coarse'), m = new Merge();
    m.add(d.corners, d.ids);
    if (!m.solid() || m.dup || m.conflict.size) return null;
    var sx = w / small.w, sy = h / small.h, full = new Merge(), ok = true;
    m.found.forEach(function (c) { if (side(c) < COARSE_MIN_SIDE) ok = false; });       // marker edge in half-size px
    if (!ok) return null;
    m.found.forEach(function (c, id) {
        var p = new Float64Array(8);
        for (var i = 0; i < 4; i++) { p[2 * i] = f32((c[2 * i] + 0.5) * sx - 0.5); p[2 * i + 1] = f32((c[2 * i + 1] + 0.5) * sy - 0.5); }
        full.add([refine(gray, p, Math.max(2, rint(0.3 * side(p) / 6)))], [id]);
    });
    var passes = 1 + complete(full, gray);
    return { found: full.found, dup: full.dup, info: { detect_passes: passes, detect_kernel: 0, detect_scale: 0.5 } };
}

O.detect = { setPolicy: function (p) { for (var k in p) policy[k] = p[k]; }, policy: policy, setTrace: function (f) { trace = f; }, ladder: ladder, findMarkers: findMarkers, findMarkersScaled: findMarkersScaled, modelDecode: modelDecode, modelFit: modelFit,
    flattenLighting: flattenLighting, markerlike: markerlike, side: side, pageQuads: pageQuads, Merge: Merge, DETECT_LONG_SIDE: DETECT_LONG_SIDE };
})(globalThis.OMR = globalThis.OMR || {});
