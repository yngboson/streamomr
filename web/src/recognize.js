(function (O) {
'use strict';
// recognize.js - omr/recognize.py minus the decision half (decide.js): one grey camera image -> RecognitionResult.
// Pipeline: ArUco corners -> homography -> canonical warp -> self-describing header decode -> timing tracks -> ring alignment ->
// illumination-robust bubble darkness -> ink floor -> fills -> decide_page.  Result objects as in web/src/TYPES.md.
var f32 = Math.fround;
var cv = O.cv, cvx = O.cvx, lay = O.layout, mk = O.markers, T = O.types, trk = O.tracks, illum = O.illum, inkmod = O.ink, dec = O.decide,
    det = O.detect, pctRows = O.fastops.pctRows, selectK = O.fastops.selectK;
var rint = cv.rint;
function now() { return typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now(); }

var DEFAULT_PX_PER_MM = 6.0;
var OK = T.OK, NO_SHEET = T.NO_SHEET, PARTIAL = T.PARTIAL, INCONSISTENT = T.INCONSISTENT, UNKNOWN_FORM = T.UNKNOWN_FORM, BAD_META = T.BAD_META,
    BAD_GEOMETRY = T.BAD_GEOMETRY;

var MIN_MARKERS = 3, FOREIGN_MIN_SIDE_FRAC = 0.6, FOREIGN_MARGIN_MM = 5.0, MAX_EDGE_RATIO = 3.0, MAX_REPROJ_MM = 2.0, MAX_REPROJ_TRACKS_MM = 8.0,
    META_MIN_CONTRAST = 0.25, HOT_MEAN_DARKNESS = 0.15, INK_PERCENTILE = 50, BLACK_PERCENTILE = 25, ABS_INK_FRAC = 0.45, MIN_CALIB_MARKS = 5,
    STRENGTH_UNCALIBRATED = 0.7;
var ALIGN_PX_PER_MM = 4.0, ALIGN_SAMPLES_PER_GROUP = 40, ALIGN_MIN_CORR = 0.35, ALIGN_SEARCH_MM = 1.4, ALIGN_REFINE_STAGES = 2, ALIGN_MIN_COVERAGE = 0.8,
    ALIGN_MAX_SHIFT_EXTRAP_MM = 4.0, RING_SCORE_SIGMA_MM = 0.5, RING_SCORE_MIN = 0.25, RING_SCORE_FRAC = 0.5, TRACKS_MIN_RING_CORR = 0.4,
    TRACKS_MIN_INLIERS = 0.55, ALIGN_MAX_SHIFT_MM = 2.0, MIN_STRENGTH = 0.6;
var SERIAL_ABSENT_RATIO = 0.65;
var UNRELIABLE_FILL = T.UNRELIABLE_FILL, UNRELIABLE_STRENGTH = T.UNRELIABLE_STRENGTH;
var REL_COVER_PCT = 35, REL_COVER_FRAC = 0.6;

// ------------------------------------------------------------------ small numeric helpers
function mean(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
function arange(start, stop, step) {       // np.arange(start, stop, step) for floats: value i = start + i * ((start + step) - start)
    var n = Math.max(Math.ceil((stop - start) / step), 0), o = new Float64Array(n), delta = (start + step) - start;
    for (var i = 0; i < n; i++) o[i] = start + i * delta;
    if (n > 1) o[1] = start + step;
    return o;
}
function matVec3(H) { return [[H[0], H[1], H[2]], [H[3], H[4], H[5]], [H[6], H[7], H[8]]]; }
function sliceMean(img, y0, y1, x0, x1) {  // img[y0:y1, x0:x1].mean() with Python slice clamping (non-negative starts)
    var h = img.h, w = img.w, d = img.data, s = 0, n = 0;
    y1 = Math.min(y1, h); x1 = Math.min(x1, w);
    for (var y = y0; y < y1; y++) for (var x = x0; x < x1; x++) { s += d[y * w + x]; n++; }
    return s / n;
}
function sliceValues(img, y0, y1, x0, x1) {
    var h = img.h, w = img.w, d = img.data, out = [];
    y1 = Math.min(y1, h); x1 = Math.min(x1, w);
    for (var y = y0; y < y1; y++) for (var x = x0; x < x1; x++) out.push(d[y * w + x]);
    return out;
}

// omr/recognize.py:_marker_corners_mm -> {id: flat 8}
function markerCornersMm(page) { return det.pageQuads(page); }

// omr/recognize.py:_quality_sharpness  quad: flat points of all markers
function qualitySharpness(gray, pts) {
    var minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (var i = 0; i < pts.length; i += 2) { minx = Math.min(minx, pts[i]); maxx = Math.max(maxx, pts[i]); miny = Math.min(miny, pts[i + 1]); maxy = Math.max(maxy, pts[i + 1]); }
    var x0 = Math.max(Math.trunc(minx), 0), y0 = Math.max(Math.trunc(miny), 0), x1 = Math.min(Math.trunc(maxx), gray.w), y1 = Math.min(Math.trunc(maxy), gray.h);
    var rw = Math.max(x1 - x0, 0), rh = Math.max(y1 - y0, 0);
    if (rw * rh < 100) return 0.0;
    var roi = cvx.mk(rw, rh);
    for (var y = y0; y < y1; y++) roi.data.set(gray.data.subarray(y * gray.w + x0, y * gray.w + x1), (y - y0) * rw);
    if (Math.max(rw, rh) > 700) {
        var f = 700 / Math.max(rw, rh);
        roi = cv.resize(roi, rint(rw * f), rint(rh * f), 'area', f, f);
    }
    return cv.laplacianVar(roi);
}

// omr/recognize.py:_plausible_sheet
function plausibleSheet(Hinv, ppm) {
    var q = cv.perspectiveTransform([0, 0, lay.PAGE_W * ppm, 0, lay.PAGE_W * ppm, lay.PAGE_H * ppm, 0, lay.PAGE_H * ppm], Hinv), cr = [], e = [], i;
    for (i = 0; i < 4; i++) {
        var a = (i + 1) % 4, b = (i + 2) % 4;
        var ux = q[2 * a] - q[2 * i], uy = q[2 * a + 1] - q[2 * i + 1], vx = q[2 * b] - q[2 * a], vy = q[2 * b + 1] - q[2 * a + 1];
        cr.push(ux * vy - uy * vx);
        e.push(Math.hypot(ux, uy));
    }
    if (!(cr.every(function (c) { return c > 0; }) || cr.every(function (c) { return c < 0; }))) return false;
    return Math.min.apply(null, e) > 0 && Math.max(e[0], e[2]) / Math.min(e[0], e[2]) < MAX_EDGE_RATIO && Math.max(e[1], e[3]) / Math.min(e[1], e[3]) < MAX_EDGE_RATIO;
}

// =============================================================================== meta strip
function stripMeans(warped, ppm, rects) {
    var means = new Float64Array(rects.length);
    for (var i = 0; i < rects.length; i++) {
        var x = rects[i][0], y = rects[i][1], w = rects[i][2], h = rects[i][3];
        var x0 = (x + 0.2 * w) * ppm, x1 = (x + 0.8 * w) * ppm, y0 = (y + 0.2 * h) * ppm, y1 = (y + 0.8 * h) * ppm;
        means[i] = sliceMean(warped, rint(y0), Math.max(rint(y1), rint(y0) + 1), rint(x0), Math.max(rint(x1), rint(x0) + 1));
    }
    return means;
}
function cellQuartile(warped, ppm, rect) {      // omr/recognize.py:_cell_quartile
    var x = rect[0], y = rect[1], w = rect[2], h = rect[3];
    var ys = rint((y + 0.2 * h) * ppm), ye = Math.max(rint((y + 0.8 * h) * ppm), ys + 1), xs = rint((x + 0.2 * w) * ppm), xe = Math.max(rint((x + 0.8 * w) * ppm), xs + 1);
    return cvx.percentile(sliceValues(warped, ys, ye, xs, xe), BLACK_PERCENTILE);
}
function printedBlack(warped, ppm, rects, whiteL, whiteR) {     // omr/recognize.py:_printed_black
    var vals = [], paper = 0.5 * (whiteL + whiteR), i, n = rects.length;
    mk.START_GUARD.forEach(function (b, i) { if (b) vals.push(cellQuartile(warped, ppm, rects[i])); });
    mk.STOP_GUARD.forEach(function (b, j) { if (b) vals.push(cellQuartile(warped, ppm, rects[n - mk.STOP_GUARD.length + j])); });
    return Math.min(Math.max(1.0 - mean(vals) / Math.max(paper, 1.0), 0.0), 1.0);
}
function guardRefs(means, sg, eg) {
    var n = means.length, bl = [], wl = [], br = [], wr = [];
    sg.forEach(function (b, i) { (b ? bl : wl).push(means[i]); });
    eg.forEach(function (b, i) { (b ? br : wr).push(means[n - eg.length + i]); });
    return { blackL: mean(bl), whiteL: mean(wl), blackR: mean(br), whiteR: mean(wr) };
}
// omr/recognize.py:_read_meta -> {meta, info}
function readMeta(warped, ppm) {
    var rects = lay.metaCellRects(), means = stripMeans(warped, ppm, rects), n = means.length;
    var g = guardRefs(means, mk.START_GUARD, mk.STOP_GUARD);
    var contrast = 1.0 - Math.min(g.blackL / Math.max(g.whiteL, 1), g.blackR / Math.max(g.whiteR, 1));
    var info = { meta_contrast: contrast };
    if (contrast < META_MIN_CONTRAST || Math.min(g.whiteL, g.whiteR) <= Math.min(g.blackL, g.blackR)) return { meta: null, info: info };
    info.meta_black = printedBlack(warped, ppm, rects, g.whiteL, g.whiteR);
    var bits = [], minMargin = Infinity;
    for (var i = 0; i < n; i++) {
        var t = (i - 1.5) / (n - 4.0), black = g.blackL + (g.blackR - g.blackL) * t, white = g.whiteL + (g.whiteR - g.whiteL) * t, mid = 0.5 * (black + white);
        bits.push(means[i] < mid);
        minMargin = Math.min(minMargin, Math.abs(means[i] - mid) / Math.max(white - black, 1));
    }
    info.meta_min_margin = minMargin;
    return { meta: mk.decodeMeta(bits), info: info };
}
// omr/recognize.py:_read_serial -> {serial, state}
function readSerial(warped, ppm) {
    var rects = lay.serialCellRects(), means = stripMeans(warped, ppm, rects), n = means.length, mn = Infinity, mx = -Infinity, i;
    for (i = 0; i < n; i++) { mn = Math.min(mn, means[i]); mx = Math.max(mx, means[i]); }
    if (mn >= SERIAL_ABSENT_RATIO * mx) return { serial: null, state: 'absent' };
    var g = guardRefs(means, mk.START_GUARD, mk.STOP_GUARD);
    var okL = 1.0 - g.blackL / Math.max(g.whiteL, 1) >= META_MIN_CONTRAST && g.whiteL > g.blackL;
    var okR = 1.0 - g.blackR / Math.max(g.whiteR, 1) >= META_MIN_CONTRAST && g.whiteR > g.blackR;
    if (!(okL || okR)) return { serial: null, state: 'absent' };
    var bL = g.blackL, wL = g.whiteL, bR = g.blackR, wR = g.whiteR;
    if (!okL) { bL = bR; wL = wR; }
    if (!okR) { bR = bL; wR = wL; }
    var bits = [];
    for (i = 0; i < n; i++) {
        var t = (i - 1.5) / (n - 4.0), black = bL + (bR - bL) * t, white = wL + (wR - wL) * t;
        bits.push(means[i] < 0.5 * (black + white));
    }
    var v = mk.decodeSerial(bits);
    return v !== null ? { serial: v, state: 'ok' } : { serial: null, state: 'unreadable' };
}

// =============================================================================== self-describing header
var HEADER_SEARCH_MM = 2.0, HEADER_BAND_MM = 1.0, HEADER_WIDE_SEARCH_MM = [4.0, 2.6], HEADER_WIDE_DISAGREE_MM = [3.5, 2.8];
var TRACK_SEG = 8, TRACK_HOP = 4, TRACK_STEP_MM = 1.2, TRACK_FIRST_MM = 3.0, TRACK_MIN_CONTRAST = 40.0;
var HEADER_FLIP_CANDIDATES = 3, HEADER_FLIP_MARGIN = 0.3;
var HEADER_REACH_MM = 40.0, HEADER_FLAT_MARGIN_MM = 4.0, HEADER_FLAT_PCT = 85, HEADER_FLAT_SIGMA_MM = 1.2;

// omr/recognize.py:_box_means: mean over [x0, x1) x [y0, y1) (px, rounded and clipped) from the integral image ii
function boxMean(ii, x0, y0, x1, y1) {
    var H = ii.h - 1, W = ii.w - 1, w1 = ii.w, d = ii.data;
    x0 = Math.min(Math.max(rint(x0), 0), W - 1); x1 = Math.min(Math.max(rint(x1), 1), W);
    y0 = Math.min(Math.max(rint(y0), 0), H - 1); y1 = Math.min(Math.max(rint(y1), 1), H);
    x1 = Math.max(x1, x0 + 1); y1 = Math.max(y1, y0 + 1);
    return (d[y1 * w1 + x1] - d[y0 * w1 + x1] - d[y1 * w1 + x0] + d[y0 * w1 + x0]) / ((x1 - x0) * (y1 - y0));
}
function bandScore(ii, ppm, xs0, xs1, y, h, dy) {      // bright band - dark band - bright band at vertical offset dy
    var inside = boxMean(ii, xs0 * ppm, (y + dy + 0.15 * h) * ppm, xs1 * ppm, (y + dy + 0.85 * h) * ppm);
    var above = boxMean(ii, xs0 * ppm, (y + dy - 0.2 - HEADER_BAND_MM) * ppm, xs1 * ppm, (y + dy - 0.2) * ppm);
    var below = boxMean(ii, xs0 * ppm, (y + dy + h + 0.2) * ppm, xs1 * ppm, (y + dy + h + 0.2 + HEADER_BAND_MM) * ppm);
    return Math.min(above, below) - inside;
}

// omr/recognize.py:_register_row -> [dx, dy] mm
function registerRow(ii, ppm, row, guard, sideLR, search, dyFixed) {
    var cell = sideLR === 'L' ? row[0] : row[row.length - 1], y = cell[1], w = cell[2], h = cell[3], nEnd = 8;
    var xs0 = sideLR === 'L' ? row[0][0] : row[row.length - nEnd][0], xs1 = sideLR === 'L' ? row[nEnd - 1][0] + w : row[row.length - 1][0] + w;
    var step = 1.0 / ppm, best = [-1e9, 0.0], k;
    if (dyFixed === null) {
        var dys = arange(-search[1], search[1] + 1e-9, step);
        if (dys.length) {
            var bk = -1, bs = -Infinity;
            for (k = 0; k < dys.length; k++) { var s = bandScore(ii, ppm, xs0, xs1, y, h, dys[k]); if (s > bs) { bs = s; bk = k; } }
            if (bs > best[0]) best = [bs, dys[bk]];
        }
    }
    var dy = dyFixed === null ? best[1] : dyFixed;
    var pat = sideLR === 'L' ? [false].concat(guard) : guard.concat([false]);
    var xFirst = sideLR === 'L' ? row[0][0] - w : row[row.length - guard.length][0];
    best = [-1e9, 0.0];
    var dxs = arange(-search[0], search[0] + 1e-9, step);
    if (dxs.length) {
        var np = pat.length, vals = new Float64Array(np), bk2 = -1, bs2 = -Infinity;
        for (k = 0; k < dxs.length; k++) {
            var base = xFirst + dxs[k], m = 0, i;
            for (i = 0; i < np; i++) { vals[i] = boxMean(ii, (base + (i + 0.25) * w) * ppm, (y + dy + 0.2 * h) * ppm, (base + (i + 0.75) * w) * ppm, (y + dy + 0.8 * h) * ppm); m += vals[i]; }
            m /= np;
            var sc = 0;
            for (i = 0; i < np; i++) sc += (pat[i] ? -1.0 : 1.0) * (vals[i] - m);
            if (sc > bs2) { bs2 = sc; bk2 = k; }
        }
        if (bs2 > best[0]) best = [bs2, dxs[bk2]];
    }
    return [best[1], dy];
}

// omr/recognize.py:_flatten_header  warped uint8 -> float32 {w,h,data}: the header band divided by the paper level of its own column
function flattenHeader(warped, ppm) {
    var rows = lay.headerCellRects(), W = warped.w, H = warped.h;
    var y0 = Math.max(rint((rows[0][0][1] - HEADER_FLAT_MARGIN_MM) * ppm), 0);
    var y1 = Math.min(rint((rows[rows.length - 1][0][1] + rows[rows.length - 1][0][3] + HEADER_FLAT_MARGIN_MM) * ppm), H);
    var oh = Math.min(y1 + rint(HEADER_REACH_MM * ppm), H), out = new Float32Array(oh * W), i, x, y;
    var src = warped.data;
    for (i = 0; i < oh * W; i++) out[i] = src[i];
    var nb = y1 - y0, col = new Float32Array(nb), paper = new Float32Array(W);
    var vi = (nb - 1) * (HEADER_FLAT_PCT / 100), lo = Math.floor(vi), hi = lo + 1 < nb ? lo + 1 : lo, t = vi - lo;
    for (x = 0; x < W; x++) {
        for (y = 0; y < nb; y++) col[y] = src[(y0 + y) * W + x];
        var a = selectK(col, nb, lo), b = a;           // order statistics lo and hi without a full sort
        if (hi !== lo) { b = Infinity; for (y = lo + 1; y < nb; y++) if (col[y] < b) b = col[y]; }
        var d = f32(b - a), o = f32(a + f32(d * t));
        if (t >= 0.5) o = f32(b - f32(d * (1 - t)));
        paper[x] = o;
    }
    paper = cvx.gaussianBlurRow(paper, HEADER_FLAT_SIGMA_MM * ppm);
    for (x = 0; x < W; x++) paper[x] = Math.max(paper[x], 8.0);
    for (y = y0; y < y1; y++) for (x = 0; x < W; x++) {
        var v = f32(f32(f32(out[y * W + x] / paper[x]) * 235.0));
        out[y * W + x] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
    return { w: W, h: oh, data: out };
}

// omr/recognize.py:_track_dy -> Float64Array(n): vertical offset (mm) of one header row at every cell, followed along the row
function trackDy(ii, ppm, row, firstSearch) {
    var n = row.length, y = row[0][1], w = row[0][2], h = row[0][3], step = 1.0 / ppm, centres = [], dys = [], prev = 0.0;
    var starts = [], i0;
    for (i0 = 0; i0 <= n - TRACK_SEG; i0 += TRACK_HOP) starts.push(i0);
    starts.push(n - TRACK_SEG);
    starts = Array.from(new Set(starts)).sort(function (a, b) { return a - b; });
    for (var k = 0; k < starts.length; k++) {
        i0 = starts[k];
        var xs0 = row[i0][0], xs1 = row[i0 + TRACK_SEG - 1][0] + w, half = k === 0 ? firstSearch : TRACK_STEP_MM;
        var best = [-1e9, prev], cand = arange(prev - half, prev + half + 1e-9, step);
        if (cand.length) {
            var bk = -1, bs = -Infinity;
            for (var q = 0; q < cand.length; q++) { var s = bandScore(ii, ppm, xs0, xs1, y, h, cand[q]); if (s > bs) { bs = s; bk = q; } }
            if (bs > best[0]) best = [bs, cand[bk]];
        }
        if (best[0] >= TRACK_MIN_CONTRAST) prev = best[1];
        centres.push(i0 + (TRACK_SEG - 1) / 2.0);
        dys.push(prev);
    }
    var out = new Float64Array(n);
    for (var i = 0; i < n; i++) {                       // np.interp(arange(n), centres, dys)
        if (i <= centres[0]) out[i] = dys[0];
        else if (i >= centres[centres.length - 1]) out[i] = dys[dys.length - 1];
        else {
            var j = 0; while (centres[j + 1] < i) j++;
            out[i] = dys[j] + (i - centres[j]) * ((dys[j + 1] - dys[j]) / (centres[j + 1] - centres[j]));
        }
    }
    return out;
}

// omr/recognize.py:_read_header -> {hdr, info}
function readHeader(warped, ppm) {
    var flat = flattenHeader(warped, ppm), ii = cv.integral(flat);
    var r = readHeaderPass(flat, ii, ppm, [HEADER_SEARCH_MM, HEADER_SEARCH_MM], [1.2, 1.5], false);
    if (r.hdr === null && (r.info.meta_contrast === undefined ? 1.0 : r.info.meta_contrast) >= META_MIN_CONTRAST) {
        var r2 = readHeaderPass(flat, ii, ppm, HEADER_WIDE_SEARCH_MM, HEADER_WIDE_DISAGREE_MM, false);
        if (r2.hdr !== null) { r2.info.header_wide = 1.0; return r2; }
        var r3 = readHeaderPass(flat, ii, ppm, HEADER_WIDE_SEARCH_MM, [HEADER_WIDE_DISAGREE_MM[0], 8.0], true);
        if (r3.hdr !== null) { r3.info.header_tracked = 1.0; return r3; }
    }
    return r;
}

// omr/recognize.py:_read_header_pass -> {hdr, info}
function readHeaderPass(flat, ii, ppm, search, disagree, track) {
    var rows = lay.headerCellRects(), n = mk.HEADER_ROW_CELLS, g0 = mk.HEADER_START_GUARD.length, g1 = mk.HEADER_STOP_GUARD.length;
    var info = {}, Dm = [], Bm = [], Wm = [], blacks = [], r, i, k;
    var blackGuard = [];
    mk.HEADER_START_GUARD.forEach(function (b, i) { if (b) blackGuard.push(i); });
    mk.HEADER_STOP_GUARD.forEach(function (b, i) { if (b) blackGuard.push(n - g1 + i); });
    blackGuard.sort(function (a, b) { return a - b; });
    for (r = 0; r < rows.length; r++) {
        var row = rows[r], dys = track ? trackDy(ii, ppm, row, TRACK_FIRST_MM) : null;
        var L = registerRow(ii, ppm, row, mk.HEADER_START_GUARD, 'L', search, dys === null ? null : dys[0]);
        var Rr = registerRow(ii, ppm, row, mk.HEADER_STOP_GUARD, 'R', search, dys === null ? null : dys[dys.length - 1]);
        var dxl = L[0], dyl = L[1], dxr = Rr[0], dyr = Rr[1];
        if (Math.abs(dxr - dxl) > disagree[0] || Math.abs(dyr - dyl) > disagree[1]) {      // the two ends disagree beyond any plausible bow
            info.header_reg_mm = Math.max(Math.abs(dxr - dxl), Math.abs(dyr - dyl));
            return { hdr: null, info: info };
        }
        var d = new Float64Array(n), dxA = new Float64Array(n), dyA = new Float64Array(n), paperA = new Float64Array(n);
        for (i = 0; i < n; i++) {
            var t = i / (n - 1), c = row[i], cx = c[0], cy = c[1], cw = c[2], ch = c[3];
            var dx = dxl + (dxr - dxl) * t, dy = dys === null ? dyl + (dyr - dyl) * t : dys[i];
            dxA[i] = dx; dyA[i] = dy;
            var cell = boxMean(ii, (cx + dx + 0.25 * cw) * ppm, (cy + dy + 0.2 * ch) * ppm, (cx + dx + 0.75 * cw) * ppm, (cy + dy + 0.8 * ch) * ppm);
            var up = boxMean(ii, (cx + dx + 0.1 * cw) * ppm, (cy + dy - 0.25 - HEADER_BAND_MM) * ppm, (cx + dx + 0.9 * cw) * ppm, (cy + dy - 0.25) * ppm);
            var dn = boxMean(ii, (cx + dx + 0.1 * cw) * ppm, (cy + dy + ch + 0.25) * ppm, (cx + dx + 0.9 * cw) * ppm, (cy + dy + ch + 0.25 + HEADER_BAND_MM) * ppm);
            paperA[i] = Math.max(0.5 * (up + dn), 1.0);
            d[i] = 1.0 - cell / paperA[i];
        }
        for (k = 0; k < blackGuard.length; k++) {      // printed black: darker quartile of the cell's pixels (robust to white toner streaks)
            i = blackGuard[k];
            var cc = row[i], x = cc[0], y = cc[1], w = cc[2], h = cc[3];
            var y0_ = rint((y + dyA[i] + 0.2 * h) * ppm), x0_ = rint((x + dxA[i] + 0.2 * w) * ppm);
            var vals = sliceValues(flat, Math.max(y0_, 0), Math.max(rint((y + dyA[i] + 0.8 * h) * ppm), y0_ + 1), Math.max(x0_, 0),
                Math.max(rint((x + dxA[i] + 0.8 * w) * ppm), x0_ + 1));
            if (vals.length) blacks.push(1.0 - cvx.percentile(vals, BLACK_PERCENTILE, true) / paperA[i]);
        }
        var sb = [], sw = [], eb = [], ew = [];
        mk.HEADER_START_GUARD.forEach(function (b, i) { (b ? sb : sw).push(i); });
        mk.HEADER_STOP_GUARD.forEach(function (b, i) { (b ? eb : ew).push(n - g1 + i); });
        function mean_(idx) { var s = 0; idx.forEach(function (j) { s += d[j]; }); return s / idx.length; }
        var bl = mean_(sb), br = mean_(eb), wl = mean_(sw), wr = mean_(ew), Bv = new Float64Array(n), Wv = new Float64Array(n);
        for (i = 0; i < n; i++) { var tt = i / (n - 1); Bv[i] = bl + (br - bl) * tt; Wv[i] = wl + (wr - wl) * tt; }
        Dm.push(d); Bm.push(Bv); Wm.push(Wv);
    }
    var contrast = Infinity, bsum = 0;
    for (r = 0; r < rows.length; r++) for (i = 0; i < n; i++) { contrast = Math.min(contrast, Bm[r][i] - Wm[r][i]); bsum += Bm[r][i]; }
    info.meta_contrast = contrast;
    if (contrast < META_MIN_CONTRAST) return { hdr: null, info: info };
    info.meta_black = blacks.length ? Math.min(Math.max(mean(blacks), 0.0), 1.0) : bsum / (rows.length * n);       // printed black
    var data = [], dmargin = [], minMargin = Infinity;
    for (r = 0; r < rows.length; r++) {
        var bits = [];
        for (i = 0; i < n; i++) {
            var thr = 0.5 * (Bm[r][i] + Wm[r][i]);
            bits.push(Dm[r][i] > thr);
            var mg = Math.abs(Dm[r][i] - thr) / Math.max(Bm[r][i] - Wm[r][i], 1e-6);
            minMargin = Math.min(minMargin, mg);
            if (i >= g0 && i < n - g1) { data.push(Dm[r][i] > thr); dmargin.push(mg); }
        }
        var okg = true;
        for (i = 0; i < g0; i++) if (bits[i] !== mk.HEADER_START_GUARD[i]) okg = false;
        for (i = 0; i < g1; i++) if (bits[n - g1 + i] !== mk.HEADER_STOP_GUARD[i]) okg = false;
        if (!okg) { info.meta_min_margin = minMargin; return { hdr: null, info: info }; }
    }
    var dmin = Infinity;
    dmargin.forEach(function (m) { dmin = Math.min(dmin, m); });
    info.meta_min_margin = dmin;
    var h = mk.decodeHeaderData(data);
    if (h === null) {
        var order = dmargin.map(function (m, i) { return i; }).sort(function (a, b) { return dmargin[a] - dmargin[b] || a - b; });
        var weak = order.slice(0, HEADER_FLIP_CANDIDATES).filter(function (i) { return dmargin[i] < HEADER_FLIP_MARGIN; });
        var tries = weak.map(function (i) { return [i]; });
        for (var a = 0; a < weak.length; a++) for (var b = a + 1; b < weak.length; b++) tries.push([weak[a], weak[b]]);
        for (var ti = 0; ti < tries.length; ti++) {
            var d2 = data.slice();
            tries[ti].forEach(function (j) { d2[j] = !d2[j]; });
            h = mk.decodeHeaderData(d2);
            if (h !== null) { info.header_flips = tries[ti].length; break; }
        }
    }
    return { hdr: h, info: info };
}

// =============================================================================== scoring
var ringTmplCache = {};
// omr/recognize.py:_ring_offsets  small: float32 {w,h,data}; centers [[x, y] mm] -> {dx, dy, corr} Float64Array(n) (NaN where the window leaves the image)
function ringOffsets(small, ppmS, centers, diameterMm, searchMm) {
    var R = diameterMm / 2, srch = rint(searchMm * ppmS), half = Math.ceil((R + 0.7) * ppmS), key = R + '_' + ppmS + '_' + half;
    var tmpl = ringTmplCache[key], tw = 2 * half + 1, x, y, i, j;
    if (!tmpl) {
        tmpl = cvx.mk(tw, tw, Float32Array);
        for (y = -half; y <= half; y++) for (x = -half; x <= half; x++) {
            var rr = f32(f32(Math.sqrt(f32(f32(x * x) + f32(y * y)))) / ppmS), z = f32(f32(rr - R) / 0.3);
            tmpl.data[(y + half) * tw + x + half] = Math.exp(f32(-0.5 * f32(z * z)));
        }
        ringTmplCache[key] = tmpl;
    }
    var H = small.h, W = small.w, n = centers.length, dxo = new Float64Array(n).fill(NaN), dyo = new Float64Array(n).fill(NaN), cro = new Float64Array(n).fill(NaN);
    var pad = half + srch, ww = 2 * pad + 1, win = cvx.mk(ww, ww, Float32Array), rw = 2 * srch + 1;
    for (i = 0; i < n; i++) {
        var cx = rint(centers[i][0] * ppmS), cy = rint(centers[i][1] * ppmS);
        if (cx - pad < 0 || cy - pad < 0 || cx + pad >= W || cy + pad >= H) continue;
        if (small.ensure) small.ensure(cx - pad, cy - pad, cx + pad + 1, cy + pad + 1);
        for (y = 0; y < ww; y++) for (x = 0; x < ww; x++) win.data[y * ww + x] = small.data[(cy - pad + y) * W + cx - pad + x];
        var r = cv.matchTemplateCcoeffNormed(win, tmpl).data;
        var best = 0;
        for (j = 1; j < r.length; j++) if (r[j] > r[best]) best = j;
        var j0 = Math.floor(best / rw), j1 = best % rw, b = r[best], dy = j0 - srch, dx = j1 - srch;
        if (j0 > 0 && j0 < rw - 1) {
            var a = r[(j0 - 1) * rw + j1], c = r[(j0 + 1) * rw + j1];
            dy = dy + f32(f32(0.5 * f32(a - c)) / f32(f32(f32(a - f32(2 * b)) + c) + f32(1e-9)));
        }
        if (j1 > 0 && j1 < rw - 1) {
            var a2 = r[j0 * rw + j1 - 1], c2 = r[j0 * rw + j1 + 1];
            dx = dx + f32(f32(0.5 * f32(a2 - c2)) / f32(f32(f32(a2 - f32(2 * b)) + c2) + f32(1e-9)));
        }
        dxo[i] = dx / ppmS; dyo[i] = dy / ppmS; cro[i] = b;
    }
    return { dx: dxo, dy: dyo, corr: cro };
}

function polyTerms(xy) {                    // omr/recognize.py:_poly_terms -> array of 6-vectors
    var out = [];
    for (var i = 0; i < xy.length; i++) {
        var x = (xy[i][0] - lay.PAGE_W / 2) / 100.0, y = (xy[i][1] - lay.PAGE_H / 2) / 100.0;
        out.push([1, x, y, x * x, x * y, y * y]);
    }
    return out;
}
function lstsq2(A, B, keep) {               // columns of B (n x 2) -> 6x2 coefficients, rows with keep[i]
    var rowsA = [], b0 = [], b1 = [];
    for (var i = 0; i < A.length; i++) if (keep[i]) { rowsA.push(A[i]); b0.push(B[i][0]); b1.push(B[i][1]); }
    var c0 = cvx.lstsq(rowsA, b0), c1 = cvx.lstsq(rowsA, b1), out = [];
    for (var k = 0; k < A[0].length; k++) out.push([c0[k], c1[k]]);
    return out;
}
function applyCoef(A, coef) {               // A @ coef -> n x 2
    var out = [];
    for (var i = 0; i < A.length; i++) {
        var x = 0, y = 0;
        for (var k = 0; k < A[i].length; k++) { x += A[i][k] * coef[k][0]; y += A[i][k] * coef[k][1]; }
        out.push([x, y]);
    }
    return out;
}
// omr/recognize.py:_robust_fit
function robustFit(A, offs, good) {
    var n = A.length, keep = good.slice(), coef = lstsq2(A, offs, keep), i;
    for (var it = 0; it < 4; it++) {          // iteratively re-weighted: drop outliers (marked bubbles, glyph overlap)
        coef = lstsq2(A, offs, keep);
        var pr = applyCoef(A, coef), res = new Float64Array(n), rk = [];
        for (i = 0; i < n; i++) { res[i] = Math.hypot(pr[i][0] - offs[i][0], pr[i][1] - offs[i][1]); if (keep[i]) rk.push(res[i]); }
        var sig = Math.max(1.4826 * cvx.median(rk), 0.05), nk = [], cnt = 0, same = true;
        for (i = 0; i < n; i++) { nk.push(good[i] && res[i] < 3 * sig + 0.05); if (nk[i]) cnt++; if (nk[i] !== keep[i]) same = false; }
        if (cnt < 12 || same) break;
        keep = nk;
    }
    return { coef: coef, keep: keep };
}

// omr/recognize.py:_fit_alignment  groups: [[centers, diameter]] -> {coef (6x2 | null), info}.  diag receives {small, ppm_s}
function fitAlignment(warped, groups, ppm, extrapolated, strict, diag) {
    var ppmS = ALIGN_PX_PER_MM, sw_ = rint(warped.w * (ppmS / ppm)), sh_ = rint(warped.h * (ppmS / ppm)), sc_ = 1 / (ppmS / ppm);
    // the reduced inverted canvas is only read around the bubbles: evaluated lazily by 32x32 tiles (identical values to cv.resize of the NOT image)
    var small = cv.lazyAreaResize(warped, sw_, sh_, sc_, sc_, true);
    if (small === null) {
        var inv = cv.bitwiseNot(warped), sm = cv.resize(inv, sw_, sh_, 'area', ppmS / ppm, ppmS / ppm);
        small = { w: sm.w, h: sm.h, data: Float32Array.from(sm.data) };
    }
    var sels = groups.map(function (g) {
        var c = g[0], stp = Math.max(1, Math.floor(c.length / ALIGN_SAMPLES_PER_GROUP)), s = [];
        for (var i = 0; i < c.length; i += stp) s.push(c[i]);
        return [s, g[1]];
    });
    var pts = [];
    sels.forEach(function (s) { s[0].forEach(function (p) { pts.push(p); }); });
    var A = polyTerms(pts), coef = null, info = {}, keep = null, offs = null, nst = 1 + (extrapolated ? ALIGN_REFINE_STAGES : 0), i;
    for (var stage = 0; stage < nst; stage++) {
        var pred = coef === null ? pts.map(function () { return [0, 0]; }) : applyCoef(A, coef);
        offs = []; var cor = [], i0 = 0;
        for (var si = 0; si < sels.length; si++) {
            var sel = sels[si][0], dia = sels[si][1], p = pred.slice(i0, i0 + sel.length);
            var ro = ringOffsets(small, ppmS, sel.map(function (c, j) { return [c[0] + p[j][0], c[1] + p[j][1]]; }), dia, ALIGN_SEARCH_MM);
            for (i = 0; i < sel.length; i++) { offs.push([ro.dx[i] + p[i][0], ro.dy[i] + p[i][1]]); cor.push(ro.corr[i]); }
            i0 += sel.length;
        }
        var good = cor.map(function (c) { return isFinite(c) && c > ALIGN_MIN_CORR; }), ng = good.filter(Boolean).length;
        var fin = cor.filter(isFinite);
        info = { align_ring_corr: fin.length ? cvx.median(fin) : 0.0, align_points: ng, align_stages: stage + 1 };
        if (ng < 12) return { coef: null, info: info };
        var rf = robustFit(A, offs, good);
        coef = rf.coef; keep = rf.keep;
    }
    if (diag) { diag.small = small; diag.ppm_s = ppmS; }
    var pr = applyCoef(A, coef), rk = [], kc = 0, maxShift = 0;
    for (i = 0; i < A.length; i++) {
        if (keep[i]) { rk.push(Math.hypot(pr[i][0] - offs[i][0], pr[i][1] - offs[i][1])); kc++; }
        maxShift = Math.max(maxShift, Math.hypot(pr[i][0], pr[i][1]));
    }
    info.align_resid_mm = cvx.median(rk);
    info.align_used = kc;
    info.align_inlier_frac = kc / keep.length;
    info.align_max_shift_mm = maxShift;
    // a page straightened by its tracks may still be off by a few mm in the interior: the larger shift is accepted when the rings back the model
    var wide = extrapolated || (strict && info.align_inlier_frac >= ALIGN_MIN_COVERAGE);
    if (info.align_max_shift_mm > (wide ? ALIGN_MAX_SHIFT_EXTRAP_MM : ALIGN_MAX_SHIFT_MM)) return { coef: null, info: info };
    if (extrapolated && info.align_inlier_frac < ALIGN_MIN_COVERAGE) return { coef: null, info: info };      // rings not confirmed over the extrapolated part
    if (strict && (info.align_ring_corr < TRACKS_MIN_RING_CORR || info.align_inlier_frac < TRACKS_MIN_INLIERS)) return { coef: null, info: info };
    return { coef: coef, info: info };
}

// omr/recognize.py:_ring_scores  centers: array of [x, y] mm -> Float64Array(n)
function ringScores(small, ppmS, centers, diameterMm) {
    var R = diameterMm / 2, half = Math.ceil((R + 1.0) * ppmS), tw = 2 * half + 1, x, y, i, t = new Float32Array(tw * tw), s = 0, nrm = 0;
    for (y = -half; y <= half; y++) for (x = -half; x <= half; x++) {
        var rr = f32(f32(Math.sqrt(f32(f32(x * x) + f32(y * y)))) / ppmS), z = f32(f32(rr - R) / RING_SCORE_SIGMA_MM);
        t[(y + half) * tw + x + half] = Math.exp(f32(-0.5 * f32(z * z)));
    }
    for (i = 0; i < t.length; i++) s += t[i];
    s /= t.length;
    for (i = 0; i < t.length; i++) { t[i] = f32(t[i] - s); nrm += t[i] * t[i]; }
    nrm = Math.sqrt(nrm);
    for (i = 0; i < t.length; i++) t[i] = f32(t[i] / nrm);
    var H = small.h, W = small.w, out = new Float64Array(centers.length), P = new Float32Array(tw * tw);
    for (var n = 0; n < centers.length; n++) {
        var cx = rint(centers[n][0] * ppmS), cy = rint(centers[n][1] * ppmS), m = 0;
        if (small.ensure) small.ensure(Math.max(cx - half, 0), Math.max(cy - half, 0), Math.min(cx + half + 1, W), Math.min(cy + half + 1, H));
        for (y = 0; y < tw; y++) {
            var yy = Math.min(Math.max(cy - half + y, 0), H - 1);
            for (x = 0; x < tw; x++) { var xx = Math.min(Math.max(cx - half + x, 0), W - 1); P[y * tw + x] = small.data[yy * W + xx]; m += P[y * tw + x]; }
        }
        m /= P.length;
        var num = 0, pn = 0;
        for (i = 0; i < P.length; i++) { var pv = f32(P[i] - m); num += pv * t[i]; pn += pv * pv; }
        out[n] = num / Math.max(Math.sqrt(pn), 1e-3);
    }
    return out;
}
// omr/recognize.py:_rows_misfit  groups: [[centers (rows of [x, y]), diameter, need]] -> array of Uint8Array(rows) (1 = row misfit)
function rowsMisfit(diag, groups) {
    var scores = groups.map(function (g) { var flat = []; g[0].forEach(function (r) { r.forEach(function (c) { flat.push(c); }); }); return ringScores(diag.small, diag.ppm_s, flat, g[1]); });
    var all = [];
    scores.forEach(function (s) { for (var i = 0; i < s.length; i++) all.push(s[i]); });
    var ref = cvx.percentile(Float32Array.from(all), 75, true), thr = f32(Math.max(RING_SCORE_MIN, RING_SCORE_FRAC * ref));
    diag.ring_ref = ref;
    return groups.map(function (g, gi) {
        var rows = g[0].length, per = g[0][0].length, out = new Uint8Array(rows);
        for (var r = 0; r < rows; r++) {
            var c = 0;
            for (var j = 0; j < per; j++) if (f32(scores[gi][r * per + j]) >= thr) c++;
            out[r] = c < g[2] ? 1 : 0;
        }
        return out;
    });
}
function applyAlignment(centers, coef) {
    if (coef === null) return centers.map(function (c) { return [c[0], c[1]]; });
    var A = polyTerms(centers), sh = applyCoef(A, coef);
    return centers.map(function (c, i) { return [c[0] + sh[i][0], c[1] + sh[i][1]]; });
}

// omr/recognize.py:_ink_level  groups: [{D, n, m}], p85s: [Float32Array] -> {level (number | null), n}
function inkLevel(groups, floor, p85s) {
    var vals = [], ff = f32(floor);
    for (var gi = 0; gi < groups.length; gi++) {
        var g = groups[gi];
        for (var r = 0; r < g.n; r++) {
            var s = 0; for (var j = 0; j < g.m; j++) s += g.D[r * g.m + j];
            if (f32(s / g.m) > f32(HOT_MEAN_DARKNESS) && p85s[gi][r] >= ff) vals.push(p85s[gi][r]);
        }
    }
    if (vals.length < MIN_CALIB_MARKS) return { level: null, n: vals.length };
    return { level: cvx.percentile(vals, INK_PERCENTILE, true), n: vals.length };
}

// omr/recognize.py:_fill  D: {D, n, m}; ink: Float32Array(n) per bubble; unreliable: Uint8Array(n) -> {fill: Float32Array(n), strength: Float64Array(n)}
function fillOf(Dm, ink, unreliable) {
    var n = Dm.n, m = Dm.m, D = Dm.D, fill = new Float32Array(n), st = new Float64Array(n), c035 = f32(0.35), c03 = f32(0.3);
    for (var i = 0; i < n; i++) {
        var lo = f32(c035 * ink[i]), sp = f32(c03 * ink[i]), sf = 0, sd = 0, cnt = 0;
        for (var j = 0; j < m; j++) {
            var d = D[i * m + j], q = f32(f32(d - lo) / sp);
            sf += q < 0 ? 0 : q > 1 ? 1 : q;
            if (d >= lo) { sd += d; cnt++; }
        }
        fill[i] = sf / m;
        st[i] = f32(sd) / Math.max(cnt, 1) / ink[i];
        if (unreliable && unreliable[i]) { fill[i] = UNRELIABLE_FILL; st[i] = UNRELIABLE_STRENGTH; }
    }
    return { fill: fill, strength: st };
}
// omr/recognize.py:_mean_dark -> Float32Array(n)
function meanDark(Dm, unreliable, p85) {
    var p35 = pctRows(Dm.D, Dm.n, Dm.m, REL_COVER_PCT), out = new Float32Array(Dm.n), c = f32(REL_COVER_FRAC);
    for (var i = 0; i < Dm.n; i++) out[i] = p35[i] >= f32(c * p85[i]) ? p85[i] : 0.0;
    if (unreliable) for (i = 0; i < Dm.n; i++) if (unreliable[i]) out[i] = NaN;
    return out;
}
function rowsOf(flat, nrows, ncols, T_) {    // flat vector -> array of typed rows (copies)
    var out = [];
    for (var r = 0; r < nrows; r++) out.push(T_.from(flat.subarray(r * ncols, (r + 1) * ncols)));
    return out;
}
function round2(x) { return rint(x * 100) / 100; }

// =============================================================================== main
var DROP_RETRY_STATUSES = [BAD_GEOMETRY, BAD_META], DROP_RETRY_SKIP_FLAGS = ['page_mismatch', 'layout_mismatch'];

// omr/recognize.py:recognize.  gray: {w, h, data: Uint8Array}; opts: {resolveLayout, pxPerMm, keepImages}
function recognize(gray, opts) {
    opts = opts || {};
    var pxPerMm = opts.pxPerMm === undefined ? DEFAULT_PX_PER_MM : opts.pxPerMm, keep = opts.keepImages === undefined ? true : opts.keepImages;
    var tStart = now(), t0 = now(), d = det.findMarkersScaled(gray), tDet = now() - t0;
    var res = recognizeWith(gray, d, opts.resolveLayout || null, pxPerMm, keep, tStart, tDet);
    if (DROP_RETRY_STATUSES.indexOf(res.status) >= 0 && !DROP_RETRY_SKIP_FLAGS.some(function (f) { return res.flags.indexOf(f) >= 0; })) {
        var page = res.page, ids = [];
        d.found.forEach(function (c, i) { if (page !== null && Math.floor(i / 4) === page) ids.push(i); });
        if (ids.length === 4) {
            var worst = worstMarker(d.found, ids, page), red = new Map();
            d.found.forEach(function (c, i) { if (i !== worst) red.set(i, c); });
            var info2 = {};
            for (var k in d.info) info2[k] = d.info[k];
            info2.dropped_marker = worst;
            var res2 = recognizeWith(gray, { found: red, dup: d.dup, info: info2 }, opts.resolveLayout || null, pxPerMm, keep, tStart, tDet);
            if (res2.status === OK) { res2.flags.push('marker_dropped'); return res2; }
            res.timing_ms.total = now() - tStart;
        }
    }
    return res;
}

// omr/recognize.py:_worst_marker
function worstMarker(found, ids, page) {
    var mm = markerCornersMm(page), src = [], dst = [];
    ids.forEach(function (i) { var c = found.get(i); for (var k = 0; k < 8; k++) src.push(c[k]); mm[i].forEach(function (v) { dst.push(v); }); });
    var H = cv.findHomography(src, dst);
    if (H === null) return ids[0];
    var p = cv.perspectiveTransform(src, H), best = -1, bi = ids[0];
    for (var n = 0; n < ids.length; n++) {
        var e = 0;
        for (var k = 0; k < 4; k++) e += Math.hypot(p[8 * n + 2 * k] - dst[8 * n + 2 * k], p[8 * n + 2 * k + 1] - dst[8 * n + 2 * k + 1]);
        e /= 4;
        if (e > best) { best = e; bi = ids[n]; }
    }
    return bi;
}

// omr/recognize.py:_recognize 'smooth': GaussianBlur(warped, sigma 0.25 mm) - computed only inside the bounding box of the bubbles (+ the reach of the
// ring search / paper annulus / contrast ring, 12 mm, + the kernel radius); everything outside is the unblurred canvas and is never read.
function smoothAround(warped, pts, dia, ppm) {
    var minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity, i;
    for (i = 0; i < pts.length; i++) { minx = Math.min(minx, pts[i][0]); maxx = Math.max(maxx, pts[i][0]); miny = Math.min(miny, pts[i][1]); maxy = Math.max(maxy, pts[i][1]); }
    var m = Math.ceil((12 + dia) * ppm) + 8, x0 = Math.max(Math.floor(minx * ppm) - m, 0), y0 = Math.max(Math.floor(miny * ppm) - m, 0);
    var x1 = Math.min(Math.ceil(maxx * ppm) + m, warped.w), y1 = Math.min(Math.ceil(maxy * ppm) + m, warped.h), W = warped.w;
    var sub = cvx.mk(x1 - x0, y1 - y0);
    for (var y = y0; y < y1; y++) sub.data.set(warped.data.subarray(y * W + x0, y * W + x1), (y - y0) * sub.w);
    var b = cv.gaussianBlur(sub, 0.25 * ppm), out = { w: W, h: warped.h, data: warped.data.slice() };
    for (y = y0; y < y1; y++) out.data.set(b.data.subarray((y - y0) * sub.w, (y - y0 + 1) * sub.w), y * W + x0);
    return out;
}

// omr/recognize.py:_recognize
function recognizeWith(gray, detRes, resolveLayout, ppm, keepImages, tStart, tDet) {
    var tm = {};
    function lap(name, t0) { tm[name] = now() - t0; }
    var PROF = O.recognize.profile, tp = now();
    function lapx(name) { if (PROF) { var t = now(); tm['x_' + name] = t - tp; tp = t; } }
    function done(res) { tm.total = now() - tStart; res.timing_ms = tm; return res; }

    // ---- 1. markers (found by the caller)
    var found = new Map(detRes.found), dup = detRes.dup, dinfo = detRes.info, i, k;
    tm.detect = tDet;
    var res = T.RecognitionResult(NO_SHEET, ppm);
    if (!found.size) return done(res);
    res.marker_ids = Array.from(found.keys()).sort(function (a, b) { return a - b; });
    res.marker_corners = [];
    found.forEach(function (c, id) { var q = []; for (var j = 0; j < 4; j++) q.push([c[2 * j], c[2 * j + 1]]); res.marker_corners.push([id, q]); });
    if (dup) { res.status = INCONSISTENT; res.flags.push('duplicate_marker'); return done(res); }
    var byGroup = {};
    found.forEach(function (c, id) { (byGroup[Math.floor(id / 4)] = byGroup[Math.floor(id / 4)] || []).push(id); });
    var gks = Object.keys(byGroup), solid = gks.filter(function (g) { return byGroup[g].length >= MIN_MARKERS; });
    if (solid.length > 1) { res.status = INCONSISTENT; res.flags.push('mixed_page_markers'); return done(res); }
    if (!solid.length) {
        // fewer than MIN_MARKERS of any page group: the sheet is in view but not usable (covered / cut off)
        var top = Math.max.apply(null, gks.map(function (g) { return byGroup[g].length; }));
        var lead = gks.filter(function (g) { return byGroup[g].length === top; });
        if (lead.length > 1) { res.status = INCONSISTENT; res.flags.push('mixed_page_markers'); return done(res); }
        var pg0 = +lead[0];
        res.status = PARTIAL; res.page = pg0;
        res.missing_corners = mk.CORNERS.filter(function (c) { return !found.has(mk.markerId(pg0, c)); });
        return done(res);
    }
    var page = +solid[0], foreignIds = [];
    found.forEach(function (c, id) { if (Math.floor(id / 4) !== page) foreignIds.push(id); });
    var kept = new Map();
    found.forEach(function (c, id) { if (Math.floor(id / 4) === page) kept.set(id, c); });
    found = kept;
    res.missing_corners = mk.CORNERS.filter(function (c) { return !found.has(mk.markerId(page, c)); });

    // ---- 2. homography from all visible corner points (>= 12)
    var t0 = now(), mm = markerCornersMm(page), idsSorted = Array.from(found.keys()).sort(function (a, b) { return a - b; });
    var src = [], dst = [];
    idsSorted.forEach(function (id) {
        var c = found.get(id);
        for (var j = 0; j < 8; j++) { src.push(c[j]); dst.push(mm[id][j] * ppm); }
    });
    var Hm = cv.findHomography(src, dst);
    if (Hm === null || !Hm.every(isFinite)) { res.status = BAD_GEOMETRY; return done(res); }
    var proj = cv.perspectiveTransform(src, Hm), npt = src.length >> 1, errMm = new Float64Array(npt);
    for (i = 0; i < npt; i++) errMm[i] = Math.hypot(proj[2 * i] - dst[2 * i], proj[2 * i + 1] - dst[2 * i + 1]) / ppm;
    var Hinv = cv.invert3x3(Hm), back = cv.perspectiveTransform(dst, Hinv), errPx = 0;
    for (i = 0; i < npt; i++) errPx = Math.max(errPx, Math.hypot(back[2 * i] - src[2 * i], back[2 * i + 1] - src[2 * i + 1]));
    var cxy = cv.perspectiveTransform([lay.PAGE_W / 2 * ppm, lay.PAGE_H / 2 * ppm], Hinv);
    var dd = cv.perspectiveTransform([lay.PAGE_W / 2 * ppm + ppm, lay.PAGE_H / 2 * ppm, lay.PAGE_W / 2 * ppm, lay.PAGE_H / 2 * ppm + ppm], Hinv);
    var srcPpm = Math.sqrt(Math.hypot(dd[0] - cxy[0], dd[1] - cxy[1]) * Math.hypot(dd[2] - cxy[0], dd[3] - cxy[1]));
    // displacement (mm) of every visible corner marker in the warp: anchors of the timing-track chains
    var cornerD = {};
    idsSorted.forEach(function (id, n_) {
        var sx = 0, sy = 0;
        for (var j = 0; j < 4; j++) { sx += proj[2 * (4 * n_ + j)] - dst[2 * (4 * n_ + j)]; sy += proj[2 * (4 * n_ + j) + 1] - dst[2 * (4 * n_ + j) + 1]; }
        cornerD[mk.CORNERS[id % 4]] = [sx / 4 / ppm, sy / 4 / ppm];
    });
    res.homography = matVec3(Hm);
    res.page = page;
    var errMax = 0, errSq = 0;
    for (i = 0; i < npt; i++) { errMax = Math.max(errMax, errMm[i]); errSq += errMm[i] * errMm[i]; }
    lapx('pre_homog');
    res.quality = { reproj_max_mm: errMax, reproj_rms_mm: Math.sqrt(errSq / npt), reproj_max_px: errPx, source_px_per_mm: srcPpm,
        sharpness: qualitySharpness(gray, src), foreign_markers: foreignIds.length, markers_used: found.size };
    for (k in dinfo) res.quality[k] = dinfo[k];
    lap('homography', t0);
    lapx('sharp+homog');
    // a sheet with timing tracks may be bent far more than a plain one: a large marker reprojection error is then only a suspicion
    var geomSuspect = MAX_REPROJ_MM < errMax && errMax <= MAX_REPROJ_TRACKS_MM;
    if ((errMax > MAX_REPROJ_MM && !geomSuspect) || !plausibleSheet(Hinv, ppm)) { res.status = BAD_GEOMETRY; return done(res); }
    // markers of another page group are tolerated only when they lie outside this sheet (another sheet / object)
    var allCorners = {}; res.marker_corners.forEach(function (e) { allCorners[e[0]] = e[1]; });
    for (var fi = 0; fi < foreignIds.length; fi++) {
        var fq = allCorners[foreignIds[fi]], flat = [];
        fq.forEach(function (p) { flat.push(p[0], p[1]); });
        var q = cv.perspectiveTransform(flat, Hm), ss = 0;
        for (k = 0; k < 4; k++) q[2 * k] /= ppm, q[2 * k + 1] /= ppm;
        for (k = 0; k < 4; k++) ss += Math.hypot(q[2 * ((k + 1) % 4)] - q[2 * k], q[2 * ((k + 1) % 4) + 1] - q[2 * k + 1]);
        if (ss / 4 < FOREIGN_MIN_SIDE_FRAC * lay.MARKER_MM) continue;      // far smaller than a corner marker, yet inside the sheet: a pattern in the print
        var ctr = [(q[0] + q[2] + q[4] + q[6]) / 4, (q[1] + q[3] + q[5] + q[7]) / 4], overlaps = false;
        found.forEach(function (c) {
            var pc = cv.perspectiveTransform(c, Hm), mc = [(pc[0] + pc[2] + pc[4] + pc[6]) / 4 / ppm, (pc[1] + pc[3] + pc[5] + pc[7]) / 4 / ppm];
            if (Math.hypot(ctr[0] - mc[0], ctr[1] - mc[1]) < lay.MARKER_MM) overlaps = true;
        });
        if (overlaps) continue;      // two markers cannot share a spot, it is a mis-decode of that marker's surroundings
        if (-FOREIGN_MARGIN_MM <= ctr[0] && ctr[0] <= lay.PAGE_W + FOREIGN_MARGIN_MM && -FOREIGN_MARGIN_MM <= ctr[1] && ctr[1] <= lay.PAGE_H + FOREIGN_MARGIN_MM) {
            res.status = INCONSISTENT; res.flags.push('mixed_page_markers'); return done(res);
        }
    }
    if (res.missing_corners.length) res.flags.push('corner_hidden');

    // ---- 3. warp
    t0 = now();
    var cw = rint(lay.PAGE_W * ppm), ch = rint(lay.PAGE_H * ppm);
    var warped = cv.warpPerspective(gray, Hm, cw, ch, { interp: 'linear', border: 'constant', value: 255 });
    lap('warp', t0);
    res.warped = keepImages ? warped : null;
    lapx('warp');

    // ---- 4. header: the sheet describes its own layout (legacy sheets: 16-bit form id, layout from the resolver)
    t0 = now();
    var tfield = null;
    if (geomSuspect) {       // only a sheet whose tracks bend it flat is accepted
        tfield = trk.measure(warped, ppm, cornerD);
        if (tfield === null) { res.status = BAD_GEOMETRY; return done(res); }
        warped = trk.rectify(warped, tfield, ppm);
        res.warped = keepImages ? warped : null;
    }
    lapx('foreign');
    var rh = readHeader(warped, ppm), hdr = rh.hdr, info = rh.info;
    lapx('header');
    if (geomSuspect && (hdr === null || hdr.version < 2)) { res.status = BAD_GEOMETRY; return done(res); }       // not a bent sheet with tracks
    var metaPre = null, metaSrc = warped;
    if (hdr === null) metaPre = readMeta(warped, ppm);        // a readable legacy strip: an older sheet, nothing to straighten
    if (hdr === null && metaPre.meta === null) {
        // header unreadable on the plain warp: a folded / curled sheet with timing tracks reads after they straighten it
        tfield = trk.measure(warped, ppm, cornerD);
        if (tfield !== null) {
            var flat_ = trk.rectify(warped, tfield, ppm), rh2 = readHeader(flat_, ppm);
            hdr = rh2.hdr;
            if (hdr !== null && hdr.version >= 2) { info = rh2.info; warped = flat_; res.warped = keepImages ? warped : null; }
            else { hdr = null; tfield = null; }
        }
    }
    for (k in info) res.quality[k] = info[k];
    var layout, hpage, total;
    if (hdr !== null) {
        layout = lay.layoutForHeader(hdr); hpage = hdr.page; total = hdr.pages;
    } else {
        var mr = (metaPre !== null && warped === metaSrc) ? metaPre : readMeta(warped, ppm);
        if (mr.meta === null) { lap('meta', t0); res.status = BAD_META; return done(res); }
        for (k in mr.info) res.quality[k] = mr.info[k];
        res.form_id = mr.meta.form_id;
        hpage = mr.meta.page; total = mr.meta.total_pages;
        layout = resolveLayout ? resolveLayout(mr.meta.form_id) : null;
        if (!layout) { lap('meta', t0); res.status = UNKNOWN_FORM; return done(res); }
        if (layout.form_id !== mr.meta.form_id) { lap('meta', t0); res.status = BAD_META; res.flags.push('layout_mismatch'); return done(res); }
    }
    lap('meta', t0);
    lapx('meta_rest');
    res.form_key = layout.form_key; res.total_pages = total;
    if (hpage !== page) { res.status = BAD_META; res.flags.push('page_mismatch'); return done(res); }
    if (layout.total_pages !== total || page >= layout.pages.length) { res.status = BAD_META; res.flags.push('layout_mismatch'); return done(res); }
    var pg = layout.pages[page];

    // ---- 4b. timing tracks (header version >= 2): bend the page flat again, so everything below sits at its nominal place
    t0 = now();
    if (pg.ticks && pg.ticks.length) {
        if (tfield === null) {          // not yet measured (the header read fine on the plain warp)
            tfield = trk.measure(warped, ppm, cornerD);
            if (tfield !== null && trk.maxDisp(tfield) >= trk.FLAT_MM) {
                warped = trk.rectify(warped, tfield, ppm);
                res.warped = keepImages ? warped : null;
            } else if (tfield !== null) {                    // flat sheet: the tracks confirm the homography, nothing to resample
                var ti = trk.info(tfield);
                for (k in ti) res.quality[k] = ti[k];
                res.quality.tracks = 1.0;
                tfield = trk.FLAT;
            }
        }
        if (tfield !== trk.FLAT) {
            res.quality.tracks = tfield !== null ? 1.0 : 0.0;
            if (tfield !== null) {
                var ti2 = trk.info(tfield);
                for (k in ti2) res.quality[k] = ti2[k];
                var tf = tfield;
                res.dewarp = function (x, y) { return trk.fieldAt(tf, x, y); };
                res.dewarp.field = tf;
            }
        }
    }
    lap('tracks', t0);
    lapx('tracks');

    // ---- 5. bubbles
    t0 = now();
    var smooth = null;      // blurred below, only where bubbles are read (needs the layout's bubble extent)
    // serial strip: a failure never invalidates the frame (serial stays null, soft flag)
    var rs = readSerial(warped, ppm);
    res.serial = rs.serial;
    if (rs.state === 'unreadable') res.flags.push('serial_unreadable');
    var qs = [];
    pg.columns.forEach(function (col) { col.questions.forEach(function (q) { qs.push(q); }); });
    qs.sort(function (a, b) { return a.number - b.number; });
    var nc = layout.template.choices, cent = [];
    qs.forEach(function (q) { q.bubbles.forEach(function (b) { cent.push([b.cx, b.cy]); }); });
    var g = pg.id_grid, sg = pg.subject_grid, icols = g.columns.concat(sg ? sg.columns : []), nId = g.columns.length, icent = [];
    icols.forEach(function (col) { col.bubbles.forEach(function (b) { icent.push([b.cx, b.cy]); }); });
    // the ID and subject grids share one bubble geometry: one group for alignment and scoring
    lapx('serial+layout');
    smooth = smoothAround(warped, cent.concat(icent), pg.bubble_diameter, ppm);
    lapx('blur');
    var adiag = {};
    var fa = fitAlignment(smooth, [[cent, pg.bubble_diameter], [icent, g.bubble_diameter]], ppm, res.missing_corners.length > 0 && tfield === null,
        tfield !== null && tfield !== trk.FLAT, adiag);
    var coef = fa.coef;
    lapx('align');
    for (k in fa.info) res.quality[k] = fa.info[k];
    cent = applyAlignment(cent, coef);
    icent = applyAlignment(icent, coef);
    var nq = qs.length, ni = icols.length;
    res.bubble_centers_mm = {
        answers: qs.map(function (q, r) { var row = []; for (var c = 0; c < nc; c++) row.push([round2(cent[r * nc + c][0]), round2(cent[r * nc + c][1])]); return row; }),
        id: [], subject: [] };
    for (i = 0; i < ni; i++) {
        var irow = [];
        for (var v = 0; v < 10; v++) irow.push([round2(icent[i * 10 + v][0]), round2(icent[i * 10 + v][1])]);
        (i < nId ? res.bubble_centers_mm.id : res.bubble_centers_mm.subject).push(irow);
    }
    var DA = illum.bubbleDarkness(smooth, cent, pg.bubble_diameter, ppm), DI = illum.bubbleDarkness(smooth, icent, g.bubble_diameter, ppm);
    lapx('illum');
    var UA = new Uint8Array(DA.n), UI = new Uint8Array(DI.n);         // illum returns "reliable"; the fills want "unreliable"
    for (i = 0; i < DA.n; i++) UA[i] = DA.reliable[i] ? 0 : 1;
    for (i = 0; i < DI.n; i++) UI[i] = DI.reliable[i] ? 0 : 1;
    if (coef !== null && tfield !== null && tfield !== trk.FLAT && adiag.small) {      // a straightened page: interior check
        var rowsA = [], rowsI = [];
        for (i = 0; i < nq; i++) rowsA.push(cent.slice(i * nc, (i + 1) * nc));
        for (i = 0; i < ni; i++) rowsI.push(icent.slice(i * 10, (i + 1) * 10));
        var bad = rowsMisfit(adiag, [[rowsA, pg.bubble_diameter, nc >= 4 ? 2 : 1], [rowsI, g.bubble_diameter, 4]]), badQ = bad[0], badI = bad[1], nbad = 0;
        for (i = 0; i < DA.n; i++) UA[i] = UA[i] | badQ[Math.floor(i / nc)];
        for (i = 0; i < DI.n; i++) UI[i] = UI[i] | badI[Math.floor(i / 10)];
        for (i = 0; i < badQ.length; i++) nbad += badQ[i];
        for (i = 0; i < badI.length; i++) nbad += badI[i];
        res.quality.rows_misfit = nbad;
    }
    var black = Math.min(Math.max(res.quality.meta_black === undefined ? 0.9 : res.quality.meta_black, META_MIN_CONTRAST), 1.0);   // printed-black darkness on this frame
    // the sheet chooses its own ink floor (never below ABS_INK_FRAC x black); illum before ink: unreliable bubbles are 0 in D
    var inkA_ = inkmod.bubbleInk(DA.D, DA.n, DA.m), inkI_ = inkmod.bubbleInk(DI.D, DI.n, DI.m), rows = [];
    for (i = 0; i < nq; i++) rows.push(inkA_.subarray(i * nc, (i + 1) * nc));
    for (i = 0; i < ni; i++) rows.push(inkI_.subarray(i * 10, (i + 1) * 10));
    var ia = inkmod.chooseFloor(rows, black), absInk = ia.floor;
    var il = inkLevel([DA, DI], absInk, [inkA_, inkI_]), pageInk = il.level;
    var ink = Math.max(absInk, pageInk || 0.0), minStr = pageInk !== null ? MIN_STRENGTH : STRENGTH_UNCALIBRATED;
    res.quality.ink_abs = absInk; res.quality.ink_marks_used = il.n; res.quality.unreliable_bubbles = (function () { var s = 0; for (var q = 0; q < UA.length; q++) s += UA[q]; for (q = 0; q < UI.length; q++) s += UI[q]; return s; })();
    // local ink levels: pencil ID block next to dark-pen answers, or a fresh pen half-way through, must each be judged by their own neighbourhood
    function hotMax(Dm, p85, nrows, ncols) {
        var out = new Float32Array(nrows);
        for (var r = 0; r < nrows; r++) {
            var mx = -Infinity;
            for (var c = 0; c < ncols; c++) {
                var b = r * ncols + c, s = 0;
                for (var j = 0; j < Dm.m; j++) s += Dm.D[b * Dm.m + j];
                var v = f32(s / Dm.m) > f32(HOT_MEAN_DARKNESS) ? p85[b] : 0.0;
                if (v > mx) mx = v;
            }
            out[r] = mx;
        }
        return out;
    }
    var rmA = hotMax(DA, inkA_, nq, nc), rmI = hotMax(DI, inkI_, ni, 10), rm = new Float32Array(nq + ni);
    rm.set(rmA); rm.set(rmI, nq);
    var idxA = [], idxI = [];
    for (i = 0; i < nq; i++) idxA.push(i);
    for (i = nq; i < nq + ni; i++) idxI.push(i);
    var lv = inkmod.localLevels(rm, absInk, ink, [[idxA, true], [idxI, false]]);
    var inkA = new Float32Array(nq * nc), inkI = new Float32Array(ni * 10), lmin = Infinity, lmax = -Infinity;
    for (i = 0; i < nq * nc; i++) inkA[i] = lv[Math.floor(i / nc)];
    for (i = 0; i < ni * 10; i++) inkI[i] = lv[nq + Math.floor(i / 10)];
    for (i = 0; i < lv.length; i++) { lmin = Math.min(lmin, lv[i]); lmax = Math.max(lmax, lv[i]); }
    res.quality.ink_local_span = lmax / Math.max(lmin, 1e-6);
    var FAv = fillOf(DA, inkA, UA), FIv = fillOf(DI, inkI, UI);
    var MAv = meanDark(DA, UA, inkA_), MIv = meanDark(DI, UI, inkI_);
    var FA = rowsOf(FAv.fill, nq, nc, Float32Array), SA = rowsOf(FAv.strength, nq, nc, Float64Array);
    var FI = rowsOf(FIv.fill, ni, 10, Float32Array), SI = rowsOf(FIv.strength, ni, 10, Float64Array);
    var MA = rowsOf(MAv, nq, nc, Float32Array), MI = rowsOf(MIv, ni, 10, Float32Array);
    var raw = T.PageRaw(qs.map(function (q) { return q.number; }), qs.map(function (q) { return q.bubbles.map(function (b) { return b.label; }); }),
        g.columns.map(function (col) { return col.digit_index; }), FA, SA, FI.slice(0, nId), SI.slice(0, nId), ink, minStr, coef !== null,
        { FS: sg ? FI.slice(nId) : null, SS: sg ? SI.slice(nId) : null, ink_source: ia.source, ink_separation: ia.separation,
          MA: MA, MI: MI.slice(0, nId), MS: sg ? MI.slice(nId) : null });
    lapx('ink+fill');
    var dc = dec.decidePage(raw);
    lapx('decide');
    res.answers = dc.answers; res.id_digits = dc.id_digits; res.student_id = dc.student_id; res.calibration = dc.calibration;
    var idflags = dc.flags;
    res.subject_digits = dc.subject_digits; res.subject = dc.subject;
    res.answers = res.answers.slice().sort(function (a, b) { return a.number - b.number; });
    res.raw = raw;
    raw.decided_sig = dec.decisionSig(dc.answers, res.student_id, res.subject);     // what `fusion.trustworthy` compares against
    if (coef === null) res.flags.push('alignment_unverified');      // bubble positions could not be verified against the printed rings: trust nothing
    idflags.forEach(function (f) { res.flags.push(f); });
    lap('score', t0);

    // ---- 7. name box crop (kept so a later stage can read/OCR it)
    if (keepImages) {
        var nb = pg.name_box, x = nb[0], y = nb[1], w = nb[2], h = nb[3];
        var nx0 = Math.trunc(x * ppm), nx1 = Math.trunc((x + w) * ppm), ny0 = Math.trunc(y * ppm), ny1 = Math.trunc((y + h) * ppm);
        nx1 = Math.min(nx1, warped.w); ny1 = Math.min(ny1, warped.h);
        var crop = cvx.mk(Math.max(nx1 - nx0, 0), Math.max(ny1 - ny0, 0));
        for (var yy = ny0; yy < ny1; yy++) crop.data.set(warped.data.subarray(yy * warped.w + nx0, yy * warped.w + nx1), (yy - ny0) * crop.w);
        res.name_crop = crop;
    } else res.name_crop = null;
    res.status = OK;
    return done(res);
}


// JIT warm-up (call once at start-up, e.g. while the page says "preparing"): runs the heavy kernels over synthetic 1080x1920 data so that
// JavaScriptCore has compiled them before the first camera frame (without it the first two frames are ~2x and ~1.7x slower).  Touches no
// state that recognition reads.  Returns the elapsed ms.
function warmup(reps) {
    var t0 = now(), W = 1080, H = 1920, n = W * H, i, k, seed = 12345;
    var d = new Uint8Array(n);
    for (i = 0; i < n; i++) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; d[i] = 150 + ((seed >> 16) & 63); }
    for (var by = 200; by < 1700; by += 300) for (var bx = 100; bx < 900; bx += 250) {          // dark squares with a bright inner one: marker-like blobs
        for (var y = by; y < by + 90; y++) for (var x = bx; x < bx + 90; x++) d[y * W + x] = (x - bx > 20 && x - bx < 70 && y - by > 20 && y - by < 70) ? 220 : 30;
    }
    var gray = { w: W, h: H, data: d };
    reps = reps || 2;
    for (k = 0; k < reps; k++) {
        try { det.findMarkersScaled(gray); } catch (e) { }                                  // raw / flat-probe / opened passes: threshold, contours, morphology, flatten
        var Hm = Float64Array.of(1, 0.02, 10, 0.01, 1, 20, 0.00001, 0.00002, 1);
        var canvas = cv.warpPerspective(gray, Hm, 1260, 1782, { interp: 'linear', border: 'constant', value: 255 });
        var sm = cv.gaussianBlur(canvas, 1.5);
        var lz = cv.lazyAreaResize(sm, 840, 1188, 1.5, 1.5, true);
        if (lz) lz.ensure(0, 0, 840, 1188);
        cv.resize(canvas, 700, 990, 'area', 700 / 1260, 990 / 1782);
        var win = cvx.mk(31, 31, Float32Array), tm = cvx.mk(19, 19, Float32Array);
        for (i = 0; i < win.data.length; i++) win.data[i] = d[i * 7 % n];
        for (i = 0; i < tm.data.length; i++) tm.data[i] = (i * 13) % 17;
        for (i = 0; i < 60; i++) cv.matchTemplateCcoeffNormed(win, tm);
        try {
            var cen = lay.trackTickCenters(), tr = {};
            ['L', 'R', 'T', 'B'].forEach(function (sd) { var c = cen[sd], dd = new Float64Array(2 * c.length); for (var q = 0; q < c.length; q++) { dd[2 * q] = 0.3 * Math.sin(q); dd[2 * q + 1] = 0.2 * Math.cos(q); } tr[sd] = trk.Track(sd, c, dd, new Float64Array(c.length).fill(1)); });
            trk.rectify(canvas, trk.Field(tr), DEFAULT_PX_PER_MM);
        } catch (e) { }
    }
    return now() - t0;
}

O.recognize = { profile: false, warmup: warmup, recognize: recognize, readHeader: readHeader, readMeta: readMeta, readSerial: readSerial, fitAlignment: fitAlignment, ringOffsets: ringOffsets,
    fillOf: fillOf, flattenHeader: flattenHeader, readHeaderPass: readHeaderPass, registerRow: registerRow, trackDy: trackDy, meanDark: meanDark, inkLevel: inkLevel, worstMarker: worstMarker, DEFAULT_PX_PER_MM: DEFAULT_PX_PER_MM };
})(globalThis.OMR = globalThis.OMR || {});
