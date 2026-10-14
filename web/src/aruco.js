(function (O) {
'use strict';
// aruco.js - cv2.aruco.ArucoDetector (DICT_4X4_50) in plain JS.  Needs cv.js.
// Reimplements OpenCV's detectMarkers: multi-window adaptive threshold -> contour candidates -> polygon filter ->
// close-candidate removal -> perspective bit extraction (Otsu) -> dictionary match -> sub-pixel corner refinement.
var cv = O.cv;
var aruco = {};

// DICT_4X4_50 exported from cv2 (getPredefinedDictionary(DICT_4X4_50).bytesList): 50 markers x 4 rotations, 16 bits row-major MSB first.
var CODES = Uint16Array.from([46386,60232,19629,4823,3994,25927,23024,58022,13101,56849,46284,34939,39238,49468,25241,15491,21662,41427,31018,52101,31181,55479,45982,60699,40494,34653,29817,47841,50418,9194,20259,22468,65242,44527,23423,63413,53078,26108,27379,16294,63889,63630,35231,28959,4519,54034,58760,18635,3767,14166,60784,27372,10767,7445,61524,43192,9393,14914,36132,16988,9790,12113,31844,35572,18021,5872,42594,3944,26112,3264,102,816,27742,10741,31286,44948,30383,40915,62830,52217,34443,5451,53601,53928,45099,39689,54285,37081,52437,12542,43827,32524,56706,49614,16827,29571,65095,40444,57983,16313,38001,45672,36393,5709,44260,2686,10037,32336,42324,26744,10917,7702,8483,23296,50308,218,13423,39793,63020,36569,17429,12496,43042,2828,22450,59330,19946,17383,40655,38271,62329,65193,61643,39339,54031,54681,2222,791,29968,59584,2345,20997,38032,41034,6261,45620,44568,11341,1279,13171,65312,52940,3574,25462,28592,28358,7258,41317,23096,42629,5912,58433,6376,33319,10792,3589,5204,41072,12940,35859,12620,51249,14514,43782,19740,24789,9448,2659,5924,50768,12011,8039,55156,59128,11583,31573,64692,43742,19300,18100,9938,11618,20526,33681,29706,35265,20499,45440,51210,397]);
aruco.codes = CODES; aruco.markerSize = 4; aruco.maxCorrectionBits = 1;

// omr/detect.py:_detector() parameters on top of OpenCV defaults
aruco.defaultParams = {
    adaptiveThreshWinSizeMin: 3, adaptiveThreshWinSizeMax: 23, adaptiveThreshWinSizeStep: 10, adaptiveThreshConstant: 7,
    minMarkerPerimeterRate: 0.03, maxMarkerPerimeterRate: 4.0, polygonalApproxAccuracyRate: 0.03, minCornerDistanceRate: 0.05,
    minDistanceToBorder: 3, minMarkerDistanceRate: 0.125, markerBorderBits: 1,
    perspectiveRemovePixelPerCell: 8, perspectiveRemoveIgnoredMarginPerCell: 0.13, maxErroneousBitsInBorderRate: 0.5,
    minOtsuStdDev: 5.0, errorCorrectionRate: 0.6,
    cornerRefinement: true, cornerRefinementWinSize: 4, relativeCornerRefinmentWinSize: 0.3, cornerRefinementMaxIterations: 30,
    cornerRefinementMinAccuracy: 0.1
};

// Dictionary.identify: bits = Uint8Array(16) row-major (1 = white) -> {id, rotation} or null
aruco.identify = function aruco_identify(bits, errorCorrectionRate) {
    var code = 0;
    for (var i = 0; i < 16; i++) code = (code << 1) | bits[i];
    return identifyCode(code, errorCorrectionRate === undefined ? 0.6 : errorCorrectionRate);
};
var POP = new Uint8Array(65536);
for (var pi = 1; pi < 65536; pi++) POP[pi] = POP[pi >> 1] + (pi & 1);
function identifyCode(code, rate) {
    var maxCorr = Math.floor(aruco.maxCorrectionBits * rate);
    for (var m = 0; m < 50; m++) {
        var best = 17, rot = -1;
        for (var r = 0; r < 4; r++) { var d = POP[CODES[m * 4 + r] ^ code]; if (d < best) { best = d; rot = r; } }
        if (best <= maxCorr) return { id: m, rotation: rot };
    }
    return null;
}

// ---- candidate search (aruco_detector.cpp _findMarkerContours)
function findCandidates(gray, P, win, binBuf, out, sizes) {
    var w = gray.w, h = gray.h, mx = Math.max(w, h);
    var minPer = Math.floor(P.minMarkerPerimeterRate * mx), maxPer = Math.floor(P.maxMarkerPerimeterRate * mx), bd = P.minDistanceToBorder;
    cv.adaptiveThreshold(gray, win, P.adaptiveThreshConstant, binBuf, true);
    var contours = cv.findContoursPadded(binBuf, w, h, 'list', 'none', minPer, maxPer);
    for (var i = 0; i < contours.length; i++) {
        var c = contours[i], n = c.length >> 1;
        if (n < minPer || n > maxPer) continue;
        var ap = cv.approxPolyDP(c, n * P.polygonalApproxAccuracyRate, true);
        if (ap.length !== 8 || !cv.isContourConvex(ap)) continue;
        var minDistSq = mx * mx, j;
        for (j = 0; j < 4; j++) {
            var dx = ap[2 * j] - ap[2 * ((j + 1) & 3)], dy = ap[2 * j + 1] - ap[2 * ((j + 1) & 3) + 1], d = dx * dx + dy * dy;
            if (d < minDistSq) minDistSq = d;
        }
        var mcd = n * P.minCornerDistanceRate;
        if (minDistSq < mcd * mcd) continue;
        var near = false;
        for (j = 0; j < 4; j++) {
            var x = ap[2 * j], y = ap[2 * j + 1];
            if (x < bd || y < bd || x > w - 1 - bd || y > h - 1 - bd) { near = true; break; }
        }
        if (near) continue;
        var cand = new Float64Array(8);
        for (j = 0; j < 8; j++) cand[j] = ap[j];
        var cross = (cand[2] - cand[0]) * (cand[5] - cand[1]) - (cand[3] - cand[1]) * (cand[4] - cand[0]);
        if (cross < 0) { var tx = cand[2], ty = cand[3]; cand[2] = cand[6]; cand[3] = cand[7]; cand[6] = tx; cand[7] = ty; }
        out.push(cand); sizes.push(n);
    }
}

// _filterTooCloseCandidates: candidates whose corners are closer than minMarkerDistanceRate * min contour length are grouped
// (OpenCV's quirky pairwise grouping, kept literally); each group keeps its candidate with the longest quad perimeter (first wins ties; measured against cv2, the area is NOT the criterion).
function filterTooClose(cands, sizes, rate) {
    var n = cands.length, i, j, fc, c;
    var cx = new Float64Array(n), cy = new Float64Array(n);
    for (i = 0; i < n; i++) { var a = cands[i]; cx[i] = (a[0] + a[2] + a[4] + a[6]) / 4; cy[i] = (a[1] + a[3] + a[5] + a[7]) / 4; }
    var group = new Int32Array(n).fill(-1), groups = [];
    for (i = 0; i < n; i++) {
        var single = true;
        for (j = i + 1; j < n; j++) {
            var T = Math.min(sizes[i], sizes[j]) * rate, T2 = T * T;
            var ddx = cx[i] - cx[j], ddy = cy[i] - cy[j];
            if (ddx * ddx + ddy * ddy >= T2) continue;        // mean corner displacement >= centroid displacement: no rotation can be close
            var A = cands[i], B = cands[j];
            for (fc = 0; fc < 4; fc++) {
                var dist = 0;
                for (c = 0; c < 4; c++) {
                    var mc = (c + fc) & 3, ex = A[2 * mc] - B[2 * c], ey = A[2 * mc + 1] - B[2 * c + 1];
                    dist += ex * ex + ey * ey;
                }
                dist /= 4;
                if (dist < T2) {
                    single = false;
                    if (group[i] < 0) {
                        if (group[j] < 0) { group[i] = group[j] = groups.length; groups.push([i, j]); }
                        else { group[i] = group[j]; groups[group[i]].push(i); }
                    } else if (group[j] < 0) { group[j] = group[i]; groups[group[i]].push(j); }
                    break;
                }
            }
        }
        if (single && group[i] < 0) { group[i] = groups.length; groups.push([i, i]); }
    }
    var oc = [], os = [];
    for (i = 0; i < groups.length; i++) {
        var g = groups[i], bi = g[0], ba = cv.arcLength(cands[bi], true);
        for (j = 1; j < g.length; j++) { var ar = cv.arcLength(cands[g[j]], true); if (ar > ba) { bi = g[j]; ba = ar; } }
        oc.push(cands[bi]); os.push(sizes[bi]);
    }
    return { cands: oc, sizes: os };
}

// ---- bit extraction (_extractBits) + identification (_identifyOneCandidate)
var RES = new Uint8Array(64 * 64);
var BITS = new Uint8Array(64);
function identifyOne(gray, corner, P) {
    var ms = 4, mb = P.markerBorderBits, sz = ms + 2 * mb, cell = P.perspectiveRemovePixelPerCell;
    var margin = Math.floor(P.perspectiveRemoveIgnoredMarginPerCell * cell), rs = sz * cell, i, x, y;
    var dst = [0, 0, rs - 1, 0, rs - 1, rs - 1, 0, rs - 1];
    var M = cv.invert3x3(cv.getPerspectiveTransform(corner, dst));       // result -> image
    var w = gray.w, h = gray.h, g = gray.data;
    // warpPerspective(INTER_NEAREST, border 0) of the candidate to rs x rs
    for (y = 0; y < rs; y++) {
        var X0 = M[1] * y + M[2], Y0 = M[4] * y + M[5], W0 = M[7] * y + M[8], row = y * rs;
        for (x = 0; x < rs; x++) {
            var Wd = W0 + M[6] * x; Wd = Wd ? 1 / Wd : 0;
            var sx = cv.rint((X0 + M[0] * x) * Wd), sy = cv.rint((Y0 + M[3] * x) * Wd);
            RES[row + x] = (sx >= 0 && sx < w && sy >= 0 && sy < h) ? g[sy * w + sx] : 0;
        }
    }
    // standard deviation of the inner region (border of cell/2 removed); flat -> all black or all white
    var half = cell >> 1, s = 0, s2 = 0, cnt = 0;
    for (y = half; y < rs - half; y++) for (i = y * rs + half; i < y * rs + rs - half; i++) { var v = RES[i]; s += v; s2 += v * v; cnt++; }
    var mean = s / cnt, vr = s2 / cnt - mean * mean, std = Math.sqrt(vr > 0 ? vr : 0);
    var nb = sz * sz;
    if (std < P.minOtsuStdDev) {
        BITS.fill(mean > 127 ? 1 : 0, 0, nb);
    } else {
        var t = cv.otsuThreshold(RES, rs * rs), area = cell - 2 * margin, half2 = (area * area) >> 1;
        for (y = 0; y < sz; y++) for (x = 0; x < sz; x++) {
            var nz = 0;
            for (var yy = y * cell + margin; yy < y * cell + cell - margin; yy++)
                for (var xx = x * cell + margin; xx < x * cell + cell - margin; xx++) if (RES[yy * rs + xx] > t) nz++;
            BITS[y * sz + x] = nz >= half2 ? 1 : 0;       // measured against cv2: a cell with exactly half its pixels white reads 1
        }
    }
    // border errors: white cells in the border ring
    var errs = 0, k;
    for (y = 0; y < sz; y++) for (k = 0; k < mb; k++) { errs += BITS[y * sz + k] !== 0 ? 1 : 0; errs += BITS[y * sz + sz - 1 - k] !== 0 ? 1 : 0; }
    for (x = mb; x < sz - mb; x++) for (k = 0; k < mb; k++) { errs += BITS[k * sz + x] !== 0 ? 1 : 0; errs += BITS[(sz - 1 - k) * sz + x] !== 0 ? 1 : 0; }
    if (errs > Math.floor(ms * ms * P.maxErroneousBitsInBorderRate)) return null;
    var code = 0;
    for (y = mb; y < sz - mb; y++) for (x = mb; x < sz - mb; x++) code = (code << 1) | BITS[y * sz + x];
    return identifyCode(code, P.errorCorrectionRate);
}

aruco._identifyOne = identifyOne;
var fr = Math.fround;
var binBuf = null;
// detect(gray {w,h,data:Uint8Array}, params?) -> {corners:[Float64Array(8)], ids:[int], rejected:[Float64Array(8)]}
aruco.detect = function aruco_detect(gray, params) {
    var P = Object.assign({}, aruco.defaultParams, params || {});
    var cands = [], sizes = [], win;
    if (!binBuf || binBuf.length < (gray.w + 2) * (gray.h + 2)) binBuf = new Int8Array((gray.w + 2) * (gray.h + 2));
    for (win = P.adaptiveThreshWinSizeMin; win <= P.adaptiveThreshWinSizeMax; win += P.adaptiveThreshWinSizeStep)
        findCandidates(gray, P, win, binBuf, cands, sizes);
    var f = filterTooClose(cands, sizes, P.minMarkerDistanceRate);
    var corners = [], ids = [], rejected = [];
    for (var i = 0; i < f.cands.length; i++) {
        var c = f.cands[i], r = identifyOne(gray, c, P);
        if (r === null) { rejected.push(c); continue; }
        var rc = new Float64Array(8);
        for (var k = 0; k < 4; k++) { var s = (k + 4 - r.rotation) & 3; rc[2 * k] = c[2 * s]; rc[2 * k + 1] = c[2 * s + 1]; }
        corners.push(rc); ids.push(r.id);
    }
    if (P.cornerRefinement) {
        for (i = 0; i < corners.length; i++) {
            var q = corners[i], per = 0;
            for (k = 0; k < 4; k++) per += Math.hypot(q[2 * k] - q[2 * ((k + 1) & 3)], q[2 * k + 1] - q[2 * ((k + 1) & 3) + 1]);
            var module = per / (4 * (4 + 2 * P.markerBorderBits));
            var wsz = Math.min(P.cornerRefinementWinSize, Math.max(1, cv.rint(P.relativeCornerRefinmentWinSize * module)));
            corners[i] = cv.cornerSubPix(gray, q, wsz, P.cornerRefinementMaxIterations, P.cornerRefinementMinAccuracy);
        }
    }
    return { corners: corners, ids: ids, rejected: rejected };
};

O.aruco = aruco;
})(globalThis.OMR = globalThis.OMR || {});
