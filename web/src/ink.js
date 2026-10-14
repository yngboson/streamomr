(function (O) {
'use strict';
// ink.js - omr/ink.py: self-calibrating mark-darkness ("ink") floor.  Pure functions over darkness arrays.
// (marker_border_samples is not used by the recogniser and is not ported.)  Darkness matrix D = {D: Float32Array(n*m), n, m}.
var f32 = Math.fround;
var cvx = O.cvx, pctRows = O.fastops.pctRows;

var BUBBLE_PERCENTILE = 85;      // per-bubble ink measure: as omr.recognize._ink_level

// omr/ink.py:50  -> Float32Array(n)
function bubbleInk(D, n, m) { return pctRows(D, n, m, BUBBLE_PERCENTILE); }

var LO_FRAC = 0.45, HI_FRAC = 0.60, DEFAULT_FRAC = 0.45, MARGIN_FRAC = 0.06, GHOST_PCT = 95, MULTI_PENALTY = 3.0, MIN_MARK_ROWS = 5,
    MIN_SEPARATION = 0.08, N_CANDIDATES = 31;

function AutoInk(floor, source, separation, weak) { return { floor: floor, source: source, separation: separation, weak: weak }; }

// omr/ink.py:97  rows: array of Float32Array (ink measures of the bubbles of one question / digit column)
function chooseFloor(rows, black) {
    var def = AutoInk(DEFAULT_FRAC * black, 'default', 0.0, true), i, j;
    var rs = [];
    for (i = 0; i < rows.length; i++) if (rows[i].length >= 2) { var s = Float32Array.from(rows[i]).sort(); s.reverse(); rs.push(s); }
    if (rs.length < MIN_MARK_ROWS) return def;
    var w = Infinity;
    for (i = 0; i < rs.length; i++) w = Math.min(w, rs[i].length);
    var nr = rs.length, t1 = new Float32Array(nr), t2 = new Float32Array(nr);
    for (i = 0; i < nr; i++) { t1[i] = rs[i][0]; t2[i] = rs[i][1]; }
    var lo = LO_FRAC * black, hi = HI_FRAC * black;
    var g = cvx.percentile(t2, GHOST_PCT, true);
    var floorMin = Math.max(lo, g + MARGIN_FRAC * black);
    if (floorMin >= hi) return def;
    var cand = new Float64Array(N_CANDIDATES), k;
    for (k = 0; k < N_CANDIDATES; k++) cand[k] = k * ((hi - floorMin) / (N_CANDIDATES - 1)) + floorMin;     // np.linspace
    cand[N_CANDIDATES - 1] = hi;
    var score = new Float32Array(N_CANDIDATES);
    for (k = 0; k < N_CANDIDATES; k++) {
        var t = cand[k], n1 = 0, n2 = 0;         // M >= t: float32 matrix against an np.float64 scalar -> compared as float64
        for (i = 0; i < nr; i++) {
            var c = 0;
            for (j = 0; j < w; j++) if (rs[i][j] >= t) c++;
            if (c === 1) n1++; else if (c >= 2) n2++;
        }
        score[k] = f32(f32(n1) - f32(MULTI_PENALTY * n2));
    }
    var best = -Infinity;
    for (k = 0; k < N_CANDIDATES; k++) if (score[k] > best) best = score[k];
    var plat = new Uint8Array(N_CANDIDATES), thr = f32(best - f32(0.01 * nr));
    for (k = 0; k < N_CANDIDATES; k++) plat[k] = score[k] >= thr ? 1 : 0;
    var k0 = 0;
    for (k = 1; k < N_CANDIDATES; k++) if (score[k] > score[k0]) k0 = k;
    var a = k0, b = k0;
    while (a - 1 >= 0 && plat[a - 1]) a--;
    while (b + 1 < N_CANDIDATES && plat[b + 1]) b++;
    var floor = cand[a];
    var marks = [];
    var ff = f32(floor);
    for (i = 0; i < nr; i++) if (t1[i] >= ff) marks.push(t1[i]);
    if (marks.length < MIN_MARK_ROWS) return def;
    var sep = f32(f32(cvx.percentile(marks, 25, true) - f32(g)) / f32(Math.max(black, 1e-6)));
    if (sep < MIN_SEPARATION) return AutoInk(DEFAULT_FRAC * black, 'default', sep, true);
    return AutoInk(floor, 'auto', sep, false);
}

var LOCAL_RADII = [4, 8, 16], LOCAL_MIN_MARKS = 3, LOCAL_DEADBAND = 0.15, LOCAL_PCT = 75;

// omr/ink.py:150  rowMarks: Float32Array; sections: [[indices, windowed]] -> Float32Array ink level per row
function localLevels(rowMarks, floor, fallback, sections) {
    var out = new Float32Array(rowMarks.length).fill(f32(fallback));
    for (var s = 0; s < sections.length; s++) {
        var idx = sections[s][0], windowed = sections[s][1], n = idx.length;
        if (!n) continue;
        var v = new Float32Array(n), ok = new Uint8Array(n), j, r;
        for (j = 0; j < n; j++) { v[j] = rowMarks[idx[j]]; ok[j] = v[j] >= f32(floor) ? 1 : 0; }
        for (j = 0; j < n; j++) {
            var radii = windowed ? LOCAL_RADII : [n];
            for (var ri = 0; ri < radii.length; ri++) {
                r = radii[ri];
                var lo = Math.max(0, j - r), hi = Math.min(n, j + r + 1), w = [];
                for (var q = lo; q < hi; q++) if (ok[q]) w.push(v[q]);
                if (w.length >= LOCAL_MIN_MARKS) {
                    var lvl = Math.max(floor, cvx.percentile(w, windowed ? LOCAL_PCT : 50, true));
                    out[idx[j]] = lvl < (1.0 - LOCAL_DEADBAND) * fallback ? lvl : fallback;
                    break;
                }
            }
        }
    }
    return out;
}

O.ink = { bubbleInk: bubbleInk, chooseFloor: chooseFloor, localLevels: localLevels, LO_FRAC: LO_FRAC, HI_FRAC: HI_FRAC, DEFAULT_FRAC: DEFAULT_FRAC };
})(globalThis.OMR = globalThis.OMR || {});
