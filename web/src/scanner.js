// Port of standalone/glue.py (the browser glue): RGBA frame -> grey -> recognize -> StreamSession.feed -> event (JSON string).
// Scanner.step mirrors _frame_step in app/main.py for the "all subjects (auto)" path.  No database, no JPEG round trip of the frame
// (the browser hands over raw RGBA), no legacy-form lookup (resolveLayout = null: sheets describe themselves in their header).
(function (O) {
'use strict';
var D = O.decide, T = O.types, pyRound = O.util.pyRound, F32 = Math.fround;

var MIN_FRAME_SIDE = 64;      // same as app/main.py
var DEV = false;  // __DEV__: scripts/build_web.py --dev sets this true (extra per-frame diagnostics for web/dev.js)

function now() { return (typeof performance !== 'undefined' ? performance.now() : Date.now()); }

// ---- JSON clean-up (glue.py::_clean): drop "*_path" keys, typed arrays / Sets / Maps -> plain
function clean(o) {
  if (o === null || typeof o !== 'object') return o;
  if (Array.isArray(o)) { var a = new Array(o.length); for (var i = 0; i < o.length; i++) a[i] = clean(o[i]); return a; }
  if (ArrayBuffer.isView(o)) return Array.prototype.slice.call(o);
  if (o instanceof Set) return clean(Array.from(o));
  if (o instanceof Map) { var m = {}; o.forEach(function (v, k) { m[k] = clean(v); }); return m; }
  var out = {};
  for (var k in o) {
    if (k.length >= 5 && k.slice(-5) === '_path') continue;
    out[k] = clean(o[k]);
  }
  return out;
}

// numbers / strings / bools of a dict (quality, calibration) rounded for the dev log
function plain(d) {
  var out = {};
  for (var k in (d || {})) {
    var v = d[k];
    if (typeof v === 'number') out[k] = pyRound(v, 4);
    else if (typeof v === 'string' || typeof v === 'boolean') out[k] = v;
  }
  return out;
}

function r3(v, n) { return v === null || v === undefined || v !== v ? null : pyRound(+v, n === undefined ? 3 : n); }

// Python repr / str of the few value kinds the "why" texts print
function pr(v) {
  if (v === null || v === undefined) return 'None';
  if (typeof v === 'number') return Number.isInteger(v) ? v.toFixed(1) : String(v);
  if (typeof v === 'string') return "'" + v + "'";
  if (Array.isArray(v)) return '[' + v.map(pr).join(', ') + ']';
  return String(v);
}
function ps(v) { return typeof v === 'string' ? v : pr(v); }

function clip(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

function median32(vals) { return D.median32(vals); }

// glue.py::_row_trace: every intermediate value behind one question / ID digit / subject digit
function rowTrace(f, st, md, cal, strict, labels, frameF, trust) {
  var n = f.length, i, fmax = -Infinity;
  for (i = 0; i < n; i++) if (f[i] > fmax) fmax = f[i];
  var tIn = new Array(n), tHi32 = F32(cal.t_hi), tLo32 = F32(cal.t_lo);
  for (i = 0; i < n; i++) tIn[i] = strict ? cal.t_hi + D.DIGIT_MARGIN * clip((st[i] - D.DIGIT_MARGIN_FROM) / (D.DIGIT_MARGIN_FULL - D.DIGIT_MARGIN_FROM), 0.0, 1.0) : cal.t_hi;
  var relOut = strict ? 0.25 : 0.35;
  var ratio = new Array(n);
  for (i = 0; i < n; i++) ratio[i] = fmax > 0 ? F32(f[i] / fmax) : 0.0;
  var ro32 = F32(relOut), r07 = F32(0.7), isIn = [], isOut = [], nIn = 0, anyIn = false, minTin = Infinity;
  for (i = 0; i < n; i++) {
    var ge = strict ? f[i] >= tIn[i] : f[i] >= tHi32;
    if (ge) anyIn = true;
    isIn[i] = ge && ratio[i] >= r07 && st[i] >= cal.min_strength;
    isOut[i] = f[i] < tLo32 || ratio[i] < ro32;
    if (isIn[i]) nIn++;
    if (tIn[i] < minTin) minTin = tIn[i];
  }
  var ab = D.decideAbs(f, st, cal, strict), sAbs = ab[0], mAbs = ab[1], cAbs = ab[2], aAbs = ab[3], confident = ab[4];
  var re = D.relativeMarks(md == null ? null : md), rel = re[0], ex = re[1];
  if (md == null) { rel = []; ex = []; }
  var excess = null;
  if (md != null && md.length >= 3) {
    var nan = false;
    for (i = 0; i < md.length; i++) if (md[i] !== md[i]) nan = true;
    if (!nan) {
      excess = [];
      for (i = 0; i < md.length; i++) {
        var oth = [];
        for (var q = 0; q < md.length; q++) if (q !== i) oth.push(md[q]);
        excess.push(F32(md[i] - median32(oth)));
      }
    }
  }
  var ff = frameF && frameF.length ? frameF : [f];
  var ch = [];
  for (i = 0; i < n; i++) {
    var mn = Infinity, mx = -Infinity, s = 0, s2 = 0;
    for (var k = 0; k < ff.length; k++) { var v = ff[k][i]; if (v < mn) mn = v; if (v > mx) mx = v; s += v; }
    var mean = s / ff.length; for (k = 0; k < ff.length; k++) s2 += (ff[k][i] - mean) * (ff[k][i] - mean);
    ch.push({ label: labels ? labels[i] : String(i), fill_pct: pyRound(f[i] * 100, 1), strength: r3(st[i]), disc_dark: r3(md == null ? null : md[i]),
      excess_vs_others: r3(excess == null ? null : excess[i]), ratio_to_max: r3(ratio[i]), t_in: r3(tIn[i]),
      abs_class: isIn[i] ? 'in' : isOut[i] ? 'out' : 'grey',
      fill_pct_min: pyRound(mn * 100, 1), fill_pct_max: pyRound(mx * 100, 1), fill_pct_std: pyRound(Math.sqrt(s2 / ff.length) * 100, 1) });
  }
  var why = [];
  if (!anyIn) {
    why.push('no choice reaches t_in (' + pr(r3(minTin)) + '); fullest fill ' + pr(r3(fmax)) + ' ' +
      (fmax < cal.t_lo ? '< t_lo -> blank' : fmax >= 0.5 * (cal.t_lo + cal.t_hi) ? '>= (t_lo+t_hi)/2 -> weak single guess, flagged' : '-> blank (but unsure)'));
  } else {
    for (i = 0; i < ch.length; i++) {
      var c = ch[i];
      if (c.abs_class === 'grey') why.push(c.label + ': grey zone (fill ' + pr(c.fill_pct) + '%, ratio ' + pr(c.ratio_to_max) + ', strength ' + pr(c.strength) + ') -> doubt');
    }
    if (nIn >= 2) why.push('two or more choices count as marked -> multi');
  }
  if (!confident && rel.length) {
    why.push('relative rule: ' + pr(rel.map(function (i) { return ch[i].label; })) + ' darker than the other choices by ' + pr(Array.prototype.map.call(ex, function (x) { return r3(x); })) +
      ' (needs >= ' + pr(D.REL_MARGIN) + '; silent from ' + pr(strict ? D.REL_SURE_DIGIT : D.REL_SURE) + '); rule ' + (trust ? 'ON' : 'OFF (region not trusted)'));
  }
  return { choices: ch, abs: { state: sAbs, marked: mAbs.slice(), confidence_pct: pyRound(cAbs * 100), ambiguous: !!aAbs, confident: !!confident },
    rel: { indices: rel.slice(), excess: Array.prototype.map.call(ex, function (x) { return r3(x); }), trusted: !!trust }, fmax_pct: pyRound(fmax * 100, 1), why: why };
}

// glue.py::explain: the decision trace of a committed sheet, from the fused measurements of its track
function explain(result) {
  var src = result.trace_src || (result.raw ? [result.raw, [result.raw]] : null);
  if (!src) return {};
  var raw = src[0], frames = src[1], i;
  var cal = D.calibrate(raw.FA, raw.ink, null, raw.min_strength);
  var preA = D.preRows(raw.FA, raw.SA, raw.MA, cal, false);
  var tp = D.relTrustAnswers(raw, preA), trust = tp[0], pageTrust = tp[1];
  var FD = raw.FS == null ? raw.FI : raw.FI.concat(raw.FS);
  var ical = D.calibrate(FD, raw.ink, cal.typ, raw.min_strength);
  var SD = raw.SS == null ? raw.SI : raw.SI.concat(raw.SS);
  var MD = raw.MI == null ? null : (raw.MS == null ? raw.MI : raw.MI.concat(raw.MS));
  var preD = D.preRows(FD, SD, MD, ical, true);
  var dtrust = pageTrust || D.relTrustDigits(FD, MD, preD);
  var out = { page: { frames_fused: frames.length, ink: r3(raw.ink), ink_source: raw.ink_source, ink_separation: r3(raw.ink_separation), min_strength: r3(raw.min_strength),
      typ_fill: r3(cal.typ), t_lo: r3(cal.t_lo), t_hi: r3(cal.t_hi), id_typ_fill: r3(ical.typ), id_t_lo: r3(ical.t_lo), id_t_hi: r3(ical.t_hi),
      alignment_verified: !!raw.verified, relative_rule_page_trust: !!pageTrust, relative_rule_digit_trust: !!dtrust,
      consts: { REL_MARGIN: D.REL_MARGIN, REL_SURE: D.REL_SURE, REL_SURE_DIGIT: D.REL_SURE_DIGIT, GUESS_GAP: D.GUESS_GAP, DIGIT_MARGIN: D.DIGIT_MARGIN,
        T_HI_OF_TYP: D.T_HI_OF_TYP, T_LO_FRAC: D.T_LO_FRAC } },
    answers: [], id: [], subject: [] };
  function col(name, j) { return frames.map(function (x) { return x[name][j]; }); }
  for (i = 0; i < result.answers.length; i++) {
    if (i >= raw.FA.length) break;
    var a = result.answers[i];
    var t = rowTrace(raw.FA[i], raw.SA[i], raw.MA == null ? null : raw.MA[i], cal, false, a.labels.slice(), col('FA', i), trust[i]);
    t.q = a.number;
    t.final = { state: a.state, marked: a.marked.slice(), confidence_pct: pyRound(a.confidence * 100), ambiguous: !!a.ambiguous, guessed: !!a.guessed, guess: a.guess };
    out.answers.push(t);
  }
  var sets = [['id', result.id_digits, raw.FI, raw.SI, raw.MI, 'FI'], ['subject', result.subject_digits, raw.FS, raw.SS, raw.MS, 'FS']];
  for (var si = 0; si < sets.length; si++) {
    var key = sets[si][0], digits = sets[si][1], F = sets[si][2], S = sets[si][3], M = sets[si][4], frKey = sets[si][5];
    if (F == null) continue;
    for (i = 0; i < digits.length; i++) {
      if (i >= F.length) break;
      var d = digits[i];
      var t2 = rowTrace(F[i], S[i], M == null ? null : M[i], ical, true, null, col(frKey, i), dtrust);
      t2.digit = i + 1;
      t2.final = { state: d.state, value: d.value, confidence_pct: pyRound(d.confidence * 100), ambiguous: !!d.ambiguous, guessed: !!d.guessed };
      out[key].push(t2);
    }
  }
  return out;
}

// same fields as app/main.py::summarize, plus the marked labels per question
function summarize(result) {
  var nb = 0, nm = 0, nd = 0;
  result.answers.forEach(function (a) { if (a.state === 'blank') nb++; if (a.state === 'multi') nm++; if (a.ambiguous) nd++; });
  return { form_key: result.form_key, page: result.page, total_pages: result.total_pages,
    student_id: result.student_id, subject: result.subject, serial: result.serial,
    n_questions: result.answers.length, n_blank: nb, n_multi: nm, n_doubtful: nd,
    flags: result.flags.slice(), needs_review: T.needsReview(result),
    sharpness: (result.quality || {}).sharpness === undefined ? null : result.quality.sharpness, severity: O.identity.severity(result),
    answers: result.answers.map(function (a) {
      return { q: a.number, state: a.state, marked: a.marked.map(function (i) { return a.labels[i]; }).join('|'), doubt: !!a.ambiguous };
    }) };
}

function digitRows(digits, centers) {
  var out = [];
  for (var i = 0; i < digits.length; i++) {
    if (i >= centers.length) continue;
    var d = digits[i];
    out.push({ state: d.state, value: d.value, doubt: !!d.ambiguous, guessed: !!d.guessed, scores: d.scores, centers: centers[i] });
  }
  return out;
}

// what the review view needs to show where a sheet is doubtful: the rectified sheet as a JPEG plus, per question / ID digit /
// subject digit, the bubble centres (mm), fill scores and the decision
function detail(result) {
  if (!result.warped) return {};
  var jpg = O.scanner.encodeJpegGray(result.warped, 75);
  var bc = result.bubble_centers_mm || {}, cal = result.calibration || {}, ans = bc.answers || [];
  var rows = [];
  for (var i = 0; i < result.answers.length; i++) {
    if (i >= ans.length) continue;
    var a = result.answers[i];
    rows.push({ q: a.number, state: a.state, marked: a.marked.slice(), labels: a.labels.slice(), scores: a.scores, doubt: !!a.ambiguous, guessed: !!a.guessed, centers: ans[i] });
  }
  return { img: b64(jpg), ppm: result.px_per_mm, t_lo: cal.t_lo === undefined ? null : cal.t_lo, t_hi: cal.t_hi === undefined ? null : cal.t_hi,
    id_t_lo: cal.id_t_lo === undefined ? null : cal.id_t_lo, id_t_hi: cal.id_t_hi === undefined ? null : cal.id_t_hi,
    answers: rows, id: digitRows(result.id_digits, bc.id || []), subject: digitRows(result.subject_digits, bc.subject || []) };
}

var B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
function b64(u8) {
  var out = [], n = u8.length, i = 0;
  for (; i + 2 < n; i += 3) {
    var v = (u8[i] << 16) | (u8[i + 1] << 8) | u8[i + 2];
    out.push(B64[v >> 18] + B64[(v >> 12) & 63] + B64[(v >> 6) & 63] + B64[v & 63]);
  }
  if (n - i === 1) { v = u8[i] << 16; out.push(B64[v >> 18] + B64[(v >> 12) & 63] + '=='); }
  else if (n - i === 2) { v = (u8[i] << 16) | (u8[i + 1] << 8); out.push(B64[v >> 18] + B64[(v >> 12) & 63] + B64[(v >> 6) & 63] + '='); }
  return out.join('');
}

// ---- baseline grey JPEG encoder (JFIF, 1 component, standard Huffman tables; cv2.imencode(".jpg", grey, quality) equivalent)
var ZZ = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63];
var QL = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99];
var DC_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], DC_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
var AC_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
var AC_VALS = [0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
  0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
  0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
  0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
  0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
  0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
  0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa];

function huffCodes(bits, vals) {      // canonical codes -> {code[], size[]} indexed by symbol
  var code = new Int32Array(256), size = new Uint8Array(256), c = 0, k = 0;
  for (var l = 1; l <= 16; l++) { for (var i = 0; i < bits[l - 1]; i++) { code[vals[k]] = c; size[vals[k]] = l; c++; k++; } c <<= 1; }
  return { code: code, size: size };
}
var DCH = huffCodes(DC_BITS, DC_VALS), ACH = huffCodes(AC_BITS, AC_VALS);
var DCT_M = (function () {     // 8x8 DCT-II matrix
  var m = new Float64Array(64);
  for (var u = 0; u < 8; u++) for (var x = 0; x < 8; x++) m[u * 8 + x] = (u === 0 ? Math.SQRT1_2 : 1) * 0.5 * Math.cos((2 * x + 1) * u * Math.PI / 16);
  return m;
})();

function encodeJpegGray(img, quality) {
  var w = img.w, h = img.h, data = img.data, q = quality === undefined ? 75 : quality;
  var scale = q < 50 ? 5000 / q : 200 - q * 2, qt = new Int32Array(64), i, j;
  for (i = 0; i < 64; i++) qt[i] = clip(Math.floor((QL[i] * scale + 50) / 100), 1, 255);
  var out = [], bitBuf = 0, bitCnt = 0, ob = new Uint8Array(1 << 16), on = 0;
  function byte(b) { out.push(b); }
  function word(v) { out.push(v >> 8, v & 255); }
  function put(code, size) {
    bitBuf = (bitBuf << size) | code; bitCnt += size;
    if (on + 8 > ob.length) { var nb2 = new Uint8Array(ob.length * 2); nb2.set(ob); ob = nb2; }
    while (bitCnt >= 8) { var b = (bitBuf >> (bitCnt - 8)) & 255; ob[on++] = b; if (b === 255) ob[on++] = 0; bitCnt -= 8; }
    bitBuf &= (1 << bitCnt) - 1;
  }
  word(0xffd8);
  word(0xffe0); word(16); byte(0x4a); byte(0x46); byte(0x49); byte(0x46); byte(0); byte(1); byte(1); byte(0); word(1); word(1); byte(0); byte(0);
  word(0xffdb); word(67); byte(0); for (i = 0; i < 64; i++) byte(qt[ZZ[i]]);
  word(0xffc0); word(11); byte(8); word(h); word(w); byte(1); byte(1); byte(0x11); byte(0);
  word(0xffc4); word(19 + DC_VALS.length); byte(0x00); for (i = 0; i < 16; i++) byte(DC_BITS[i]); for (i = 0; i < DC_VALS.length; i++) byte(DC_VALS[i]);
  word(0xffc4); word(19 + AC_VALS.length); byte(0x10); for (i = 0; i < 16; i++) byte(AC_BITS[i]); for (i = 0; i < AC_VALS.length; i++) byte(AC_VALS[i]);
  word(0xffda); word(8); byte(1); byte(1); byte(0); byte(0); byte(63); byte(0);
  var blk = new Float64Array(64), tmp = new Float64Array(64), zq = new Int32Array(64), pred = 0, M = DCT_M;
  var rq = new Float64Array(64); for (i = 0; i < 64; i++) rq[i] = qt[i];
  var dcc = DCH.code, dcs = DCH.size, acc = ACH.code, acs = ACH.size;
  function nbits(v) { return 32 - Math.clz32(v); }
  for (var by = 0; by < h; by += 8) for (var bx = 0; bx < w; bx += 8) {
    if (bx + 8 <= w && by + 8 <= h) {
      for (j = 0; j < 8; j++) { var yy = (by + j) * w + bx, jb = j * 8; for (i = 0; i < 8; i++) blk[jb + i] = data[yy + i] - 128; }
    } else for (j = 0; j < 8; j++) { var yy2 = Math.min(by + j, h - 1) * w; for (i = 0; i < 8; i++) blk[j * 8 + i] = data[yy2 + Math.min(bx + i, w - 1)] - 128; }
    for (j = 0; j < 8; j++) {      // rows (same summation order as the plain loop: bit-identical output)
      var jb2 = j * 8, b0 = blk[jb2], b1 = blk[jb2 + 1], b2 = blk[jb2 + 2], b3 = blk[jb2 + 3], b4 = blk[jb2 + 4], b5 = blk[jb2 + 5], b6 = blk[jb2 + 6], b7 = blk[jb2 + 7];
      for (var u = 0; u < 8; u++) {
        var ub = u * 8;
        tmp[jb2 + u] = ((((((M[ub] * b0 + M[ub + 1] * b1) + M[ub + 2] * b2) + M[ub + 3] * b3) + M[ub + 4] * b4) + M[ub + 5] * b5) + M[ub + 6] * b6) + M[ub + 7] * b7;
      }
    }
    for (u = 0; u < 8; u++) {      // columns
      var t0_ = tmp[u], t1 = tmp[8 + u], t2 = tmp[16 + u], t3 = tmp[24 + u], t4 = tmp[32 + u], t5 = tmp[40 + u], t6 = tmp[48 + u], t7 = tmp[56 + u];
      for (var v = 0; v < 8; v++) {
        var vb = v * 8, k = vb + u;
        var s = ((((((M[vb] * t0_ + M[vb + 1] * t1) + M[vb + 2] * t2) + M[vb + 3] * t3) + M[vb + 4] * t4) + M[vb + 5] * t5) + M[vb + 6] * t6) + M[vb + 7] * t7;
        zq[k] = Math.round(s / rq[k]);
      }
    }
    var dc = zq[0], diff = dc - pred; pred = dc;
    var a = diff < 0 ? -diff : diff, nb = nbits(a);
    put(dcc[nb], dcs[nb]); if (nb) put(diff < 0 ? (diff - 1) & ((1 << nb) - 1) : diff, nb);
    var run = 0;
    for (i = 1; i < 64; i++) {
      var co = zq[ZZ[i]];
      if (co === 0) { run++; continue; }
      while (run > 15) { put(acc[0xf0], acs[0xf0]); run -= 16; }
      a = co < 0 ? -co : co; nb = nbits(a);
      var sym = (run << 4) | nb;
      put(acc[sym], acs[sym]); put(co < 0 ? (co - 1) & ((1 << nb) - 1) : co, nb);
      run = 0;
    }
    if (run) put(acc[0], acs[0]);
  }
  if (bitCnt) put((1 << (8 - bitCnt)) - 1, 8 - bitCnt);
  var hdrLen = out.length, res = new Uint8Array(hdrLen + on + 2);
  res.set(out); res.set(ob.subarray(0, on), hdrLen); res[hdrLen + on] = 0xff; res[hdrLen + on + 1] = 0xd9;
  return res;
}

// ---- the scanner
function Scanner() {
  this.session = new O.stream.StreamSession({});
  this._convertMs = 0.0;
}
var P = Scanner.prototype;

P.reset = function () { this.session.reset(); this.session.resetTracking(); };

// buf: RGBA bytes (Uint8Array / Uint8ClampedArray) of a w x h frame.  Returns the event as a JSON string.
// RGBA -> grey exactly as cv2.cvtColor(COLOR_RGBA2GRAY) (cv2 5.0: 15-bit fixed point, R 9798, G 19235, B 3735, round 2^14).
function rgbaToGray(buf, w, h) {
  var n = w * h, g = new Uint8Array(n), j = 0;
  for (var i = 0; i < n; i++, j += 4) g[i] = (buf[j] * 9798 + buf[j + 1] * 19235 + buf[j + 2] * 3735 + 16384) >> 15;
  return { w: w, h: h, data: g };
}

P.stepRgba = function (buf, w, h) {
  var t0 = now(), img = rgbaToGray(buf, w, h);
  this._convertMs = now() - t0;
  return JSON.stringify(this.step(img, t0));
};

P.step = function (img, t0) {
  var session = this.session;
  if (t0 === undefined || t0 === null) t0 = now();
  if (Math.min(img.w, img.h) < MIN_FRAME_SIDE) {
    return { state: session.state, hint: 'Camera frame too small (' + img.w + 'x' + img.h + ')', overlay: {}, error: 'frame too small: ' + img.w + 'x' + img.h, w: img.w, h: img.h };
  }
  var res = O.recognize.recognize(img, { resolveLayout: null });
  var tFeed = now();
  var ev = session.feed(res, O.scanner.clock ? O.scanner.clock() : now() / 1000, null);   // clock: test hook (deterministic stream time)
  tFeed = now() - tFeed;
  ev.status = res.status;
  ev.w = img.w; ev.h = img.h;
  ['commit', 'revision'].forEach(function (name) {     // a revision = a better reading of the sheet committed before (omr/revision.py)
    var c = ev[name];
    delete ev[name];
    if (c === undefined || c === null) return;
    var o = { summary: summarize(c.result), frames: c.frames, detail: detail(c.result) };
    ev[name] = o;
    if (name === 'revision') { o.changed = c.changed; o.contradiction = c.contradiction; o.prev_severity = c.prev_severity; o.n = c.revision; }
    if (DEV) {
      var cr = c.result;
      o.dev = { quality: plain(cr.quality), calibration: plain(cr.calibration), fused_frames: cr.fused_frames, frame_index: c.frame_index === undefined ? null : c.frame_index,
        marker_ids: cr.marker_ids.slice(), missing_corners: cr.missing_corners.slice(), trace: explain(cr) };
    }
  });
  if (DEV) {
    var tm = {}, tms = res.timing_ms || {};
    for (var k in tms) tm[k] = pyRound(tms[k], 1);
    ev.dev = { quality: plain(res.quality), flags: res.flags.slice(), markers: res.marker_ids.slice(), missing: res.missing_corners.slice(), form_key: res.form_key, page: res.page, timing: tm };
  }
  var missed = session.missed();
  if (missed && missed.length) ev.missed = missed;
  ev.ms = { total: pyRound(now() - t0, 1), recognize: pyRound((res.timing_ms || {}).total || 0, 1) };
  if (DEV) { ev.ms.convert = pyRound(this._convertMs, 2); ev.ms.feed = pyRound(tFeed, 1); }
  this._convertMs = 0.0;
  return clean(ev);
};

var _scanner = null;
O.scanner = { clock: null, rgbaToGray: rgbaToGray, Scanner: Scanner, summarize: summarize, detail: detail, explain: explain, clean: clean, encodeJpegGray: encodeJpegGray, b64: b64,
  init: function () { _scanner = new Scanner(); }, reset: function () { _scanner.reset(); },
  frame: function (buf, w, h) { return _scanner.stepRgba(buf, w, h); } };
})(globalThis.OMR = globalThis.OMR || {});
