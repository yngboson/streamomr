// The decision half of omr/recognize.py (lines 741-1155): bubble measurements (PageRaw) -> decisions (PageDecision).  Used per frame
// by recognize() and on track-fused measurements by fusion.js.  Pure arithmetic, no image code.
// numpy 2 (NEP 50) keeps float32 arrays float32 when they meet Python floats, so every float32 step here is rounded with
// Math.fround to give the same numbers (and the same threshold decisions) as Python; float64 steps stay plain doubles.
(function (O) {
'use strict';
var T = O.types, pyRound = O.util.pyRound, F32 = Math.fround;

var MIN_STRENGTH = 0.6, DEFAULT_TYP_FILL = 0.9;
var T_HI_MIN = 0.45, T_HI_MAX = 0.65, T_HI_OF_TYP = 0.65, T_LO_FRAC = 0.2;
var DIGIT_MARGIN = 0.08, DIGIT_MARGIN_FROM = 0.7, DIGIT_MARGIN_FULL = 1.0;
var REL_MARGIN = 0.15, REL_SURE = 0.20, REL_SURE_DIGIT = 0.20;
var GUESS_GAP = 0.15, GUESS_REL_GAP = 0.10;
var REL_TRUST_ROWS = 0.4, REL_TRUST_GHOSTS = 0.02, REL_TRUST_WINDOW = 7, REL_STRONG = 0.30, REL_TRUST_MIN_DIGITS = 3;
var UNRELIABLE_FILL = T.UNRELIABLE_FILL, UNRELIABLE_STRENGTH = T.UNRELIABLE_STRENGTH;

var f_015 = F32(REL_MARGIN), f_03 = F32(0.3), f_07 = F32(0.7), f_055 = F32(0.55), f_08 = F32(0.8), f_010 = F32(GUESS_REL_GAP),
  f_strong = F32(REL_STRONG), f_sure = F32(REL_SURE), f_sureD = F32(REL_SURE_DIGIT);

function clip(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

// median of float32 values, as np.median returns it for float32 (even count: mean of the two middle values in float32); NaN -> NaN
function median32(vals) {
  var n = vals.length, a = new Float32Array(vals);
  for (var i = 0; i < n; i++) if (a[i] !== a[i]) return NaN;
  a.sort();
  if (n % 2) return a[(n - 1) / 2];
  return F32(F32(a[n / 2 - 1] + a[n / 2]) / 2);
}

function unreliableMask(f, strength) {   // omr/recognize.py:770
  var out = new Array(f.length);
  for (var i = 0; i < f.length; i++) out[i] = f[i] === UNRELIABLE_FILL && strength[i] === UNRELIABLE_STRENGTH;
  return out;
}

// omr/recognize.py:775: best-guess choice of a row with no mark reaching t_in
function weakPick(f, strength, tLo, need) {
  need = need || 0.0;
  var bad = unreliableMask(f, strength), nbad = 0, anyOk = false, i, tl = F32(tLo);
  for (i = 0; i < f.length; i++) { if (bad[i]) nbad++; else anyOk = true; }
  if (nbad === 1 && anyOk) {
    var allOut = true;
    for (i = 0; i < f.length; i++) if (!bad[i] && !(f[i] < tl)) allOut = false;
    if (allOut) return [bad.indexOf(true)];
  }
  if (!anyOk) return [];
  var top = -Infinity;
  for (i = 0; i < f.length; i++) if (!bad[i] && f[i] > top) top = f[i];
  var thr = F32(top - 1e-6), best = [];
  for (i = 0; i < f.length; i++) if (!bad[i] && f[i] >= thr) best.push(i);
  return best.length === 1 && top >= need ? [best[0]] : [];
}

// omr/recognize.py:878: page-level thresholds from the fills of all bubbles (rows: questions x choices)
function calibrate(Frows, ink, defaultTyp, minStrength) {
  var hot = [];
  for (var i = 0; i < Frows.length; i++) {
    var m = -Infinity, r = Frows[i];
    for (var j = 0; j < r.length; j++) if (r[j] > m) m = r[j];
    if (m >= f_055) hot.push(m);
  }
  var typ = hot.length >= 5 ? median32(hot) : (defaultTyp != null ? defaultTyp : DEFAULT_TYP_FILL);
  var tHi = clip(T_HI_OF_TYP * typ, T_HI_MIN, T_HI_MAX);
  return { ink: ink, typ: typ, t_hi: tHi, t_lo: T_LO_FRAC * tHi, min_strength: minStrength === undefined ? MIN_STRENGTH : minStrength };
}

// omr/recognize.py:913: one question (or ID column) from its bubbles' fills -> [state, marked, confidence, ambiguous, confident]
function decideAbs(f, strength, c, strict) {
  var n = f.length, i, fmax = -Infinity;
  for (i = 0; i < n; i++) if (f[i] > fmax) fmax = f[i];
  var relOut = strict ? 0.25 : 0.35, tIn = null, tHi32 = F32(c.t_hi), tLo32 = F32(c.t_lo);
  if (strict) {      // per bubble, float64: the margin grows with strength
    tIn = new Float64Array(n);
    for (i = 0; i < n; i++) tIn[i] = c.t_hi + DIGIT_MARGIN * clip((strength[i] - DIGIT_MARGIN_FROM) / (DIGIT_MARGIN_FULL - DIGIT_MARGIN_FROM), 0.0, 1.0);
  }
  var fin = new Array(n), anyIn = false;
  for (i = 0; i < n; i++) { fin[i] = strict ? f[i] >= tIn[i] : f[i] >= tHi32; if (fin[i]) anyIn = true; }
  if (!anyIn) {
    if (fmax < c.t_lo) return ['blank', [], Math.min(1.0, 0.5 + 0.5 * (c.t_lo - fmax) / c.t_lo), false, false];
    var half = 0.5 * (c.t_lo + c.t_hi);
    var guess = fmax >= half ? weakPick(f, strength, c.t_lo, half) : [];
    return [guess.length ? 'single' : 'blank', guess, 0.2, true, false];
  }
  var r = new Float32Array(n), isIn = new Array(n), grey = false, marked = [], rest = -Infinity, nrest = 0;
  var r07 = f_07, ro32 = F32(relOut);
  for (i = 0; i < n; i++) {
    r[i] = f[i] / fmax;
    isIn[i] = fin[i] && r[i] >= r07 && strength[i] >= c.min_strength;
    var isOut = f[i] < tLo32 || r[i] < ro32;
    if (!isIn[i] && !isOut) grey = true;
    if (isIn[i]) marked.push(i); else { nrest++; if (f[i] > rest) rest = f[i]; }
  }
  if (!marked.length) {     // filled but weak (smudge / ghost): best guess, always flagged
    var pick = weakPick(f, strength, c.t_lo);
    return [pick.length ? 'single' : 'blank', pick, 0.2, true, false];
  }
  // confidence: how far the weakest "in" bubble is above the threshold, and how small the strongest competing bubble is
  var confIn, mn = Infinity;
  if (strict) {
    for (i = 0; i < n; i++) if (isIn[i]) mn = Math.min(mn, f[i] - tIn[i]);
    confIn = clip(0.5 + mn / 0.3, 0.0, 1.0);
  } else {
    for (i = 0; i < n; i++) if (isIn[i]) mn = Math.min(mn, F32(f[i] - tHi32));
    confIn = clip(F32(0.5 + F32(mn / f_03)), 0.0, 1.0);        // float32 arithmetic (np.float32 scalar)
  }
  var confOut = nrest ? clip(F32(1.0 - F32(F32(rest / fmax) / 0.5)), 0.0, 1.0) : 1.0;
  var conf = Math.min(confIn, confOut);
  return [marked.length >= 2 ? 'multi' : 'single', marked, conf, grey, true];
}

// omr/recognize.py:850: _relative_marks for every row of a mean-darkness matrix -> per row [indices, excess Float32Array]
function relRows(M) {
  if (M == null) return null;
  var n = M.length, out = new Array(n), k = n ? M[0].length : 0, i, j;
  var none = [[], new Float32Array(0)];
  for (i = 0; i < n; i++) out[i] = none;
  if (k < 3) return out;
  for (i = 0; i < n; i++) {
    var row = M[i], nan = false;
    for (j = 0; j < k; j++) if (row[j] !== row[j]) nan = true;
    if (nan) continue;
    var idx = [], ex = [];
    for (j = 0; j < k; j++) {
      var oth = [];
      for (var q = 0; q < k; q++) if (q !== j) oth.push(row[q]);
      var e = F32(row[j] - median32(oth));
      if (e >= f_015) { idx.push(j); ex.push(e); }
    }
    if (idx.length) out[i] = [idx, new Float32Array(ex)];
  }
  return out;
}

// omr/recognize.py:1061: absolute decision + relative marks of every row, computed once
function preRows(F, S, M, cal, strict) {
  var rel = relRows(M) || [], out = [];
  for (var i = 0; i < F.length; i++) out.push([decideAbs(F[i], S[i], cal, strict), i < rel.length ? rel[i] : [[], new Float32Array(0)]]);
  return out;
}

function minOf(a) { var m = Infinity; for (var i = 0; i < a.length; i++) if (a[i] < m) m = a[i]; return m; }

// omr/recognize.py:887: absolute rules plus the relative rule for rows the absolute rules cannot call with confidence
function decideRow(md, trustRel, pre, strict) {
  var a = pre[0], state = a[0], marked = a[1], conf = a[2], amb = a[3], confident = a[4];
  if (md == null || !trustRel) return [state, marked, conf, amb];
  if (confident) {
    if (marked.length && !amb && marked.some(function (i) { return md[i] === 0.0; })) return [state, marked, conf, true];
    return [state, marked, conf, amb];
  }
  var rel = pre[1][0], ex = pre[1][1];
  if (!rel.length) return [state, marked, conf, amb];
  var exMin = minOf(ex), sure = exMin >= (strict ? f_sureD : f_sure);
  if (!marked.every(function (i) { return rel.indexOf(i) >= 0; })) return [state, marked, conf, amb];
  var rconf = Math.min(Math.max(F32(0.5 + F32(F32(exMin - f_015) / f_03)), 0.0), f_08);
  return [rel.length >= 2 ? 'multi' : 'single', rel, rconf, !sure];
}

// omr/recognize.py:992: most plausible choice of an UNCERTAIN row, or null when no choice stands out
function guess(f, strength, c, md) {
  var bad = unreliableMask(f, strength), cand = [], i;
  for (i = 0; i < f.length; i++) if (!bad[i]) cand.push(i);
  if (f.length >= 2 && cand.length >= 1) {
    cand.sort(function (a, b) { return f[b] - f[a]; });
    var top = f[cand[0]], second = cand.length > 1 ? f[cand[1]] : 0.0;
    if (top >= c.t_lo && top - second >= GUESS_GAP) return cand[0];
  }
  if (md != null && md.length >= 3) {
    var nan = false;
    for (i = 0; i < md.length; i++) if (md[i] !== md[i]) nan = true;
    if (!nan) {
      var ex = [], o = [];
      for (i = 0; i < md.length; i++) {
        var oth = [];
        for (var q = 0; q < md.length; q++) if (q !== i) oth.push(md[q]);
        ex.push(F32(md[i] - median32(oth)));
        o.push(i);
      }
      o.sort(function (a, b) { return ex[b] - ex[a]; });
      if (ex[o[0]] >= f_015 && F32(ex[o[0]] - ex[o[1]]) >= f_010) return o[0];
    }
  }
  return null;
}

function roundList(f, n) { var out = []; for (var i = 0; i < f.length; i++) out.push(pyRound(f[i], n)); return out; }

// omr/recognize.py:1012: digit columns (ID, subject code): strict rules -> [digits, value string or null, why-not-read or null, any guessed]
function readDigits(F, S, cal, verified, M, trustRel, pre) {
  var digits = [], di;
  for (di = 0; di < F.length; di++) {
    var f = F[di], st = S[di], md = M == null ? null : M[di];
    var d = decideRow(md, trustRel, pre[di], true), state = d[0], marked = d[1], conf = d[2], amb = d[3];
    var val = null, guessed = false;
    if (state === 'single' && !amb && verified) val = marked[0];
    else if (amb || !verified) {
      var g = guess(f, st, cal, md);
      if (g !== null && (state !== 'blank' || amb)) { val = g; guessed = true; amb = true; }
    }
    digits.push(T.DigitResult(di, state, val, pyRound(conf, 3), amb, roundList(f, 4), guessed));
  }
  var anyGuess = digits.some(function (d) { return d.guessed; });
  if (digits.every(function (d) { return d.value !== null; })) return [digits, digits.map(function (d) { return String(d.value); }).join(''), null, anyGuess];
  return [digits, null, digits.every(function (d) { return d.state === 'blank' && !d.ambiguous; }) ? 'blank' : 'unreadable', anyGuess];
}

// omr/recognize.py:1044: per row (relative-only mark, the same with excess >= REL_STRONG, ghost-like other choices beside a confident mark)
function relEvidence(n, pre) {
  var relOnly = new Array(n).fill(false), strong = new Array(n).fill(false), ghosts = new Array(n).fill(0);
  for (var i = 0; i < n; i++) {
    var a = pre[i][0], rel = pre[i][1][0], ex = pre[i][1][1];
    if (a[4]) ghosts[i] = rel.filter(function (j) { return a[1].indexOf(j) < 0; }).length;
    else if (rel.length) { relOnly[i] = true; strong[i] = minOf(ex) >= f_strong; }
  }
  return [relOnly, strong, ghosts];
}

function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += +a[i]; return s; }

// omr/recognize.py:1070: per question, does the page or ITS REGION say that relative-only marks are marks rather than ghosts?
function relTrustAnswers(raw, pre) {
  var n = raw.numbers.length;
  if (raw.MA == null || !n) return [new Array(n).fill(false), false];
  var ev = relEvidence(n, pre), relOnly = ev[0], strong = ev[1], ghosts = ev[2];
  if (sum(relOnly) >= REL_TRUST_ROWS * n && sum(ghosts) <= REL_TRUST_GHOSTS * n) return [new Array(n).fill(true), true];
  var out = new Array(n).fill(false);
  for (var i = 0; i < n; i++) {
    var lo = Math.max(0, i - REL_TRUST_WINDOW), hi = Math.min(n, i + REL_TRUST_WINDOW + 1), m = hi - lo;
    out[i] = sum(strong.slice(lo, hi)) >= REL_TRUST_ROWS * m && sum(ghosts.slice(lo, hi)) <= REL_TRUST_GHOSTS * m;
  }
  return [out, false];
}

// omr/recognize.py:1094: the same for the ID + subject block as one region (strict digit rules)
function relTrustDigits(F, M, pre) {
  if (M == null || !F.length) return false;
  var ev = relEvidence(F.length, pre), n = F.length;
  return sum(ev[1]) >= Math.max(REL_TRUST_MIN_DIGITS, REL_TRUST_ROWS * n) && sum(ev[2]) <= REL_TRUST_GHOSTS * n;
}

// omr/recognize.py:1104: comparable signature (a JSON string here; Python: a tuple)
function decisionSig(answers, studentId, subject) {
  return JSON.stringify([answers.map(function (a) { return [a.number, a.state, a.marked, a.guess]; }), studentId, subject]);
}

function concat(a, b) { return b == null ? a : a.concat(Array.prototype.slice.call(b)); }

// omr/recognize.py:1108: the one place where bubble measurements become decisions
// -> {answers, id_digits, student_id, flags, calibration, subject_digits, subject}
function decidePage(raw) {
  var cal = calibrate(raw.FA, raw.ink, null, raw.min_strength);
  var answers = [];
  var preA = preRows(raw.FA, raw.SA, raw.MA, cal, false);
  var tp = relTrustAnswers(raw, preA), trust = tp[0], pageTrust = tp[1], i;
  for (i = 0; i < raw.numbers.length; i++) {
    var md = raw.MA != null ? raw.MA[i] : null, f = raw.FA[i];
    var d = decideRow(md, !!trust[i], preA[i], false), state = d[0], marked = d[1], conf = d[2], amb = !!(d[3] || !raw.verified);
    var g = null;
    if (amb) {       // an uncertain row keeps its flag AND gets the most plausible value (a clear blank / confident row: none)
      if (state === 'single') g = marked[0];
      else if (state === 'blank') g = guess(f, raw.SA[i], cal, md);     // a multi-mark row gets NO guess
    }
    answers.push(T.QuestionResult(raw.numbers[i], state, marked, raw.labels[i].slice(), pyRound(conf, 3), amb, roundList(f, 4), g, g !== null));
  }
  var FD = concat(raw.FI.slice(), raw.FS), SD = concat(raw.SI.slice(), raw.SS);
  var MD = raw.MI == null ? null : concat(raw.MI.slice(), raw.MS);
  if (raw.MI != null && raw.MS == null) MD = raw.MI;
  var ical = calibrate(FD, raw.ink, cal.typ, raw.min_strength);
  var preD = preRows(FD, SD, MD, ical, true), nI = raw.FI.length;
  var dtrust = pageTrust || relTrustDigits(FD, MD, preD);
  var rd = readDigits(raw.FI, raw.SI, ical, raw.verified, raw.MI, dtrust, preD.slice(0, nI));
  var digits = rd[0], studentId = rd[1], why = rd[2], gs = rd[3];
  for (i = 0; i < digits.length; i++) digits[i].index = raw.digit_index[i];
  var flags = why === null ? [] : ['id_' + why];
  if (gs && studentId !== null) flags.push('id_guessed');
  var sdigits = [], subject = '';
  if (raw.FS != null && raw.FS.length) {
    rd = readDigits(raw.FS, raw.SS, ical, raw.verified, raw.MS, dtrust, preD.slice(nI));
    sdigits = rd[0]; subject = rd[1];
    if (rd[2] !== null) flags.push('subject_' + rd[2]);
    if (rd[3] && subject !== null) flags.push('subject_guessed');
  }
  if (answers.some(function (a) { return a.guessed; })) flags.push('guessed_answers');
  var calib = { ink: cal.ink, typ_fill: cal.typ, t_lo: cal.t_lo, t_hi: cal.t_hi, id_typ_fill: ical.typ, id_t_lo: ical.t_lo, id_t_hi: ical.t_hi,
    ink_auto: raw.ink_source === 'auto' ? 1.0 : 0.0, ink_source: raw.ink_source, ink_separation: raw.ink_separation };
  return { answers: answers, id_digits: digits, student_id: studentId, flags: flags, calibration: calib, subject_digits: sdigits, subject: subject };
}

O.decide = { MIN_STRENGTH: MIN_STRENGTH, T_HI_MIN: T_HI_MIN, T_HI_MAX: T_HI_MAX, T_HI_OF_TYP: T_HI_OF_TYP, T_LO_FRAC: T_LO_FRAC,
  DIGIT_MARGIN: DIGIT_MARGIN, REL_MARGIN: REL_MARGIN, median32: median32, unreliableMask: unreliableMask, weakPick: weakPick,
  calibrate: calibrate, decideAbs: decideAbs, relRows: relRows, preRows: preRows, guess: guess, readDigits: readDigits,
  decisionSig: decisionSig, decidePage: decidePage,
  // added for scanner.js (dev trace): thresholds and the relative-rule helpers
  DIGIT_MARGIN_FROM: DIGIT_MARGIN_FROM, DIGIT_MARGIN_FULL: DIGIT_MARGIN_FULL, REL_SURE: REL_SURE, REL_SURE_DIGIT: REL_SURE_DIGIT,
  GUESS_GAP: GUESS_GAP, GUESS_REL_GAP: GUESS_REL_GAP, relTrustAnswers: relTrustAnswers, relTrustDigits: relTrustDigits,
  relativeMarks: function (md) { return relRows([md])[0]; } };
})(globalThis.OMR = globalThis.OMR || {});
