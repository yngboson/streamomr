// omr/fusion.py: multi-frame fusion: one physical sheet seen in many frames -> one set of decisions.
// A *track* is a run of frames believed to be the same physical sheet.  Per-bubble raw measurements (result.raw) are fused with a
// per-bubble median over the track's recent frames, and the ONE decision function (decide.decidePage) runs on the fused arrays.
// Pure: no I/O, no clock.  float32 matrices (FA, FI, FS, MA, MI, MS) are Float32Array rows, strengths Float64Array rows.
(function (O) {
'use strict';
var T = O.types, D = O.decide, F32 = Math.fround;

var WINDOW = 12;              // most recent frames fused (the median adapts, memory is bounded)
var CONTRADICT_FILL = 0.6;    // per-bubble fill swing that counts as a contradiction ...
var CONTRADICT_BUBBLES = 2;   // ... when at least this many bubbles swing
var JUMP_FRACTION = 0.2;      // sheet-quad centre jump (fraction of sheet size) between consecutive frames
var SCALE_JUMP = 0.35;        // relative change of the sheet size between consecutive frames
var ID_FLAGS = ['id_blank', 'id_unreadable', 'alignment_unverified', 'subject_blank', 'subject_unreadable', 'id_guessed',
  'subject_guessed', 'guessed_answers'];
var f_contradict = F32(CONTRADICT_FILL);

// omr/fusion.py:40: (cx, cy, size) of the projected page rectangle in image px (independent of which markers were visible)
function sheetQuad(r) {
  if (r.homography == null) return null;
  var q;
  try { q = T.imagePoints(r, [[0, 0], [210, 0], [210, 297], [0, 297]]); } catch (e) { return null; }
  var sx = 0, sy = 0, i, xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (i = 0; i < 4; i++) {
    sx += q[i][0]; sy += q[i][1];
    xmin = Math.min(xmin, q[i][0]); xmax = Math.max(xmax, q[i][0]); ymin = Math.min(ymin, q[i][1]); ymax = Math.max(ymax, q[i][1]);
  }
  return [sx / 4, sy / 4, Math.max(xmax - xmin, ymax - ymin)];
}

// A frame can feed a track: raw measurements present, verified, and consistent with the frame's decisions
function trustworthy(r) {
  var raw = r.raw;
  if (raw == null || !raw.verified) return false;
  var sig = raw.decided_sig;
  if (sig == null) { var d = D.decidePage(raw); sig = D.decisionSig(d.answers, d.student_id, d.subject); }
  return sig === D.decisionSig(r.answers, r.student_id, r.subject);
}

function jumped(a, b) {
  var x0 = a[0], y0 = a[1], s0 = a[2], x1 = b[0], y1 = b[1], s1 = b[2];
  return Math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)) > JUMP_FRACTION * Math.max(s0, s1) ||
    Math.abs(s1 - s0) > SCALE_JUMP * Math.max(s0, s1);
}

// per element: median of the non-NaN values of the frames (NaN when none); typed = Float32Array / Float64Array constructor
function medianCols(mats, Ctor, rejectBad) {
  var n = mats.length, rows = mats[0].length, out = [];
  for (var i = 0; i < rows; i++) {
    var cols = mats[0][i].length, row = new Ctor(cols);
    for (var j = 0; j < cols; j++) {
      var v = [];
      for (var k = 0; k < n; k++) { var x = mats[k][i][j]; if (x === x && !(rejectBad && rejectBad[k][i][j])) v.push(x); }
      if (!v.length) { row[j] = NaN; continue; }
      v.sort(function (a, b) { return a - b; });
      var m = v.length;
      if (m % 2) row[j] = v[(m - 1) / 2];
      else row[j] = Ctor === Float32Array ? F32(F32(v[m / 2 - 1] + v[m / 2]) / 2) : (v[m / 2 - 1] + v[m / 2]) / 2;
    }
    out.push(row);
  }
  return out;
}

function medianScalars(vals) {
  var v = vals.slice().sort(function (a, b) { return a - b; }), m = v.length;
  return m % 2 ? v[(m - 1) / 2] : (v[m / 2 - 1] + v[m / 2]) / 2;
}

// fill + strength: a frame that judged a bubble unknowable (sentinel) does not vote on it; unknowable in every frame -> stays unknowable
function medFs(fr, ff, sf) {
  var Fs = fr.map(function (x) { return x[ff]; }), Ss = fr.map(function (x) { return x[sf]; });
  var bad = Fs.map(function (F, k) { return F.map(function (row, i) { return D.unreliableMask(row, Ss[k][i]); }); });
  var mf = medianCols(Fs, Float32Array, bad), ms = medianCols(Ss, Float64Array, bad);
  for (var i = 0; i < mf.length; i++) for (var j = 0; j < mf[i].length; j++) {
    var allbad = true;
    for (var k = 0; k < fr.length; k++) if (!bad[k][i][j]) { allbad = false; break; }
    if (allbad) { mf[i][j] = T.UNRELIABLE_FILL; ms[i][j] = T.UNRELIABLE_STRENGTH; }
  }
  return [mf, ms];
}

// disc darkness (relative rule): NaN = unknowable bubble; a frame without it does not vote; null when no frame has it
function nanmed(fr, f) {
  var have = fr.filter(function (x) { return x[f] != null; }).map(function (x) { return x[f]; });
  return have.length ? medianCols(have, Float32Array, null) : null;
}

function Track(no, r, t) {
  this.no = no;
  this.form_key = r.form_key; this.page = r.page;
  this.frames = [];
  this.serials = new Map();
  this.t0 = this.t_last = t;
  this.n = 0;
  this.quad = null;
  this.committed = false;
  this.held_since = null;       // set while a stable-but-unreadable-ID result is being held
  this.best = null;             // [sharpness, result, payload, t] of the sharpest frame
  this._dec = null;
  this._raw = null;
  this.add(r, t, null);
}

// ------------------------------------------------------------------ fusion
Track.prototype.fusedRaw = function () {
  if (this._raw === null) {
    var fr = this.frames, r0 = fr[fr.length - 1];
    var a = medFs(fr, 'FA', 'SA'), i = medFs(fr, 'FI', 'SI'), subj = {};
    if (r0.FS != null) {     // same form on every frame of a track (form_key), so FS is present on all or none
      var s = medFs(fr, 'FS', 'SS');
      subj = { FS: s[0], SS: s[1], MS: nanmed(fr, 'MS') };
    }
    subj.MA = nanmed(fr, 'MA'); subj.MI = nanmed(fr, 'MI');
    subj.ink_source = fr.some(function (x) { return x.ink_source === 'default'; }) ? 'default' : 'auto';
    subj.ink_separation = medianScalars(fr.map(function (x) { return x.ink_separation; }));
    this._raw = T.PageRaw(r0.numbers, r0.labels, r0.digit_index, a[0], a[1], i[0], i[1],
      medianScalars(fr.map(function (x) { return x.ink; })), medianScalars(fr.map(function (x) { return x.min_strength; })), true, subj);
  }
  return this._raw;
};

Track.prototype.decisions = function () {
  if (this._dec === null) this._dec = D.decidePage(this.fusedRaw());
  return this._dec;
};

Track.prototype.serial = function () {      // most common serial (first inserted among ties)
  var best = null, bn = 0;
  this.serials.forEach(function (n, s) { if (n > bn) { best = s; bn = n; } });
  return best;
};

// ------------------------------------------------------------------ membership
Track.prototype.incompatible = function (r) {
  if (r.form_key !== this.form_key || r.page !== this.page) return 'other_sheet';
  var ts = this.serial();
  if (r.serial != null && ts != null && r.serial !== ts) return 'other_serial';
  var q = sheetQuad(r);
  if (q !== null && this.quad !== null && jumped(this.quad, q)) return 'moved';
  if (this.n >= 2 && this._contradicts(r)) return 'contradiction';
  return null;
};

Track.prototype._contradicts = function (r) {
  var dec = this.decisions(), i, n;
  n = Math.min(dec.answers.length, r.answers.length);
  for (i = 0; i < n; i++) {
    var a = dec.answers[i], b = r.answers[i];
    if (!a.ambiguous && !b.ambiguous && (a.state !== b.state || JSON.stringify(a.marked) !== JSON.stringify(b.marked))) return true;   // (a guessed row is ambiguous: never a contradiction)
  }
  var pairs = [];
  n = Math.min(dec.id_digits.length, r.id_digits.length);
  for (i = 0; i < n; i++) pairs.push([dec.id_digits[i], r.id_digits[i]]);
  n = Math.min(dec.subject_digits.length, r.subject_digits.length);
  for (i = 0; i < n; i++) pairs.push([dec.subject_digits[i], r.subject_digits[i]]);
  for (i = 0; i < pairs.length; i++) {
    var p = pairs[i][0], q = pairs[i][1];
    if (p.value !== null && q.value !== null && !(p.guessed || q.guessed) && p.value !== q.value) return true;
  }
  var fr = this.fusedRaw(), raw = r.raw, swing = 0;
  function count(A, B) {
    var c = 0;
    for (var i = 0; i < A.length; i++) for (var j = 0; j < A[i].length; j++) if (Math.abs(F32(A[i][j] - B[i][j])) >= f_contradict) c++;
    return c;
  }
  swing += count(fr.FA, raw.FA) + count(fr.FI, raw.FI);
  if (fr.FS != null && raw.FS != null) swing += count(fr.FS, raw.FS);
  return swing >= CONTRADICT_BUBBLES;
};

Track.prototype.add = function (r, t, payload) {
  this.frames.push(r.raw);
  if (this.frames.length > WINDOW) this.frames.shift();
  this.n += 1;
  this.t_last = t;
  if (r.serial != null) this.serials.set(r.serial, (this.serials.get(r.serial) || 0) + 1);
  this.quad = sheetQuad(r) || this.quad;
  this._raw = this._dec = null;
  var sharp = Number((r.quality || {}).sharpness || 0.0);
  if (this.best === null || sharp > this.best[0]) this.best = [sharp, r, payload, t];
};

// ------------------------------------------------------------------ result
// Copy of frame result `r` carrying the track's fused decisions
Track.prototype.apply = function (r) {
  var dec = this.decisions(), out = Object.assign({}, r);
  out.answers = dec.answers; out.id_digits = dec.id_digits; out.student_id = dec.student_id; out.calibration = dec.calibration;
  out.subject = dec.subject; out.subject_digits = dec.subject_digits;
  out.flags = r.flags.filter(function (f) { return ID_FLAGS.indexOf(f) < 0; }).concat(dec.flags);
  var s = this.serial();
  if (s !== null) {
    out.serial = s;
    out.flags = out.flags.filter(function (f) { return f !== 'serial_unreadable'; });
  }
  out.fused_frames = this.frames.length;
  out.trace_src = [this.fusedRaw(), this.frames.slice()];   // the fused measurements + the per-frame ones, for the dev log
  return out;
};

// No confident ID yet: unreadable, or only a best guess (more frames may still settle it)
Track.prototype.idUnreadable = function () {
  var f = this.decisions().flags;
  return f.indexOf('id_unreadable') >= 0 || f.indexOf('id_guessed') >= 0;
};

Track.prototype.subjectUnreadable = function () {
  var f = this.decisions().flags;
  return f.indexOf('subject_unreadable') >= 0 || f.indexOf('subject_guessed') >= 0;
};

O.fusion = { WINDOW: WINDOW, CONTRADICT_FILL: CONTRADICT_FILL, CONTRADICT_BUBBLES: CONTRADICT_BUBBLES, JUMP_FRACTION: JUMP_FRACTION,
  SCALE_JUMP: SCALE_JUMP, ID_FLAGS: ID_FLAGS, sheetQuad: sheetQuad, trustworthy: trustworthy, jumped: jumped, Track: Track };
})(globalThis.OMR = globalThis.OMR || {});
