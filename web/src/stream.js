// omr/stream.py: frame-by-frame "barcode scanner" state machine.  Pure: no I/O, no clock, no threads.
//
//   var session = new OMR.stream.StreamSession();      // self-describing sheets: no database or layout resolver is needed
//   var event = session.feed(recognitionResult, tSeconds, payload);
//
// event = {state, hint, overlay, frame_index, present, possibly_same, unread, commit?, revision?}
// See the long rule description in omr/stream.py (states idle / detecting / committed / cooldown; K consecutive agreeing ok frames
// commit; cooldown, interruption, sheet identity, fusion tracks, hold of an unreadable ID, revision of a committed sheet).
(function (O) {
'use strict';
var T = O.types, U = O.util, FU = O.fusion, ID = O.identity, REV = O.revision, LAY = O.layout;
var Track = FU.Track, trustworthy = FU.trustworthy, pyRoundInt = U.pyRoundInt, pyRound = U.pyRound;

var IDLE = 'idle', DETECTING = 'detecting', COMMITTED = 'committed', COOLDOWN = 'cooldown';
var DEFAULT_K = 3, DEFAULT_M = 5;
var DEFAULT_HINT_AFTER = 2.0;     // seconds a sheet may be in view without stabilising before a hint appears
var INTERRUPT_FRAMES = 2;         // consecutive non-ok frames that count as "the sheet was disturbed"
var JUMP_FRACTION = 0.2;          // marker-quad centre jump, as a fraction of the sheet size
var SCALE_JUMP = 0.35;            // relative change of the sheet size between consecutive frames
var SWAP_FRACTION = 0.3;          // share of differing questions that marks a different physical sheet
var LOW_RES_PX_PER_MM = 2.2;      // source px per mm below which "move closer" is the likely fix
var LOW_SHARPNESS = 40.0;         // Laplacian variance below which "hold still" is the likely fix
var DEFAULT_BUBBLE_MM = 4.0;
var HOLD_SECONDS = 1.5;           // an unreadable ID is given this long (sheet still in view) to resolve through fusion
var TRACK_GAP = INTERRUPT_FRAMES; // consecutive non-contributing frames that end a track
var MISSED_MIN_FRAMES = 2;        // ok frames a sheet must be seen in to count as "seen" for the missed-sheet check
var REVISE_FRAMES = 40;           // ok frames after a commit during which the same sheet's evidence may still revise it ...
var REVISE_SECONDS = 6.0;         // ... or this many seconds, whichever ends first
var ABSENT = ['no_sheet', 'partial'];
var B_EMPTY = 0, B_MARKED = 1, B_MULTI = 2, B_AMBIGUOUS = 3;     // overlay bubble codes

var J = JSON.stringify;

// omr/stream.py:92: key = [form_key, page, student_id, [[number, state, marked]...], sheet_key, subject]
function decisionKey(r) {
  return [r.form_key, r.page, r.student_id, r.answers.map(function (a) { return [a.number, a.state, a.marked.slice()]; }), ID.sheetKey(r), r.subject];
}
function identity(r) { return [r.form_key, r.page, r.student_id, r.subject]; }
function sheetId(r) { return [r.form_key, r.page, ID.sheetKey(r)]; }       // the physical-sheet identity used by the missed-sheet check

function answerDistance(a, b) {
  var da = {}, db = {}, keys = {}, k;
  a.forEach(function (x) { da[x[0]] = J([x[1], x[2]]); keys[x[0]] = 1; });
  b.forEach(function (x) { db[x[0]] = J([x[1], x[2]]); keys[x[0]] = 1; });
  var ks = Object.keys(keys);
  if (!ks.length) return 0.0;
  var diff = 0;
  for (var i = 0; i < ks.length; i++) { k = ks[i]; if (da[k] !== db[k]) diff++; }
  return diff / ks.length;
}

// (cx, cy, size) of the marker quad in image px, or null
function quadOf(r) {
  var xs = [], ys = [];
  r.marker_corners.forEach(function (e) { e[1].forEach(function (p) { xs.push(p[0]); ys.push(p[1]); }); });
  if (!xs.length) return null;
  var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
  return [(x0 + x1) / 2, (y0 + y1) / 2, Math.max(x1 - x0, y1 - y0)];
}

// Is `key` a different physical sheet than the one already committed (both ok-frame keys)?
function isOtherSheet(committedKey, key) {
  var f0 = committedKey[0], p0 = committedKey[1], s0 = committedKey[2], a0 = committedKey[3], k0 = committedKey[4], u0 = committedKey[5];
  var f1 = key[0], p1 = key[1], s1 = key[2], a1 = key[3], k1 = key[4], u1 = key[5];
  if (f0 !== f1 || p0 !== p1 || k0 !== k1) return true;
  if (s0 != null && s1 != null && s0 !== s1) return true;
  if (u0 != null && u1 != null && u0 !== u1) return true;
  return answerDistance(a0, a1) > SWAP_FRACTION;
}

function StreamSession(o) {
  o = o || {};
  this.fuse = o.fuse === undefined ? true : o.fuse;
  this.hold_s = o.hold_s === undefined ? HOLD_SECONDS : o.hold_s;
  var k = o.k === undefined ? DEFAULT_K : o.k;
  if (k < 2) throw new Error('k must be >= 2: a single frame must never commit');
  this.k = k; this.m = o.m === undefined ? DEFAULT_M : o.m;
  this.hint_after = o.hint_after === undefined ? DEFAULT_HINT_AFTER : o.hint_after;
  this.allowed_forms = o.allowed_forms == null ? null : new Set(o.allowed_forms);
  this.allowed_subjects = o.allowed_subjects == null ? null : new Set(o.allowed_subjects);
  this.resolve_layout = o.resolve_layout || null;       // legacy sheets only: form_id -> Layout (for the overlay bubble size)
  this.frame_index = -1;
  this.resetTracking();
  this.reset();
}
var P = StreamSession.prototype;

// Forget which sheets were seen / committed (start of a new scanning session)
P.resetTracking = function () {
  this._seen = new Map();          // 't:<track no>' (or 's:'+J(sheet_id) for unfused frames) -> info
  this._committed_ids = new Set(); // J(sheet_id)
  this._current = null;            // _seen key of the sheet in view right now
  this._track = null;
  this._track_no = 0;
  this._tgap = 0;                  // consecutive frames that did not contribute to the track
  this._ep = null;                 // presence episode in progress
  this._episodes = [];             // finished episodes
  this._gap = 0;                   // consecutive no_sheet frames
};

// Sheets (tracks) seen in >= MISSED_MIN_FRAMES ok frames but never committed (the sheet in view is excluded unless final)
P.missed = function (final) {
  var out = [], self = this;
  this._seen.forEach(function (info, key) {
    if (info.committed || self._committed_ids.has(J(info.sid)) || info.frames < MISSED_MIN_FRAMES) return;
    if (key === self._current && !final) return;
    var m = { form_key: info.sid[0], page: info.sid[1], serial: info.serial, student_id: info.student_id, frames: info.frames };
    if (info.possibly_same_as) m.possibly_same_as = info.possibly_same_as;
    out.push(m);
  });
  return out.map(function (m, i) { return [m, i]; }).sort(function (a, b) {
    var ka = [a[0].serial || 0, a[0].page != null ? a[0].page : -1], kb = [b[0].serial || 0, b[0].page != null ? b[0].page : -1];
    return ka[0] - kb[0] || ka[1] - kb[1] || a[1] - b[1];
  }).map(function (x) { return x[0]; });
};

P._trackPresence = function (status, t) {
  if (status === 'no_sheet') {
    this._gap += 1;
    if (this._ep !== null && this._gap >= this.m) { this._episodes.push(this._ep); this._ep = null; }
    return;
  }
  this._gap = 0;
  if (this._ep === null) this._ep = { t0: t, t1: t, frames: 0, ok: 0, commits: 0, statuses: {}, page: null };
  var e = this._ep;
  e.t1 = t;
  e.frames += 1;
  e.ok += status === 'ok' ? 1 : 0;
  e.statuses[status] = (e.statuses[status] || 0) + 1;
};

// Sheet-present stretches (any marker visible, incl. `partial`) that never produced a commit.  The one in progress is included
// only with `final`.
P.unreadEpisodes = function (final) {
  var eps = this._episodes.slice();
  if (final && this._ep !== null) eps.push(this._ep);
  return eps.filter(function (e) { return e.commits === 0 && e.frames >= MISSED_MIN_FRAMES; }).map(function (e) { return Object.assign({}, e); });
};

P.reset = function () {
  this.state = IDLE;
  this._absent = 0;
  this._run_key = null;
  this._run_n = 0;
  this._best = null;               // [sharpness, result, payload, t]
  this._committed_key = null;
  this._committed_res = null;      // the committed (fused) result, for identity-by-evidence
  this._since = null;              // when the current uncommitted sheet came into view
  this._nonok = 0;                 // consecutive non-ok frames
  this._interrupted = false;       // sheet disturbed since the last commit
  this._prev_quad = null;          // (cx, cy, size) of the previous ok frame
  this._rev = null;                // revision watch of the committed track (see _revise)
};

function clearRun(s) { s._run_key = null; s._run_n = 0; s._best = null; }

// ------------------------------------------------------------------ main entry
P.feed = function (r, t, payload) {
  if (payload === undefined) payload = null;
  this.frame_index += 1;
  var commit = null, hint = '', status = r.status, pre = null;
  this._revision = null;
  if (status === 'ok') this._nonok = 0;
  else {
    this._nonok += 1;
    this._prev_quad = null;
    if (this._nonok >= INTERRUPT_FRAMES && this._committed_key !== null) this._interrupted = true;
  }
  // ---- track bookkeeping (fusion)
  if (status === 'ok' && this.fuse && trustworthy(r)) {
    this._tgap = 0;
    var why = this._track !== null ? this._track.incompatible(r) : null;
    if (why !== null) pre = this._endTrack();
    if (this._track === null) {
      this._track_no += 1;
      this._track = new Track(this._track_no, r, t);
      this._track.best = [Number((r.quality || {}).sharpness || 0.0), r, payload, t];
    } else this._track.add(r, t, payload);
    r = this._track.apply(r);
  } else {
    this._tgap += 1;
    if (this._tgap >= TRACK_GAP && this._track !== null) pre = this._endTrack();
  }
  this._trackPresence(status, t);

  if (pre !== null) commit = pre;
  if (ABSENT.indexOf(status) >= 0) {
    clearRun(this);
    this._absent += 1;
    if (this._absent >= this.m) { this._current = null; this.reset(); }
    else if (this.state === COMMITTED) this.state = COOLDOWN;
    hint = this._absentHint(r);
  } else {
    this._absent = 0;
    if (this._since === null && (this.state === IDLE || this.state === DETECTING)) this._since = t;
    if (this.state === IDLE) this.state = DETECTING;
    else if (this.state === COMMITTED) this.state = COOLDOWN;
    if (status === 'ok') {
      var fo = this._feedOk(r, t, payload);
      hint = fo[0];
      commit = commit || fo[1];
    } else {
      clearRun(this);
      hint = this._badHint(r, t);
    }
  }

  if (commit !== null) {
    this.state = COMMITTED;
    hint = 'Scanned';
    if (this._ep !== null) this._ep.commits += 1;
  } else if (this.state === COOLDOWN && !hint) hint = 'Scanned - show the next sheet';
  else if (this.state === IDLE && !hint) hint = 'Show a sheet';
  var possibly = false;
  this._seen.forEach(function (i) { if (i.possibly_same_as && !i.committed) possibly = true; });
  var ev = { state: this.state, hint: hint, overlay: this.overlay(r), frame_index: this.frame_index,
    present: status !== 'no_sheet', possibly_same: possibly,
    unread: !!(this._ep !== null && this._ep.commits === 0 && this._ep.frames >= MISSED_MIN_FRAMES) };
  if (commit !== null) ev.commit = commit;
  else if (this._revision !== null) ev.revision = this._revision;
  return ev;
};

// End of stream: a held (stable, ID still unreadable) track is committed as it is
P.finish = function () {
  var c = this._endTrack();
  if (c !== null) this.state = COMMITTED;
  return c;
};

// ------------------------------------------------------------------ tracks
// Close the current track; commit it if it was held (stable but not yet committed)
P._endTrack = function () {
  var tr = this._track;
  this._track = null;
  this._tgap = 0;
  if (tr === null || tr.committed || tr.held_since === null) return null;
  var payload = tr.best[2], t = tr.best[3], r0 = tr.best[1];
  var final = tr.apply(r0), key = decisionKey(final);
  if (this._committed_key !== null && this.state === COOLDOWN) {      // same re-commit rules as for a live frame
    if (this._interrupted) {
      if (J(key) === J(this._committed_key)) return null;
      if (this._sameByEvidence(final)) { this._notePossible('t:' + tr.no, final); return null; }
    } else if (!isOtherSheet(this._committed_key, key)) return null;
  }
  return this._doCommit(final, payload, t, key, tr.n, tr);
};

P._doCommit = function (final, payload, t, key, frames, track) {
  var commit = { result: final, payload: payload, key: key, identity: identity(final), t: t, frames: frames,
    frame_index: this.frame_index, fused_frames: final.fused_frames };
  this._committed_key = key;
  this._committed_res = final;
  this._rev = null;
  if (track && this.fuse && REV.doubtful(final)) this._rev = { track: track, base: final, t0: t, n: 0, contra: new Set(), count: 0 };
  this._committed_ids.add(J(sheetId(final)));
  if (track) {
    track.committed = true;
    var info = this._seen.get('t:' + track.no);
    if (info) info.committed = true;
  }
  clearRun(this);
  this._since = null;
  this._interrupted = false;
  return commit;
};

// ------------------------------------------------------------------ ok frames
P._feedOk = function (r, t, payload) {
  if (this.allowed_forms !== null && !this.allowed_forms.has(r.form_key)) { clearRun(this); return ['Sheet is another form', null]; }
  if (this.allowed_subjects !== null && r.subject != null && !T.subjectGuessed(r) && !this.allowed_subjects.has(r.subject)) {
    clearRun(this);        // (a guessed subject never filters a sheet out)
    return ['Sheet is another subject', null];
  }
  var key = decisionKey(r), sid = sheetId(r);
  var tr = r.fused_frames ? this._track : null;
  var skey = tr !== null ? 't:' + tr.no : 's:' + J(sid);
  this._current = skey;
  var seen = this._seen.get(skey);
  if (!seen) { seen = { frames: 0, committed: false }; this._seen.set(skey, seen); }
  seen.sid = sid; seen.serial = ID.serialOf(r); seen.student_id = r.student_id; seen.subject = r.subject;
  seen.frames += 1;
  var q = quadOf(r);
  if (q !== null && this._prev_quad !== null && this._committed_key !== null && FU.jumped(this._prev_quad, q)) this._interrupted = true;
  this._prev_quad = q;
  if (this._run_key !== null && J(key) === J(this._run_key)) this._run_n += 1;
  else { this._run_key = key; this._run_n = 1; this._best = null; }
  var sharp = Number((r.quality || {}).sharpness || 0.0);
  if (this._best === null || sharp > this._best[0]) this._best = [sharp, r, payload, t];
  if (tr !== null && tr.committed) {
    this._revise(tr, r, t);                // one commit per track; later evidence may only revise it
    return ['', null];
  }
  var committedHere = this.state === COOLDOWN && this._committed_key !== null;
  if (committedHere) {
    if (this._interrupted) {
      var same = J(key) !== J(this._committed_key) && this._sameByEvidence(r);
      if (same) this._notePossible(skey, r);
      if (J(key) === J(this._committed_key) || same) {    // same sheet put back / re-seen
        if (this._run_n >= this.k) this._interrupted = false;
        return ['', null];
      }
    } else if (!isOtherSheet(this._committed_key, key)) return ['', null];     // same sheet still in view (maybe flickering)
  }
  if (this._run_n < this.k) return [this.state !== COOLDOWN ? this._unstableHint(r, t) : '', null];
  if (tr !== null && (tr.idUnreadable() || tr.subjectUnreadable())) {    // stable, but may still resolve: hold while in view
    if (tr.held_since === null) tr.held_since = t;
    if (t - tr.held_since < this.hold_s) return ['Reading the student ID...', null];
  }
  var best = this._pick();
  var commit = this._doCommit(this._finalize(best[1]), best[2], best[3], key, this._run_n, tr);
  return ['', commit];
};

// Same track, already committed: upgrade the committed result from the newly fused frame `r` (see omr/revision.py)
P._revise = function (tr, r, t) {
  var rv = this._rev;
  if (rv === null || rv.track !== tr) return;
  rv.n += 1;
  if (rv.n > REVISE_FRAMES || t - rv.t0 > REVISE_SECONDS) { this._rev = null; return; }
  var base = rv.base, m = REV.merge(base, r), changed = m[1], contra = m[2];
  var newContra = contra.filter(function (c) { return !rv.contra.has(c); });
  if (!changed.length && !newContra.length) return;
  m = REV.merge(base, tr.apply(tr.best[1]));      // images / geometry of the sharpest frame so far
  var merged = m[0];
  changed = m[1]; contra = m[2];
  contra.forEach(function (c) { rv.contra.add(c); });
  rv.count += 1;
  var key = decisionKey(merged);
  var ev = { result: merged, payload: tr.best[2], key: key, identity: identity(merged), t: t, frames: tr.n, frame_index: this.frame_index,
    fused_frames: merged.fused_frames, changed: changed, contradiction: Array.from(rv.contra).sort(), new_contradiction: newContra,
    revision: rv.count, severity: ID.severity(merged), prev_severity: ID.severity(base), prev_key: this._committed_key };
  rv.base = merged;
  this._committed_key = key; this._committed_res = merged;
  this._committed_ids.add(J(sheetId(merged)));
  if (!REV.doubtful(merged)) this._rev = null;
  this._revision = ev;
};

// After an interruption: a re-sighting whose ID is only partly readable but whose readable digits (at least half) all agree with
// the committed sheet's ID, with IDENTICAL decisions on every answer and equal serials, is treated as that sheet put back.
P._sameByEvidence = function (r) {
  var c = this._committed_res;
  if (c === null || c.student_id == null || T.idGuessed(c) || (r.student_id != null && !T.idGuessed(r)) || c.form_key !== r.form_key || c.page !== r.page)
    return false;
  if (r.subject != null && !T.subjectGuessed(r) && r.subject !== c.subject) return false;    // a readable, different subject code is another sheet
  if ((r.serial != null || c.serial != null) && r.serial !== c.serial) return false;
  var read = [];
  r.id_digits.forEach(function (d) { if (d.value != null && !d.guessed) read.push([d.index, d.value]); });   // a guess is no evidence
  if (read.length === r.id_digits.length) return false;
  if (read.length * 2 < c.id_digits.length) return false;
  for (var i = 0; i < read.length; i++) {
    var idx = read[i][0];
    if (idx < c.student_id.length && c.student_id.charAt(idx) !== String(read[i][1])) return false;
  }
  function rows(x) { return J(x.answers.map(function (a) { return [a.number, a.state, a.marked]; })); }
  return rows(c) === rows(r);
};

// A commit was suppressed by `_sameByEvidence`: keep it visible (missed report + event)
P._notePossible = function (skey, r) {
  var info = this._seen.get(skey);
  if (info) { info.possibly_same_as = this._committed_res.student_id; info.student_id = r.student_id; }
};

// The frame result chosen for the commit, carrying the track's latest fused decisions
P._finalize = function (r) {
  if (r.fused_frames && this._track !== null) return this._track.apply(r);
  return r;
};

P._pick = function () { return this._best; };     // seam: fuse the agreeing frames' scores here instead of picking one

// ------------------------------------------------------------------ hints
P._stuck = function (t) { return this._since !== null && (t - this._since) >= this.hint_after; };

P._unstableHint = function (r, t) {
  if (!this._stuck(t)) return 'Reading...';
  var q = r.quality || {};
  if (r.student_id == null || T.idGuessed(r)) return 'Student ID not readable - hold still, check the ID marks';
  if (r.subject == null || T.subjectGuessed(r)) return 'Subject code not readable - hold still, check the subject marks';
  if ((q.source_px_per_mm !== undefined ? q.source_px_per_mm : 99) < LOW_RES_PX_PER_MM) return 'Move closer';
  if ((q.sharpness !== undefined ? q.sharpness : 1e9) < LOW_SHARPNESS) return 'Hold still (blurry)';
  return 'Hold still - reading keeps changing (glare or shadow?)';
};

P._badHint = function (r, t) {
  var s = r.status;
  if (s === 'unknown_form') return 'Unknown sheet (not created in this system)';
  if (!this._stuck(t)) return 'Reading...';
  if (s === 'bad_geometry') return 'Flatten the sheet and keep it square to the camera';
  if (s === 'bad_meta') return 'Cannot read the header - move closer or reduce glare';
  if (s === 'inconsistent') return 'Several sheets visible - show one sheet';
  return 'Hold still';
};

P._absentHint = function (r) {
  if (r.status === 'partial') {
    var c = r.missing_corners.join(',');
    return c ? 'Marker covered or out of frame (' + c + ')' : 'Marker covered or out of frame';
  }
  return '';
};

// ------------------------------------------------------------------ overlay
P.overlay = function (r) {
  var ov = { markers: r.marker_corners.map(function (e) { return e[1].map(function (p) { return [pyRoundInt(p[0]), pyRoundInt(p[1])]; }); }) };
  if (r.homography != null) {
    try { ov.sheet = T.imagePoints(r, [[0, 0], [210, 0], [210, 297], [0, 297]]).map(function (p) { return [pyRoundInt(p[0]), pyRoundInt(p[1])]; }); }
    catch (e) { /* no sheet outline */ }
  }
  if (r.status === 'ok' && r.homography != null && r.bubble_centers_mm && Object.keys(r.bubble_centers_mm).length) {
    var b = this._bubbles(r);
    ov.bubbles = b[0]; ov.bubble_r = b[1];
  }
  return ov;
};

P._bubbles = function (r) {
  var pts = [], codes = [], bc = r.bubble_centers_mm, i, k;
  for (i = 0; i < Math.min(r.answers.length, bc.answers.length); i++) {
    var a = r.answers[i], row = bc.answers[i];
    for (k = 0; k < row.length; k++) {
      pts.push(row[k]);
      if (a.marked.indexOf(k) >= 0) codes.push(a.state === 'multi' ? B_MULTI : a.ambiguous ? B_AMBIGUOUS : B_MARKED);
      else codes.push(a.ambiguous && (a.guess === k || (a.scores && a.scores[k] > 0.1)) ? B_AMBIGUOUS : B_EMPTY);
    }
  }
  [[r.id_digits, bc.id], [r.subject_digits, bc.subject || []]].forEach(function (g) {
    var digits = g[0], rows = g[1];
    for (var j = 0; j < Math.min(digits.length, rows.length); j++) {
      var d = digits[j];
      for (var kk = 0; kk < rows[j].length; kk++) {
        pts.push(rows[j][kk]);
        if (d.value === kk) codes.push(d.guessed ? B_AMBIGUOUS : B_MARKED);
        else if (d.ambiguous && d.scores && d.scores[kk] > 0.1) codes.push(B_AMBIGUOUS);
        else codes.push(B_EMPTY);
      }
    }
  });
  var img = T.imagePoints(r, pts), dia = DEFAULT_BUBBLE_MM;
  if (r.page != null) {
    try {
      var lay;
      if (r.form_id != null) lay = this.resolve_layout(r.form_id);       // legacy sheet: layout from the resolver
      else lay = LAY.layoutForHeader(LAY.headerFromFormKey(r.form_key));  // self-describing: from the form_key
      dia = lay.pages[r.page].bubble_diameter;
    } catch (e) { /* default size */ }
  }
  var ppm = (r.quality || {}).source_px_per_mm;
  var rad = Math.max(1.5, 0.5 * dia * Number(ppm !== undefined ? ppm : 3.0));
  return [img.map(function (p, j) { return [pyRoundInt(p[0]), pyRoundInt(p[1]), codes[j]]; }), pyRound(rad, 1)];
};

O.stream = { IDLE: IDLE, DETECTING: DETECTING, COMMITTED: COMMITTED, COOLDOWN: COOLDOWN, DEFAULT_K: DEFAULT_K, DEFAULT_M: DEFAULT_M,
  HOLD_SECONDS: HOLD_SECONDS, REVISE_FRAMES: REVISE_FRAMES, REVISE_SECONDS: REVISE_SECONDS, MISSED_MIN_FRAMES: MISSED_MIN_FRAMES,
  decisionKey: decisionKey, identity: identity, sheetId: sheetId, isOtherSheet: isOtherSheet, StreamSession: StreamSession };
})(globalThis.OMR = globalThis.OMR || {});
