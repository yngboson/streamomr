// jsc test: replays the Python-recorded fixtures (web/test/fixtures/logic/*.json) through the JS logic modules.
// usage: jsc util.js ... stream.js test.js -- <fixtures dir>      (run.py does this)
(function (ARGS) {
'use strict';
var O = globalThis.OMR, U = O.util, T = O.types;
var DIR = ARGS[0];
var TOL = 1e-9;
var stats = {}, shown = 0, MAXSHOW = 25;

function rec(area, ok, msg) {
  var s = stats[area] || (stats[area] = { pass: 0, fail: 0 });
  if (ok) s.pass++; else { s.fail++; if (shown++ < MAXSHOW) print('FAIL [' + area + '] ' + msg); }
}

// compare expected (parsed JSON) with actual (plain-ified); returns '' or the first difference
function diff(e, a, path) {
  if (typeof e === 'number' && typeof a === 'number') {
    if (e === a) return '';
    if (Math.abs(e - a) <= TOL * Math.max(1, Math.abs(e))) return '';
    return path + ': expected ' + e + ' got ' + a;
  }
  if (e === null || a === null || e === undefined || a === undefined || typeof e !== 'object' || typeof a !== 'object') {
    if (e === a) return '';
    if (e == null && a == null) return '';
    return path + ': expected ' + JSON.stringify(e) + ' got ' + JSON.stringify(a);
  }
  if (Array.isArray(e) !== Array.isArray(a)) return path + ': array/object mismatch';
  if (Array.isArray(e)) {
    if (e.length !== a.length) return path + ': length ' + e.length + ' vs ' + a.length;
    for (var i = 0; i < e.length; i++) { var d = diff(e[i], a[i], path + '[' + i + ']'); if (d) return d; }
    return '';
  }
  var ke = Object.keys(e).filter(function (k) { return e[k] !== undefined; }), ka = Object.keys(a).filter(function (k) { return a[k] !== undefined; });
  for (var j = 0; j < ke.length; j++) {
    if (!(ke[j] in a) || a[ke[j]] === undefined) { if (e[ke[j]] === null) continue; return path + '.' + ke[j] + ': missing in actual'; }
    var dd = diff(e[ke[j]], a[ke[j]], path + '.' + ke[j]); if (dd) return dd;
  }
  for (j = 0; j < ka.length; j++) if (!(ka[j] in e) && a[ka[j]] !== null) return path + '.' + ka[j] + ': extra in actual (' + JSON.stringify(a[ka[j]]).slice(0, 60) + ')';
  return '';
}

function plain(v) {      // typed arrays -> arrays, NaN -> null, Sets -> sorted arrays
  if (v === undefined) return undefined;
  if (typeof v === 'number') return isFinite(v) ? v : null;
  if (v === null || typeof v !== 'object') return v;
  if (v instanceof Set) return Array.from(v).sort();
  if (Array.isArray(v) || ArrayBuffer.isView(v)) return Array.prototype.map.call(v, plain);
  var o = {};
  Object.keys(v).forEach(function (k) { if (typeof v[k] !== 'function') o[k] = plain(v[k]); });
  return o;
}

function load(name) { return JSON.parse(readFile(DIR + '/' + name)); }

// ---------------------------------------------------------------- fixture -> JS result objects
var F32M = ['FA', 'FI', 'FS', 'MA', 'MI', 'MS'], F64M = ['SA', 'SI', 'SS'];
function mat(rows, Ctor) {
  if (rows == null) return null;
  return rows.map(function (r) { return Ctor.from(r.map(function (x) { return x === null ? NaN : x; })); });
}
function loadRaw(d) {
  if (d == null) return null;
  var r = Object.assign({}, d);
  F32M.forEach(function (k) { r[k] = mat(d[k], Float32Array); });
  F64M.forEach(function (k) { r[k] = mat(d[k], Float64Array); });
  r.decided_sig = d.decided_sig == null ? null : JSON.stringify(d.decided_sig);
  return r;
}
function makeField(s) {
  function edge(side, u) {
    var xs = s.fn[side].u, v = s.fn[side].v, n = xs.length;
    if (u <= xs[0]) return v[0].slice();
    if (u >= xs[n - 1]) return v[n - 1].slice();
    var i = 0; while (xs[i + 1] < u) i++;
    var t = (u - xs[i]) / (xs[i + 1] - xs[i]);
    return [v[i][0] + (v[i + 1][0] - v[i][0]) * t, v[i][1] + (v[i + 1][1] - v[i][1]) * t];
  }
  function cl(v, a, b) { return Math.min(Math.max(v, a), b); }
  return function (x, y) {      // omr/tracks.py Field.__call__ (test-side stand-in for J3's tracks.js)
    var sx = cl((x - s.XL) / (s.XR - s.XL), 0, 1), ty = cl((y - s.YT) / (s.YB - s.YT), 0, 1), xc = cl(x, s.XL, s.XR), yc = cl(y, s.YT, s.YB);
    var eT = edge('T', xc), eB = edge('B', xc), eL = edge('L', yc), eR = edge('R', yc), c = s.c, d = [0, 0];
    for (var k = 0; k < 2; k++)
      d[k] = (1 - ty) * eT[k] + ty * eB[k] + (1 - sx) * eL[k] + sx * eR[k] -
        ((1 - sx) * (1 - ty) * c.TL[k] + sx * (1 - ty) * c.TR[k] + (1 - sx) * ty * c.BL[k] + sx * ty * c.BR[k]);
    return d;
  };
}
function loadResult(d) {
  var r = Object.assign({}, d);
  r.raw = loadRaw(d.raw);
  r.dewarp = d.dewarp ? makeField(d.dewarp) : null;
  return r;
}

function plainRaw(raw) {
  if (raw == null) return null;
  var o = plain(Object.assign({}, raw, { decided_sig: null }));
  o.decided_sig = raw.decided_sig == null ? null : JSON.parse(raw.decided_sig);
  return o;
}
function plainEventResult(r) {
  var o = {};
  Object.keys(r).forEach(function (k) { if (['raw', 'dewarp', 'trace_src', 'warped', 'name_crop', 'timing_ms'].indexOf(k) < 0) o[k] = plain(r[k]); });
  o.timing_ms = {};
  o.fused_raw = r.trace_src ? plainRaw(r.trace_src[0]) : null;
  o.trace_n = r.trace_src ? r.trace_src[1].length : null;
  return o;
}
function plainEvent(ev) {
  var o = {};
  Object.keys(ev).forEach(function (k) {
    var v = ev[k];
    if (k === 'commit' || k === 'revision') { var c = Object.assign({}, v); c.result = plainEventResult(v.result); o[k] = plain(c); }
    else o[k] = plain(v);
  });
  return o;
}
function plainResultNoRaw(r) {
  var o = {};
  Object.keys(r).forEach(function (k) { if (['raw', 'dewarp', 'trace_src'].indexOf(k) < 0) o[k] = plain(r[k]); });
  o.raw = null; o.dewarp = null;
  return o;
}

// ---------------------------------------------------------------- markers
function testMarkers() {
  var M = O.markers, f = load('markers.json'), i;
  f.bits.forEach(function (s, id) { rec('markers', M.markerBits(id).map(function (r) { return r.map(function (b) { return b ? '1' : '0'; }).join(''); }).join('') === s, 'bits ' + id); });
  f.ids.forEach(function (x) { rec('markers', M.markerId(x[0], x[1]) === x[2] && M.decodeMarkerId(x[2])[0] === x[0] && M.decodeMarkerId(x[2])[1] === x[1], 'ids'); });
  Object.keys(f.image).forEach(function (k) {
    var e = f.image[k], px = { 0: 200, 5: 100, 47: 37 }[k], im = M.markerImage(Number(k), px), s = 0, h = 0;
    for (var j = 0; j < im.data.length; j++) { s += im.data[j]; h = (h + im.data[j] * j) % 1000003; }
    rec('markers', s === e.sum && im.w === e.shape[1] && im.h === e.shape[0] && h === e.hash, 'image ' + k);
  });
  f.crc8.forEach(function (x) { rec('markers', M.crc8(x[0]) === x[1] && M.crc16(x[0]) === x[2], 'crc'); });
  f.meta.forEach(function (x) {
    rec('markers', diff(x.bits, M.encodeMeta(x.args[0], x.args[1], x.args[2]), 'meta enc') === '', 'meta enc');
    rec('markers', diff(x.dec, M.decodeMeta(x.bits2), 'meta') === '', 'meta dec ' + JSON.stringify(x.dec));
  });
  f.serial.forEach(function (x) {
    rec('markers', diff(x.bits, M.encodeSerial(x.serial), 'ser') === '', 'serial enc');
    rec('markers', M.decodeSerial(x.bits2) === x.dec, 'serial dec');
  });
  f.headers.forEach(function (x) {
    var h = x.h;
    rec('markers', M.headerFormKey(h) === x.form_key && diff(x.labels, M.headerChoiceLabels(h), 'l') === '', 'form_key/labels');
    var hh = M.Header(h.questions, h.used, h.choices, h.id_digits, h.subject_digits, h.pages, h.page, h.labels, h.bubble, h.version);
    rec('markers', diff(x.rows, M.encodeHeader(hh), 'rows') === '', 'header encode');
    rec('markers', diff(x.dec, M.decodeHeader(x.rows2), 'dec') === '', 'header decode');
    rec('markers', diff(x.dec_data, M.decodeHeaderData([].concat.apply([], x.rows.map(function (r) { return r.slice(4, r.length - 4); }))), 'dd') === '', 'header data decode');
  });
  f.labels.forEach(function (x) { rec('markers', diff(x[2], M.choiceLabels(x[0], x[1]), 'lab') === '', 'labels'); });
  f.style.forEach(function (x) { rec('markers', M.labelStyle(x[0]) === x[1], 'style ' + JSON.stringify(x[0])); });
  f.form_keys.forEach(function (x) { rec('markers', diff(x.header, O.layout.headerFromFormKey(x.key, 0), 'fk') === '', 'form key'); });
}

// ---------------------------------------------------------------- layout
function testLayouts() {
  var L = O.layout, f = load('layouts.json');
  f.layouts.forEach(function (it, i) {
    var name = 'layout#' + i + ' ' + JSON.stringify(it.template).slice(0, 80);
    var built = null, err = null;
    try { built = L.buildLayout(L.Template(it.template)); } catch (e) { err = e; }
    if (it.error) { rec('layout', err !== null, name + ' should throw'); return; }
    if (err) { rec('layout', false, name + ' threw ' + err); return; }
    var d = diff(it.layout, plain(built), 'layout'); rec('layout', d === '', name + ' ' + d);
    rec('layout', L.describe(built) === it.describe, name + ' describe');
    var hs = built.pages.map(function (p) { return L.layoutHeader(built, p.index); });
    d = diff(it.headers, hs, 'headers'); rec('layout', d === '', name + ' ' + d);
    var fh = L.layoutFromHeader(hs[0]);
    d = diff(it.from_header, plain(fh), 'from_header'); rec('layout', d === '', name + ' ' + d);
    rec('layout', fh === L.layoutForHeader(hs[0]) || L.layoutForHeader(hs[0]).form_key === it.form_key, name + ' form_key');
    d = diff(it.layout, plain(L.layoutFromDict(it.layout)), 'fromDict'); rec('layout', d === '', name + ' fromDict ' + d);
  });
  var r = f.rects;
  [0, 1, 2].forEach(function (p) { rec('layout', diff(r.marker[p], L.markerRects(p), 'm') === '', 'marker rects'); });
  rec('layout', diff(r.meta, L.metaCellRects(), 'meta') === '', 'meta rects');
  rec('layout', diff(r.header, L.headerCellRects(), 'hdr') === '', 'header rects');
  rec('layout', diff(r.ticks, L.trackTickRects(), 'ticks') === '', 'tick rects');
  rec('layout', diff(r.centers, L.trackTickCenters(), 'cent') === '', 'tick centres');
  rec('layout', diff(r.serial, L.serialCellRects(), 'ser') === '', 'serial rects');
  rec('layout', diff(r.serial_label, L.serialLabel(37), 'sl') === '', 'serial label');
}

// ---------------------------------------------------------------- decide
function testDecide() {
  var cases = load('decide.json');
  cases.forEach(function (c, i) {
    var raw = loadRaw(c.raw), d = O.decide.decidePage(raw);
    var df = diff(c.dec, plain(d), 'dec');
    rec('decide', df === '', 'case ' + i + ' ' + df);
    rec('decide', JSON.stringify(c.sig) === O.decide.decisionSig(d.answers, d.student_id, d.subject), 'sig ' + i);
  });
}

// ---------------------------------------------------------------- identity / revision
function testIdentityRevision() {
  var f = load('identity_revision.json');
  f.identity.forEach(function (x, i) {
    var r = loadResult(x.r), I = O.identity;
    var got = { serial_of: I.serialOf(r), id_text: I.studentIdText(r), subject_text: I.subjectText(r), sheet_key: I.sheetKey(r), severity: I.severity(r),
      needs_review: T.needsReview(r), id_guessed: T.idGuessed(r), subject_guessed: T.subjectGuessed(r), fmt: I.formatSerial(r.serial), key: I.makeSheetKey(5, '12?4', '0') };
    if (r.status === 'ok') got.doubtful = O.revision.doubtful(r); else got.doubtful = null;
    var exp = { serial_of: x.serial_of, id_text: x.id_text, subject_text: x.subject_text, sheet_key: x.sheet_key, severity: x.severity,
      needs_review: x.needs_review, id_guessed: x.id_guessed, subject_guessed: x.subject_guessed, fmt: x.fmt, key: x.key, doubtful: x.doubtful };
    var d = diff(exp, got, 'identity'); rec('identity', d === '', 'case ' + i + ' ' + d);
  });
  f.revision.forEach(function (x, i) {
    var m = O.revision.merge(loadResult(x.base), loadResult(x.now));
    var d = diff(x.merged, plainResultNoRaw(m[0]), 'merged');
    rec('revision', d === '' && diff(x.changed, m[1], 'ch') === '' && diff(x.contra, m[2], 'co') === '', 'case ' + i + ' ' + d);
  });
}

// ---------------------------------------------------------------- fusion + stream
var pool = null;
function poolResult(i) { return pool[i]; }

function testFusion(fx) {
  var F = O.fusion;
  fx.fusion.forEach(function (steps, ti) {
    var track = null, n = 0;
    steps.forEach(function (st, si) {
      var r = poolResult(st.idx), name = 'trace ' + ti + ' step ' + si, got = {};
      got.idx = st.idx; got.trust = F.trustworthy(r);
      got.quad = r.status === 'ok' ? plain(F.sheetQuad(r)) : null;
      if (r.status === 'ok' && got.trust) {
        if (track !== null) { got.incompat = track.incompatible(r); if (got.incompat) track = null; }
        if (track === null) track = new F.Track(si + 1, r, 0.0); else track.add(r, 0.0, null);
        got.fused = plainRaw(track.fusedRaw()); got.dec = plain(track.decisions()); got.serial = track.serial();
        got.idu = track.idUnreadable(); got.sbu = track.subjectUnreadable(); got.n = track.n;
        got.applied = plainEventResult(track.apply(r));
      }
      var d = diff(st, got, 'fusion');
      rec('fusion', d === '', name + ' ' + d);
    });
  });
}

function testStream(fx) {
  fx.streams.forEach(function (s) {
    var p = Object.assign({}, s.params), sess = new O.stream.StreamSession(p), bad = '';
    for (var i = 0; i < s.frames.length && !bad; i++) {
      var ev = plainEvent(sess.feed(poolResult(s.frames[i]), s.ts[i], i)), d = diff(s.events[i], ev, 'event');
      rec('stream', d === '', s.name + ' frame ' + i + ' ' + d);
      if (d) bad = d;
    }
    if (bad) return;
    var fin = sess.finish(s.ts[s.ts.length - 1]), fe = null;
    if (fin) { var c = Object.assign({}, fin); c.result = plainEventResult(fin.result); fe = plain(c); }
    var tail = { finish: fe, missed: plain(sess.missed()), missed_final: plain(sess.missed(true)), unread: plain(sess.unreadEpisodes()),
      unread_final: plain(sess.unreadEpisodes(true)), state: sess.state };
    var d2 = diff({ finish: s.finish, missed: s.missed, missed_final: s.missed_final, unread: s.unread, unread_final: s.unread_final, state: s.state }, tail, 'tail');
    rec('stream', d2 === '', s.name + ' tail ' + d2);
  });
}

function main() {
  var areas = [['markers', testMarkers], ['layout', testLayouts], ['decide', testDecide], ['identity', testIdentityRevision]];
  areas.forEach(function (a) { try { a[1](); } catch (e) { rec(a[0], false, 'exception ' + e + ' ' + (e.stack || '')); } });
  try {
    var fx = load('stream.json');
    pool = fx.pool.map(loadResult);
    testFusion(fx);
    testStream(fx);
  } catch (e) { rec('stream', false, 'exception ' + e + ' ' + (e.stack || '')); }
  var fails = 0;
  Object.keys(stats).forEach(function (k) { print('RESULT ' + k + ' pass ' + stats[k].pass + ' fail ' + stats[k].fail); fails += stats[k].fail; });
  print(fails ? 'FAILED ' + fails : 'ALL PASSED');
}
main();
})(arguments);
