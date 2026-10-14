// omr/layout.py: template params -> Layout.  All coordinates in mm, page-relative, origin top-left.
// The Layout is the single source of truth for both the PDF renderer and the recognizer.  Rect = [x, y, w, h].
// Text y is the baseline.  Layout objects are plain objects with the Python dataclass field names (snake_case).
(function (O) {
'use strict';
var mk = O.markers, pyRound = O.util.pyRound;

// ------------------------------------------------------------------ design constants
var PAGE_W = 210.0, PAGE_H = 297.0;
var MARGIN = 8.0;
var MARKER_MM = 16.0;
var QUIET_MM = 3.0;
var META_CELL_W = 3.6, META_CELL_H = 4.5, META_Y = 9.0;     // legacy single-row meta strip (form id only)
var HEADER_CELL_W = 3.6, HEADER_ROW_H = 4.0;                // self-describing header: 2 rows
var HEADER_ROW_Y = [8.0, 13.5];                             // 1.5 mm white gap between rows (and above / below): local paper level
var TITLE_Y = 23.5;
var SUBJ_GAP = 7.0;                                         // ID grid -> subject grid (room for its row labels)
var SERIAL_CELL_W = 3.6, SERIAL_CELL_H = 4.5, SERIAL_Y = 275.0;   // footer strip, right end kept clear of the BR marker
var SERIAL_CLEARANCE = 6.0;   // mm between the strip / "No." label and the marker squares (ink spread, blur)
var INFO_TOP = 30.0;
var PAD = 3.0;
var FRAME_INSET = 0.25;       // keeps frame strokes inside the margin
// ---- timing tracks (header version 2): rows of small black squares along the four edges, outside the frames.  A
// folded / curled sheet is not flat: the recognizer measures where each tick really is and bends the page back.
var TRACK_PITCH = 6.0;                  // mm between tick centres
var TRACK_LEN = 3.0, TRACK_LR_W = 2.4, TRACK_TB_H = 2.4;   // tick size along the track, across it (left / right), across it (top / bottom)
var TRACK_EDGE_LR = 6.0, TRACK_EDGE_TB = 4.4;              // distance of the ticks from the paper edge (printers clip ~3 mm)
var TRACK_N_LR = 40, TRACK_N_TB = 26;                      // ticks per left / right edge, per top / bottom edge
var FRAME_INSET_TRACKS = 1.4;           // frames start this far inside the margin when tracks are printed (clear of the left / right ticks)
var ID_WRITEIN_H = 7.0;
var ID_ROW_LABEL_W = 5.0;
var NAME_BOX_H = 25.0;
var ANS_HEADER_H = 6.0;
var ANS_GAP = 3.0;
var QNUM_W = 8.0;
var MIN_PITCH = 5.5;
var MAX_PITCH = 7.5;
var MAX_BUBBLE = 6.0;
var BAND_ROWS = 5;
var MAX_CHOICES = 10, MAX_ID_DIGITS = 12;

// Template (omr/layout.py:44): what a form looks like.  Everything that decides geometry is encoded in the sheet's header, so
// a sheet can be read without knowing the template in advance; the texts (title, labels, instruction) are only printed.
function Template(o) {
  var t = {
    questions: 45, choices: 5, id_digits: 8, pages: null, title: '', choice_labels: null, min_bubble_mm: 4.0, tracks: true,
    used: null, subject_digits: 2, label_id: '수험번호', label_name: '성명', label_subject: '과목코드',
    instruction: '보기 원 안을 진하게 완전히 칠하세요. 고칠 때는 깨끗이 지우세요.'
  };
  if (o) for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) t[k] = o[k];
  return t;
}

function Text(text, x, y, sizePt, align, gray, kind) {
  return { text: text, x: x, y: y, size_pt: sizePt, align: align === undefined ? 'l' : align, gray: gray === undefined ? 0.0 : gray,
    kind: kind === undefined ? '' : kind };
}

// Python float floor division (a // b)
function floorDiv(a, b) {
  var mod = a % b, div = (a - mod) / b;
  if (mod !== 0) { if ((b < 0) !== (mod < 0)) { mod += b; div -= 1; } }
  var fl;
  if (div !== 0) { fl = Math.floor(div); if (div - fl > 0.5) fl += 1; } else fl = 0;
  return fl;
}

function _r(v) { return pyRound(Number(v), 3); }
function _rect() { var out = []; for (var i = 0; i < arguments.length; i++) out.push(_r(arguments[i])); return out; }

// Corner name -> [x, y, w, h] (mm) of the marker's black square for page index p.  Marker positions are fixed design
// constants (independent of the template), so a recognizer can use them before it knows which form it is looking at.
function markerRects(p) {                // omr/layout.py:206
  var far = PAGE_W - MARGIN - MARKER_MM, low = PAGE_H - MARGIN - MARKER_MM;
  var pos = { TL: [MARGIN, MARGIN], TR: [far, MARGIN], BR: [far, low], BL: [MARGIN, low] };
  var out = {};
  ['TL', 'TR', 'BR', 'BL'].forEach(function (c) { out[c] = [_r(pos[c][0]), _r(pos[c][1]), MARKER_MM, MARKER_MM]; });
  return out;
}

function metaCellRects() {               // omr/layout.py:218
  var n = mk.META_CELLS, mx = (PAGE_W - META_CELL_W * n) / 2, out = [];
  for (var i = 0; i < n; i++) out.push(_rect(mx + i * META_CELL_W, META_Y, META_CELL_W, META_CELL_H));
  return out;
}

// [row][cell] -> [x, y, w, h] (mm) of the self-describing header (template independent)
function headerCellRects() {             // omr/layout.py:225
  var n = mk.HEADER_ROW_CELLS, x0 = (PAGE_W - HEADER_CELL_W * n) / 2;
  return HEADER_ROW_Y.map(function (y) {
    var row = [];
    for (var i = 0; i < n; i++) row.push(_rect(x0 + i * HEADER_CELL_W, y, HEADER_CELL_W, HEADER_ROW_H));
    return row;
  });
}

// Side -> [[x, y, w, h], ...] (mm) of the timing ticks (template independent; used when the header says version >= 2)
function trackTickRects() {              // omr/layout.py:232
  var out = {}, ys = [], xs = [], k;
  for (k = 0; k < TRACK_N_LR; k++) ys.push((PAGE_H - (TRACK_N_LR - 1) * TRACK_PITCH) / 2 + k * TRACK_PITCH);
  for (k = 0; k < TRACK_N_TB; k++) xs.push((PAGE_W - (TRACK_N_TB - 1) * TRACK_PITCH) / 2 + k * TRACK_PITCH);
  [['L', TRACK_EDGE_LR], ['R', PAGE_W - TRACK_EDGE_LR - TRACK_LR_W]].forEach(function (s) {
    out[s[0]] = ys.map(function (y) { return _rect(s[1], y - TRACK_LEN / 2, TRACK_LR_W, TRACK_LEN); });
  });
  [['T', TRACK_EDGE_TB], ['B', PAGE_H - TRACK_EDGE_TB - TRACK_TB_H]].forEach(function (s) {
    out[s[0]] = xs.map(function (x) { return _rect(x - TRACK_LEN / 2, s[1], TRACK_LEN, TRACK_TB_H); });
  });
  return out;
}

// Side -> array of [cx, cy] tick centres (mm)  (Python: (n, 2) array)
function trackTickCenters() {            // omr/layout.py:245
  var rs = trackTickRects(), out = {};
  Object.keys(rs).forEach(function (s) { out[s] = rs[s].map(function (r) { return [r[0] + r[2] / 2, r[1] + r[3] / 2]; }); });
  return out;
}

function serialCellRects() {             // omr/layout.py:251
  var n = mk.SERIAL_CELLS, x1 = PAGE_W - MARGIN - MARKER_MM - SERIAL_CLEARANCE, x0 = x1 - n * SERIAL_CELL_W, out = [];
  for (var i = 0; i < n; i++) out.push(_rect(x0 + i * SERIAL_CELL_W, SERIAL_Y, SERIAL_CELL_W, SERIAL_CELL_H));
  return out;
}

function pad4(n) { return ('0000' + n).slice(-4); }

function serialLabel(serial) {           // omr/layout.py:258
  var x1 = PAGE_W - MARGIN - MARKER_MM - SERIAL_CLEARANCE;
  return Text('No. ' + pad4(serial), x1, 285.0, 9, 'r', 0.0, 'serial');
}

// ------------------------------------------------------------------ geometry
function idGeometry(t) {                 // omr/layout.py:270 -> [dia, pitch_x, pitch_y]
  var dia = Math.max(t.min_bubble_mm, 4.2);
  return [dia, Math.max(6.2, dia * 1.45), Math.max(MIN_PITCH, dia * 1.3)];
}

function infoBottom(t) {
  var py = idGeometry(t)[2];
  return INFO_TOP + PAD + 5 + ID_WRITEIN_H + 2 + 10 * py + PAD;
}

function frameLeft(t) { return MARGIN + (t.tracks ? FRAME_INSET_TRACKS : FRAME_INSET); }

function answerArea(t) {
  var top = infoBottom(t) + 3.0, bottom = PAGE_H - MARGIN - MARKER_MM - QUIET_MM, left = frameLeft(t);
  return [left, top, PAGE_W - 2 * left, bottom - top];
}

// Best [ncols, rows, pitch_x, pitch_y, col_w] for q questions on one page, or null
function chooseGrid(t, q) {              // omr/layout.py:290
  var area = answerArea(t), aw = area[2], ah = area[3];
  var availH = ah - ANS_HEADER_H - 2.0, best = null;
  for (var ncols = 1; ncols < (q > 0 ? q + 1 : 2); ncols++) {
    var colW = (aw - (ncols - 1) * ANS_GAP) / ncols;
    var px = Math.min(MAX_PITCH, (colW - QNUM_W) / t.choices);
    var rows = Math.ceil(q / ncols);
    var py = Math.min(MAX_PITCH, availH / rows);
    if (px < MIN_PITCH || py < MIN_PITCH) continue;
    var score = Math.min(px, py);
    if (best === null || score > best[0] + 1e-9) best = [score, ncols, rows, px, py, colW];
  }
  return best ? best.slice(1) : null;
}

function pageCapacity(t) {               // omr/layout.py:307: max questions on one page at minimum pitch
  var area = answerArea(t), aw = area[2], ah = area[3];
  var availH = ah - ANS_HEADER_H - 2.0;
  var rows = floorDiv(availH, MIN_PITCH);
  var minCol = QNUM_W + t.choices * MIN_PITCH;
  var ncols = floorDiv(aw + ANS_GAP, minCol + ANS_GAP);
  return Math.max(0, rows * ncols);
}

function resolveSplit(t) {               // omr/layout.py:318: printed questions per page: an even split over the pages
  var Q = t.questions, cap = pageCapacity(t), n;
  if (cap < 1) throw new Error('no room for any question at minimum bubble size');
  if (t.pages !== null && t.pages !== undefined) {
    n = t.pages;
    if (n < 1 || n > Q) throw new Error('invalid page count');
    if (Math.ceil(Q / n) > cap) throw new Error(Q + ' questions in ' + n + ' pages exceeds capacity ' + cap + '/page');
  } else {
    n = Math.ceil(Q / cap);   // automatic split
  }
  var base = Math.floor(Q / n), extra = Q % n, out = [];
  for (var i = 0; i < n; i++) out.push(base + (i < extra ? 1 : 0));
  return out;
}

function bubbleClass(minBubbleMm) {      // omr/layout.py:337
  var k = (minBubbleMm - mk.BUBBLE_BASE_MM) / mk.BUBBLE_STEP_MM;
  if (Math.abs(k - Math.round(k)) > 1e-6 || !(Math.round(k) >= 0 && Math.round(k) < 8))
    throw new Error('min_bubble_mm must be ' + mk.BUBBLE_BASE_MM + ' + k*' + mk.BUBBLE_STEP_MM + ' (k = 0..7)');
  return Math.round(k);
}

function textWidthMm(s, sizePt) {        // omr/layout.py:344
  var em = sizePt * 0.3528, w = 0;
  Array.from(s).forEach(function (c) { w += em * (c.codePointAt(0) > 0x2E80 ? 1.0 : 0.55); });
  return w;
}

function validate(t) {                   // omr/layout.py:349
  if (t.questions < 1) throw new Error('questions must be >= 1');
  if (!(t.choices >= 2 && t.choices <= MAX_CHOICES)) throw new Error('choices must be 2..' + MAX_CHOICES);
  if (!(t.id_digits >= 1 && t.id_digits <= MAX_ID_DIGITS)) throw new Error('id_digits must be 1..' + MAX_ID_DIGITS);
  if (t.choice_labels !== null && t.choice_labels !== undefined && t.choice_labels.length !== t.choices) throw new Error('choice_labels length must equal choices');
  bubbleClass(t.min_bubble_mm);
  if (t.questions >= (1 << 10)) throw new Error('questions must be < 1024');
  if (t.used !== null && t.used !== undefined && !(t.used >= 1 && t.used <= t.questions)) throw new Error('used must be 1..questions');
  if (!(t.subject_digits >= 0 && t.subject_digits <= 3)) throw new Error('subject_digits must be 0..3');
  if (t.choice_labels !== null && t.choice_labels !== undefined && mk.labelStyle(Array.from(t.choice_labels)) === null)
    throw new Error('choice_labels must be 1 2 3.., A B C.., a b c.. or 가 나 다.. (the header stores only the style)');
}

// The header of page `page` of layout L (omr/layout.py Layout.header)
function layoutHeader(L, page) {
  var t = L.template;
  return mk.Header(t.questions, t.used || t.questions, t.choices, t.id_digits, t.subject_digits, L.total_pages,
    page, mk.labelStyle(t.choice_labels), bubbleClass(t.min_bubble_mm), L.version || mk.HEADER_VERSION);
}

function buildLayout(template) {         // omr/layout.py:372
  var t = Template(template);
  if (typeof t.choice_labels === 'string') t.choice_labels = Array.from(t.choice_labels);
  validate(t);
  if (t.choice_labels === null || t.choice_labels === undefined) {
    t.choice_labels = [];
    for (var i = 0; i < t.choices; i++) t.choice_labels.push(String(i + 1));
  }
  var split = resolveSplit(t), total = split.length;
  if (total > mk.MAX_PAGES) throw new Error(total + ' pages needed; max is ' + mk.MAX_PAGES);
  var used = t.used || t.questions, sumButLast = 0;
  for (i = 0; i < total - 1; i++) sumButLast += split[i];
  if (used <= sumButLast) throw new Error('only ' + used + ' of ' + t.questions + ' questions used: page ' + total + ' would be empty (use fewer pages)');
  t.pages = total;
  var pages = [], first = 1, usedPP = [];
  for (var p = 0; p < split.length; p++) {
    var nq = split[p], nUsed = Math.max(0, Math.min(nq, used - (first - 1)));
    pages.push(buildPage(t, p, total, first, nq, nUsed));
    usedPP.push(nUsed);
    first += nq;
  }
  var L = { page_w: PAGE_W, page_h: PAGE_H, margin: MARGIN, form_id: null, total_pages: total, questions_per_page: usedPP,
    stroke_mm: 0.25, template: t, pages: pages, form_key: '', version: t.tracks ? 2 : 1 };
  L.form_key = mk.headerFormKey(layoutHeader(L, 0));
  pages.forEach(function (pg) {    // the header cells are only known once the layout exists
    var rows = mk.encodeHeader(layoutHeader(L, pg.index)), hr = headerCellRects(), cells = [];
    for (var r = 0; r < hr.length; r++) for (var c = 0; c < hr[r].length; c++) cells.push({ rect: hr[r][c], bit: rows[r][c] });
    pg.meta_cells = cells;
  });
  return L;
}

// The layout a self-describing sheet describes (texts are not part of the header: no title)
function layoutFromHeader(h) {           // omr/layout.py:402
  return buildLayout(Template({ questions: h.questions, choices: h.choices, id_digits: h.id_digits, pages: h.pages,
    choice_labels: mk.headerChoiceLabels(h), used: h.used, subject_digits: h.subject_digits,
    min_bubble_mm: mk.BUBBLE_BASE_MM + mk.BUBBLE_STEP_MM * h.bubble, title: '', tracks: h.version >= 2 }));
}

var _layoutCache = {};
// omr/recognize.py:570 layout_for_header: cached by form_key (the cached Layout is shared: treat it as read-only)
function layoutForHeader(h) {
  var key = mk.headerFormKey(h), L = _layoutCache[key];
  if (!L) L = _layoutCache[key] = layoutFromHeader(h);
  return L;
}

// Inverse of headerFormKey ("Q100U80C5D8S2P2L0B0V1") for self-describing sheets; throws otherwise
function headerFromFormKey(formKey, page) {   // omr/layout.py:411
  var m = /^Q(\d+)U(\d+)C(\d+)D(\d+)S(\d+)P(\d+)L(\d+)B(\d+)V(\d+)$/.exec(formKey || '');
  if (m === null) throw new Error('not a self-describing form key: ' + JSON.stringify(formKey));
  var v = m.slice(1).map(Number);
  var h = mk.Header(v[0], v[1], v[2], v[3], v[4], v[5], page || 0, v[6], v[7], v[8]);
  mk.validateHeader(h);
  return h;
}

function describe(L) {                   // omr/layout.py:422: one-line human description of a form (Korean)
  var t = L.template, used = t.used || t.questions;
  var q = used === t.questions ? used + '문항' : t.questions + '문항 양식 중 ' + used + '문항';
  return q + ' · ' + t.choices + '지선다 · ' + L.total_pages + '쪽 · 수험번호 ' + t.id_digits +
    '자리' + (t.subject_digits ? ' · 과목코드 ' + t.subject_digits + '자리' : '');
}

// Layout.from_dict: layouts saved by older versions load (missing fields take defaults, legacy sheets are fixed up).  Input is
// a plain object as produced by JSON.parse(layout JSON) / Python Layout.to_dict(); returns a normalised deep copy.
function layoutFromDict(d) {
  var L = JSON.parse(JSON.stringify(d));
  L.template = Template(L.template);
  L.version = L.version || 0;
  L.form_key = L.form_key || '';
  if (L.form_id === undefined) L.form_id = null;
  L.pages.forEach(function (pg) {
    if (pg.subject_grid === undefined) pg.subject_grid = null;
    if (pg.subject_label === undefined) pg.subject_label = null;
    if (pg.ticks === undefined) pg.ticks = [];
  });
  if (L.version === 0) {      // legacy layout: no subject code, form identified by its 16-bit id
    L.template.subject_digits = 0;
    L.form_key = L.form_key || 'F' + ('0000' + L.form_id.toString(16).toUpperCase()).slice(-4);
  }
  if (L.version < 2) L.template.tracks = false;   // saved before timing tracks existed
  return L;
}

function idColumns(n, gx0, writeinTop, bubTop, ipx, ipy) {   // omr/layout.py:434
  var cols = [];
  for (var i = 0; i < n; i++) {
    var cx = gx0 + (i + 0.5) * ipx, bubbles = [];
    for (var v = 0; v < 10; v++) bubbles.push({ value: v, cx: _r(cx), cy: _r(bubTop + (v + 0.5) * ipy) });
    cols.push({ digit_index: i, writein: _rect(cx - ipx / 2 + 0.4, writeinTop, ipx - 0.8, ID_WRITEIN_H), bubbles: bubbles });
  }
  return cols;
}

function rowLabels(gx0, bubTop, ipy, kind) {
  var out = [];
  for (var v = 0; v < 10; v++) out.push(Text(String(v), _r(gx0 - 1.5), _r(bubTop + (v + 0.5) * ipy + 1.1), 7, 'r', 0.5, kind));
  return out;
}

// nq: printed question slots on this page (geometry); nUsed: how many of them are used (printed and read)
function buildPage(t, p, total, firstQ, nq, nUsed) {     // omr/layout.py:446
  var mspecs = [];
  mk.CORNERS.forEach(function (c) {
    var xy = markerRects(p)[c], x = xy[0], y = xy[1];
    mspecs.push({ id: mk.markerId(p, c), corner: c, rect: _rect(x, y, MARKER_MM, MARKER_MM),
      quiet: _rect(x - QUIET_MM, y - QUIET_MM, MARKER_MM + 2 * QUIET_MM, MARKER_MM + 2 * QUIET_MM) });
  });
  var hr = headerCellRects();
  var metaRect = _rect(hr[0][0][0], hr[0][0][1], HEADER_CELL_W * mk.HEADER_ROW_CELLS, hr[hr.length - 1][0][1] + HEADER_ROW_H - hr[0][0][1]);

  var texts = [];
  var freeW = PAGE_W - 2 * (MARGIN + MARKER_MM + QUIET_MM);     // title (shrunk to fit between markers)
  if (t.title) {
    var size = 13.0;
    while (size > 7 && textWidthMm(t.title, size) > freeW) size -= 0.5;
    texts.push(Text(t.title, PAGE_W / 2, TITLE_Y, size, 'c', 0.0, 'title'));
  }
  var fx = MARGIN + MARKER_MM + QUIET_MM, used = t.used || t.questions;
  var qtxt = 'Q1-' + used + (used !== t.questions ? ' of ' + t.questions : '');
  texts.push(Text('page ' + (p + 1) + ' / ' + total + '  |  ' + qtxt, fx, 280.0, 8, 'l', 0.3, 'footer'));
  if (t.instruction) texts.push(Text(t.instruction, fx, 285.0, 7, 'l', 0.4, 'instruction'));

  // ---- student info area: ID grid, subject-code grid, name box
  var g3 = idGeometry(t), idia = g3[0], ipx = g3[1], ipy = g3[2];
  var iBottom = infoBottom(t), left = frameLeft(t);
  var infoBox = _rect(left, INFO_TOP, PAGE_W - 2 * left, iBottom - INFO_TOP);
  var labelY = INFO_TOP + PAD + 3.5;
  var idLabel = Text(t.label_id, left + PAD, labelY, 9, 'l', 0.0, 'id_label');
  var writeinTop = INFO_TOP + PAD + 5, bubTop = writeinTop + ID_WRITEIN_H + 2, gx0 = left + PAD + ID_ROW_LABEL_W;
  var grid = { bubble_diameter: _r(idia), columns: idColumns(t.id_digits, gx0, writeinTop, bubTop, ipx, ipy),
    row_labels: rowLabels(gx0, bubTop, ipy, 'id_row') };
  var nx = gx0 + t.id_digits * ipx + 8, subjectGrid = null, subjectLabel = null;
  if (t.subject_digits) {
    var sx0 = gx0 + t.id_digits * ipx + SUBJ_GAP;
    subjectGrid = { bubble_diameter: _r(idia), columns: idColumns(t.subject_digits, sx0, writeinTop, bubTop, ipx, ipy),
      row_labels: rowLabels(sx0, bubTop, ipy, 'subject_row') };
    subjectLabel = Text(t.label_subject, _r(sx0 - ID_ROW_LABEL_W + 1.0), labelY, 9, 'l', 0.0, 'subject_label');
    nx = Math.max(sx0 + t.subject_digits * ipx, sx0 - ID_ROW_LABEL_W + 1.0 + textWidthMm(t.label_subject, 9)) + 8;
  }
  var nw = PAGE_W - left - PAD - nx;
  if (nw < 25) throw new Error('no room for the name box (too many ID / subject digits)');
  var nameBox = _rect(nx, writeinTop, nw, NAME_BOX_H);
  var nameLabel = Text(t.label_name, _r(nx), _r(labelY), 9, 'l', 0.0, 'name_label');

  // ---- answer area (geometry from all nq slots; only the first nUsed are printed and read)
  var abox = answerArea(t), cg = chooseGrid(t, nq);
  var ncols = cg[0], rows = cg[1], px = cg[2], py = cg[3], colW = cg[4];
  var dia = Math.max(t.min_bubble_mm, Math.min(MAX_BUBBLE, 0.7 * Math.min(px, py)));
  var columns = [], q = 0, yRows = abox[1] + ANS_HEADER_H + 2.0;
  for (var c = 0; c < ncols; c++) {
    var cx0 = abox[0] + c * (colW + ANS_GAP), inCol = Math.min(rows, nq - q, nUsed - q);
    if (inCol <= 0) break;
    var headers = [];
    for (var k = 0; k < t.choices; k++)
      headers.push(Text(t.choice_labels[k], _r(cx0 + QNUM_W + (k + 0.5) * px), _r(abox[1] + 4.2), 8, 'c', 0.35, 'choice_label'));
    var bands = [], qs = [];
    for (var r = 0; r < inCol; r++) {
      if (Math.floor(r / BAND_ROWS) % 2 === 1) bands.push({ rect: _rect(cx0, yRows + r * py, colW, py), gray: 0.95 });
      var cy = yRows + (r + 0.5) * py, number = firstQ + q, bubbles = [];
      for (k = 0; k < t.choices; k++) bubbles.push({ choice: k, label: t.choice_labels[k], cx: _r(cx0 + QNUM_W + (k + 0.5) * px), cy: _r(cy) });
      qs.push({ number: number, num_text: Text(String(number), _r(cx0 + QNUM_W - 2.0), _r(cy + 1.2), 9, 'r', 0.0, 'qnum'), bubbles: bubbles });
      q += 1;
    }
    columns.push({ rect: _rect(cx0, abox[1], colW, ANS_HEADER_H + 2.0 + rows * py), headers: headers, bands: bands, questions: qs });
  }
  var ticks = [];
  if (t.tracks) {
    var tr = trackTickRects();
    Object.keys(tr).forEach(function (side) { tr[side].forEach(function (rc, i) { ticks.push({ side: side, index: i, rect: rc }); }); });
  }
  return { index: p, markers: mspecs, meta_rect: metaRect, meta_cells: [], info_box: infoBox, id_grid: grid, id_label: idLabel,
    name_box: _rect.apply(null, nameBox), name_label: nameLabel, answer_box: _rect.apply(null, abox), bubble_diameter: _r(dia),
    first_question: firstQ, n_questions: nUsed, columns: columns, texts: texts, subject_grid: subjectGrid, subject_label: subjectLabel,
    ticks: ticks };
}

O.layout = {
  PAGE_W: PAGE_W, PAGE_H: PAGE_H, MARGIN: MARGIN, MARKER_MM: MARKER_MM, QUIET_MM: QUIET_MM, META_CELL_W: META_CELL_W, META_CELL_H: META_CELL_H,
  META_Y: META_Y, HEADER_CELL_W: HEADER_CELL_W, HEADER_ROW_H: HEADER_ROW_H, HEADER_ROW_Y: HEADER_ROW_Y, TITLE_Y: TITLE_Y,
  SERIAL_CELL_W: SERIAL_CELL_W, SERIAL_CELL_H: SERIAL_CELL_H, SERIAL_Y: SERIAL_Y, SERIAL_CLEARANCE: SERIAL_CLEARANCE, INFO_TOP: INFO_TOP,
  TRACK_PITCH: TRACK_PITCH, TRACK_LEN: TRACK_LEN, TRACK_LR_W: TRACK_LR_W, TRACK_TB_H: TRACK_TB_H, TRACK_EDGE_LR: TRACK_EDGE_LR,
  TRACK_EDGE_TB: TRACK_EDGE_TB, TRACK_N_LR: TRACK_N_LR, TRACK_N_TB: TRACK_N_TB, FRAME_INSET_TRACKS: FRAME_INSET_TRACKS,
  Template: Template, Text: Text, markerRects: markerRects, metaCellRects: metaCellRects, headerCellRects: headerCellRects,
  trackTickRects: trackTickRects, trackTickCenters: trackTickCenters, serialCellRects: serialCellRects, serialLabel: serialLabel,
  pageCapacity: pageCapacity, validate: validate, buildLayout: buildLayout, layoutHeader: layoutHeader,
  layoutFromHeader: layoutFromHeader, layoutForHeader: layoutForHeader, headerFromFormKey: headerFromFormKey, describe: describe,
  layoutFromDict: layoutFromDict, floorDiv: floorDiv
};
})(globalThis.OMR = globalThis.OMR || {});
