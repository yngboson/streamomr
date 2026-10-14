// omr/markers.py: ArUco corner markers (ids, bit patterns) and the metadata / serial / header bit-strip codecs.
(function (O) {
'use strict';
var U = O.util;

var ARUCO_DICT_NAME = 'DICT_4X4_50';
var MAX_PAGES = 12;                              // 4 ids per page, 50 ids in the dictionary -> 12 pages
var CORNERS = ['TL', 'TR', 'BR', 'BL'];          // id offsets 0..3

var META_FORM_BITS = 16, META_PAGE_BITS = 4, META_TOTAL_BITS = 4, META_CRC_BITS = 8;
var START_GUARD = [true, false, true, false];
var STOP_GUARD = [false, true, false, true];
var META_CELLS = START_GUARD.length + META_FORM_BITS + META_PAGE_BITS + META_TOTAL_BITS + META_CRC_BITS + STOP_GUARD.length;

// cv2.aruco DICT_4X4_50 marker patterns incl. the black border, exported from cv2 (generateImageMarker, 6x6, row-major, 1 = black)
var MARKER_BITS = ["111111101001110101111001111011111111", "111111111111100001101101101011111111", "111111111001111001111011100101111111", "111111101101101101110111110011111111", "111111110101110111101101100011111111", "111111110001101101100111100101111111", "111111101101100011111011100011111111", "111111100111110111100001111011111111", "111111100001100011100101101011111111", "111111100111100001110101110011111111", "111111100001101101101101111101111111", "111111111101111101101011110001111111", "111111111111100011101001110001111111", "111111111011101011111111100001111111", "111111111011110111101001111101111111", "111111111011110011111001100011111111", "111111110111110011110011110101111111", "111111110011110011111111111111111111", "111111110011100111110101100011111111", "111111110001110011101011100001111111", "111111101111110011101111101001111111", "111111101001111111111011101001111111", "111111100111100111100101110101111111", "111111100101100101101111111011111111", "111111100001100011110111110001111111", "111111101101110111110001111101111111", "111111101011100111100011110111111111", "111111101011110101110101110111111111", "111111111011111101111011111001111111", "111111111001110111110011100001111111", "111111110111110111111101110101111111", "111111110101110001101001111011111111", "111111101101100011100111100001111111", "111111100001111111100111101001111111", "111111111111101111101011100011111111", "111111111111101101111011101101111111", "111111111101101111110001110101111111", "111111111111110111100001100001111111", "111111111111100101100001110011111111", "111111111101100111110101101011111111", "111111111101110001111101101111111111", "111111111011101011111011101111111111", "111111111001111011101111100111111111", "111111111001101111101001111011111111", "111111111011110111100011101111111111", "111111111011100011100011101001111111", "111111111011100101111001100001111111", "111111110111101001110011110111111111", "111111110101111111111011100011111111", "111111110101111111111101111001111111"];

function markerId(page, corner) {        // omr/markers.py:27
  if (!(page >= 0 && page < MAX_PAGES)) throw new Error('page index ' + page + ' out of range 0..' + (MAX_PAGES - 1));
  return 4 * page + CORNERS.indexOf(corner);
}

function decodeMarkerId(mid) {           // omr/markers.py:33 -> [page, corner]
  if (!(mid >= 0 && mid < 4 * MAX_PAGES)) throw new Error('marker id ' + mid);
  return [Math.floor(mid / 4), CORNERS[mid % 4]];
}

// omr/markers.py:40: Marker pattern incl. black border as 6 rows of 6 booleans, true = black
function markerBits(mid) {
  var s = MARKER_BITS[mid], out = [];
  for (var r = 0; r < 6; r++) {
    var row = [];
    for (var c = 0; c < 6; c++) row.push(s.charAt(r * 6 + c) === '1');
    out.push(row);
  }
  return out;
}

// omr/markers.py:47: grey image {w, h, data: Uint8Array} of the marker with a white quiet zone (in marker cells)
function markerImage(mid, px, quietCells) {
  px = px === undefined ? 200 : px; quietCells = quietCells === undefined ? 1 : quietCells;
  var bits = markerBits(mid), k = bits.length;
  var cell = Math.max(1, Math.floor(px / k)), q = quietCells * cell, n = k * cell + 2 * q;
  var data = new Uint8Array(n * n).fill(255);
  for (var r = 0; r < k; r++) for (var c = 0; c < k; c++) {
    if (!bits[r][c]) continue;
    for (var y = 0; y < cell; y++) for (var x = 0; x < cell; x++) data[(q + r * cell + y) * n + q + c * cell + x] = 0;
  }
  return { w: n, h: n, data: data };
}

// ---------------------------------------------------------------- metadata strip
function crc8(data, poly) {              // omr/markers.py:68
  poly = poly === undefined ? 0x07 : poly;
  var crc = 0;
  for (var i = 0; i < data.length; i++) {
    crc ^= data[i];
    for (var j = 0; j < 8; j++) crc = (crc & 0x80) ? (((crc << 1) ^ poly) & 0xFF) : ((crc << 1) & 0xFF);
  }
  return crc;
}

function payloadBytes(formId, page, totalPages) {
  return [formId >> 8, formId & 0xFF, (page << 4) | totalPages];
}

function intBits(v, n) {                 // big-endian bits of v
  var out = [];
  for (var i = 0; i < n; i++) out.push(Math.floor(v / Math.pow(2, n - 1 - i)) % 2 === 1);
  return out;
}

function bitsInt(bits) {
  var v = 0;
  for (var i = 0; i < bits.length; i++) v = v * 2 + (bits[i] ? 1 : 0);
  return v;
}

function encodeMeta(formId, page, totalPages) {      // omr/markers.py:98
  if (!(formId >= 0 && formId < (1 << META_FORM_BITS))) throw new Error('form_id must be 16 bit');
  if (!(totalPages >= 1 && totalPages <= MAX_PAGES)) throw new Error('total_pages must be 1..' + MAX_PAGES);
  if (!(page >= 0 && page < totalPages)) throw new Error('page must be in 0..total_pages-1');
  var crc = crc8(payloadBytes(formId, page, totalPages));
  return START_GUARD.concat(intBits(formId, META_FORM_BITS), intBits(page, META_PAGE_BITS), intBits(totalPages, META_TOTAL_BITS),
    intBits(crc, META_CRC_BITS), STOP_GUARD);
}

function eqBits(a, b) {
  if (a.length !== b.length) return false;
  for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// omr/markers.py:110: Decode META_CELLS bits (true = black) -> {form_id, page, total_pages} or null on guard/checksum/range failure
function decodeMeta(bits) {
  bits = Array.prototype.map.call(bits, Boolean);
  if (bits.length !== META_CELLS) return null;
  var g = START_GUARD.length;
  if (!eqBits(bits.slice(0, g), START_GUARD) || !eqBits(bits.slice(bits.length - STOP_GUARD.length), STOP_GUARD)) return null;
  var i = g;
  var form = bitsInt(bits.slice(i, i + META_FORM_BITS)); i += META_FORM_BITS;
  var page = bitsInt(bits.slice(i, i + META_PAGE_BITS)); i += META_PAGE_BITS;
  var total = bitsInt(bits.slice(i, i + META_TOTAL_BITS)); i += META_TOTAL_BITS;
  var crc = bitsInt(bits.slice(i, i + META_CRC_BITS));
  if (crc !== crc8(payloadBytes(form, page, total))) return null;
  if (!(total >= 1 && total <= MAX_PAGES && page < total)) return null;
  return { form_id: form, page: page, total_pages: total };
}

// ---------------------------------------------------------------- serial strip (per printed copy)
var SERIAL_BITS = 12;
var SERIAL_MAX = (1 << SERIAL_BITS) - 1;     // 4095; 0 = "no serial"
var SERIAL_CRC_BITS = 8;
var SERIAL_CELLS = START_GUARD.length + SERIAL_BITS + SERIAL_CRC_BITS + STOP_GUARD.length;
var SERIAL_TAG = 0x5A;                       // domain separation from the meta strip's CRC

function serialPayload(serial) { return [SERIAL_TAG, serial >> 8, serial & 0xFF]; }

function encodeSerial(serial) {              // omr/markers.py:150
  if (!(serial >= 1 && serial <= SERIAL_MAX)) throw new Error('serial must be 1..' + SERIAL_MAX);
  return START_GUARD.concat(intBits(serial, SERIAL_BITS), intBits(crc8(serialPayload(serial)), SERIAL_CRC_BITS), STOP_GUARD);
}

function decodeSerial(bits) {                // omr/markers.py:157 -> serial or null
  bits = Array.prototype.map.call(bits, Boolean);
  if (bits.length !== SERIAL_CELLS) return null;
  var g = START_GUARD.length;
  if (!eqBits(bits.slice(0, g), START_GUARD) || !eqBits(bits.slice(bits.length - STOP_GUARD.length), STOP_GUARD)) return null;
  var v = bitsInt(bits.slice(g, g + SERIAL_BITS));
  var crc = bitsInt(bits.slice(g + SERIAL_BITS, g + SERIAL_BITS + SERIAL_CRC_BITS));
  if (v === 0 || crc !== crc8(serialPayload(v))) return null;
  return v;
}

// ---------------------------------------------------------------- self-describing header (format v1 / v2)
var HEADER_VERSIONS = [1, 2];      // 1: plain sheet; 2: also printed with timing tracks
var HEADER_VERSION = 2;            // what new sheets are printed as
var HEADER_START_GUARD = [true, true, false, true];
var HEADER_STOP_GUARD = [true, false, true, true];
var HEADER_ROWS = 2;
var HEADER_ROW_DATA = 32;
var HEADER_ROW_CELLS = HEADER_START_GUARD.length + HEADER_ROW_DATA + HEADER_STOP_GUARD.length;
var LABEL_STYLES = ['1', 'A', 'a', '가'];            // choice labels "1 2 3 ..", "A B C ..", "a b c ..", "가 나 다 .."
var HANGUL_LABELS = '가나다라마바사아자차';    // 가나다라마바사아자차
var BUBBLE_BASE_MM = 4.0, BUBBLE_STEP_MM = 0.25;         // min bubble = base + step * class (class 0..7)
var HEADER_FIELDS = [['version', 3], ['questions', 10], ['used', 10], ['choices', 4], ['id_digits', 4],
  ['subject_digits', 2], ['pages', 4], ['page', 4], ['labels', 2], ['bubble', 3], ['reserved', 2]];
var HEADER_PAYLOAD_BITS = HEADER_FIELDS.reduce(function (s, f) { return s + f[1]; }, 0);
if (HEADER_PAYLOAD_BITS + 16 !== HEADER_ROWS * HEADER_ROW_DATA) throw new Error('header layout');
// the whitening pattern is numpy default_rng(0x0A11CE).integers(0, 2, 64), exported from Python
var WHITEN = '0011000101001100101011011100011011100011001011010011011010010011'.split('').map(function (c) { return c === '1'; });
var HEADER_TAG = [0x4F, 0x4D, 0x52, 0x31];               // b"OMR1": domain separation for the CRC

// Header = plain object with the dataclass fields (questions, used, choices, id_digits, subject_digits, pages, page,
// labels, bubble, version); form_key / choice_labels are the functions headerFormKey / headerChoiceLabels below.
function Header(questions, used, choices, idDigits, subjectDigits, pages, page, labels, bubble, version) {
  return { questions: questions, used: used, choices: choices, id_digits: idDigits, subject_digits: subjectDigits,
    pages: pages, page: page, labels: labels === undefined ? 0 : labels, bubble: bubble === undefined ? 0 : bubble,
    version: version === undefined ? HEADER_VERSION : version };
}

// Identity of the printed form (everything but the page index): equal on every page of one form.
function headerFormKey(h) {
  return 'Q' + h.questions + 'U' + h.used + 'C' + h.choices + 'D' + h.id_digits + 'S' + h.subject_digits +
    'P' + h.pages + 'L' + h.labels + 'B' + h.bubble + 'V' + h.version;
}

function headerChoiceLabels(h) { return choiceLabels(h.labels, h.choices); }

function choiceLabels(style, n) {            // omr/markers.py:216
  var s = LABEL_STYLES[style], out = [], i;
  if (s === '1') { for (i = 0; i < n; i++) out.push(String(i + 1)); return out; }
  if (s === '가') return HANGUL_LABELS.slice(0, n).split('');
  for (i = 0; i < n; i++) out.push(String.fromCharCode(s.charCodeAt(0) + i));
  return out;
}

function labelStyle(labels) {                // omr/markers.py:226: style index or null
  for (var i = 0; i < LABEL_STYLES.length; i++) {
    if (labels.length <= (LABEL_STYLES[i] === '가' ? HANGUL_LABELS.length : 26) && eqBits(choiceLabels(i, labels.length), labels)) return i;
  }
  return null;
}

function crc16(data, poly, init) {           // CRC-16/CCITT-FALSE
  poly = poly === undefined ? 0x1021 : poly; init = init === undefined ? 0xFFFF : init;
  var crc = init;
  for (var i = 0; i < data.length; i++) {
    crc ^= data[i] << 8;
    for (var j = 0; j < 8; j++) crc = (crc & 0x8000) ? (((crc << 1) ^ poly) & 0xFFFF) : ((crc << 1) & 0xFFFF);
  }
  return crc;
}

function headerPayload(h) {
  var out = [];
  for (var i = 0; i < HEADER_FIELDS.length; i++) {
    var name = HEADER_FIELDS[i][0], n = HEADER_FIELDS[i][1], v = name === 'reserved' ? 0 : h[name];
    if (!(v >= 0 && v < Math.pow(2, n))) throw new Error('header field ' + name + '=' + v + ' does not fit in ' + n + ' bits');
    out = out.concat(intBits(v, n));
  }
  return out;
}

function bitsBytes(bits) {
  bits = bits.slice();
  while (bits.length % 8) bits.push(false);
  var out = [];
  for (var i = 0; i < bits.length; i += 8) out.push(bitsInt(bits.slice(i, i + 8)));
  return out;
}

function validateHeader(h) {                 // omr/markers.py:268
  if (HEADER_VERSIONS.indexOf(h.version) < 0) throw new Error('unsupported header version ' + h.version);
  if (!(h.used >= 1 && h.used <= h.questions)) throw new Error('used must be 1..questions');
  if (!(h.choices >= 2 && h.choices <= 10) || !(h.id_digits >= 1 && h.id_digits <= 12) || !(h.subject_digits >= 0 && h.subject_digits <= 3))
    throw new Error('choices / id_digits / subject_digits out of range');
  if (!(h.pages >= 1 && h.pages <= MAX_PAGES) || !(h.page >= 0 && h.page < h.pages)) throw new Error('pages / page out of range');
  if (h.labels >= LABEL_STYLES.length || !(h.bubble >= 0 && h.bubble < 8)) throw new Error('labels / bubble out of range');
}

function encodeHeader(h) {                   // omr/markers.py:281: rows of cells (true = black), guards included
  validateHeader(h);
  var pay = headerPayload(h);
  var data = pay.concat(intBits(crc16(HEADER_TAG.concat(bitsBytes(pay))), 16));
  data = data.map(function (b, i) { return b !== WHITEN[i]; });
  var rows = [];
  for (var r = 0; r < HEADER_ROWS; r++) rows.push(HEADER_START_GUARD.concat(data.slice(r * HEADER_ROW_DATA, (r + 1) * HEADER_ROW_DATA), HEADER_STOP_GUARD));
  return rows;
}

// omr/markers.py:292: 64 data bits (guards stripped, still whitened) -> Header, or null on CRC / range failure
function decodeHeaderData(data) {
  data = Array.prototype.map.call(data, Boolean);
  if (data.length !== HEADER_ROWS * HEADER_ROW_DATA) return null;
  data = data.map(function (b, i) { return b !== WHITEN[i]; });
  var pay = data.slice(0, HEADER_PAYLOAD_BITS), crc = bitsInt(data.slice(HEADER_PAYLOAD_BITS));
  if (crc !== crc16(HEADER_TAG.concat(bitsBytes(pay)))) return null;
  var vals = {}, i = 0;
  for (var k = 0; k < HEADER_FIELDS.length; k++) {
    vals[HEADER_FIELDS[k][0]] = bitsInt(pay.slice(i, i + HEADER_FIELDS[k][1]));
    i += HEADER_FIELDS[k][1];
  }
  if (vals.reserved !== 0) return null;
  var h = Header(vals.questions, vals.used, vals.choices, vals.id_digits, vals.subject_digits, vals.pages, vals.page,
    vals.labels, vals.bubble, vals.version);
  try { validateHeader(h); } catch (e) { return null; }
  return h;
}

function decodeHeader(rows) {                // omr/markers.py:310: rows of cells (guards included) -> Header or null
  var g0 = HEADER_START_GUARD.length, g1 = HEADER_STOP_GUARD.length, data = [];
  for (var r = 0; r < rows.length; r++) {
    var row = Array.prototype.map.call(rows[r], Boolean);
    if (row.length !== HEADER_ROW_CELLS || !eqBits(row.slice(0, g0), HEADER_START_GUARD) || !eqBits(row.slice(row.length - g1), HEADER_STOP_GUARD)) return null;
    data = data.concat(row.slice(g0, row.length - g1));
  }
  return decodeHeaderData(data);
}

O.markers = {
  ARUCO_DICT_NAME: ARUCO_DICT_NAME, MAX_PAGES: MAX_PAGES, CORNERS: CORNERS, META_CELLS: META_CELLS, MARKER_BITS: MARKER_BITS,
  SERIAL_BITS: SERIAL_BITS, SERIAL_MAX: SERIAL_MAX, SERIAL_CELLS: SERIAL_CELLS, SERIAL_CRC_BITS: SERIAL_CRC_BITS,
  START_GUARD: START_GUARD, STOP_GUARD: STOP_GUARD, HEADER_VERSIONS: HEADER_VERSIONS, HEADER_VERSION: HEADER_VERSION,
  HEADER_START_GUARD: HEADER_START_GUARD, HEADER_STOP_GUARD: HEADER_STOP_GUARD, HEADER_ROWS: HEADER_ROWS,
  HEADER_ROW_DATA: HEADER_ROW_DATA, HEADER_ROW_CELLS: HEADER_ROW_CELLS, LABEL_STYLES: LABEL_STYLES,
  BUBBLE_BASE_MM: BUBBLE_BASE_MM, BUBBLE_STEP_MM: BUBBLE_STEP_MM,
  markerId: markerId, decodeMarkerId: decodeMarkerId, markerBits: markerBits, markerImage: markerImage,
  crc8: crc8, crc16: crc16, encodeMeta: encodeMeta, decodeMeta: decodeMeta, encodeSerial: encodeSerial, decodeSerial: decodeSerial,
  Header: Header, headerFormKey: headerFormKey, headerChoiceLabels: headerChoiceLabels, choiceLabels: choiceLabels,
  labelStyle: labelStyle, validateHeader: validateHeader, encodeHeader: encodeHeader, decodeHeaderData: decodeHeaderData,
  decodeHeader: decodeHeader
};
})(globalThis.OMR = globalThis.OMR || {});
