// Result objects of the recognizer (omr/recognize.py dataclasses) as plain JS objects, with the SAME snake_case field names.
// See web/src/TYPES.md.  Constructors fill the Python defaults; the helper functions replace the Python properties
// (a plain object cannot compute them, and shallow copies with edited fields must recompute them).
(function (O) {
'use strict';

var OK = 'ok', NO_SHEET = 'no_sheet', PARTIAL = 'partial', INCONSISTENT = 'inconsistent';
var UNKNOWN_FORM = 'unknown_form', BAD_META = 'bad_meta', BAD_GEOMETRY = 'bad_geometry';
var DEFAULT_PX_PER_MM = 6.0;
var SOFT_FLAGS = ['serial_unreadable', 'corner_hidden', 'guessed_answers', 'marker_dropped'];   // informational: never a reason for human review
var UNRELIABLE_FILL = 0.5, UNRELIABLE_STRENGTH = 0.3;   // an unknowable bubble (see omr.illum): ambiguous, never a read digit

function hardFlags(flags) {              // omr/recognize.py:78
  return (flags || []).filter(function (f) { return SOFT_FLAGS.indexOf(f) < 0; });
}

// QuestionResult (omr/recognize.py:84).  marked: choice indices, labels: strings, scores: numbers.
function QuestionResult(number, state, marked, labels, confidence, ambiguous, scores, guess, guessed) {
  return { number: number, state: state, marked: marked, labels: labels, confidence: confidence, ambiguous: ambiguous, scores: scores,
    guess: guess === undefined ? null : guess, guessed: guessed === undefined ? false : guessed };
}

// the row's value: the single mark, or - for an uncertain blank row - the best guess (null otherwise)
function answerOf(a) { return a.state === 'single' ? a.marked[0] : a.guess; }

// DigitResult (omr/recognize.py:105)
function DigitResult(index, state, value, confidence, ambiguous, scores, guessed) {
  return { index: index, state: state, value: value, confidence: confidence, ambiguous: ambiguous, scores: scores,
    guessed: guessed === undefined ? false : guessed };
}

// RecognitionResult (omr/recognize.py:116) with its defaults.
function RecognitionResult(status, pxPerMm) {
  return {
    status: status, form_key: null, form_id: null, page: null, total_pages: null, serial: null, answers: [],
    student_id: null, id_digits: [], subject: null, subject_digits: [], flags: [], marker_ids: [],
    marker_corners: [],            // JS: array of [id, [[x, y] x4]] in detection order (Python: dict id -> quad, insertion order)
    missing_corners: [], homography: null, px_per_mm: pxPerMm === undefined ? DEFAULT_PX_PER_MM : pxPerMm,
    quality: {}, calibration: {}, bubble_centers_mm: {}, timing_ms: {},
    warped: null, name_crop: null,     // grey images {w, h, data} (or null)
    raw: null, fused_frames: 0,
    dewarp: null                        // null (flat sheet) or a function (x_mm, y_mm) -> [dx_mm, dy_mm]  (omr.tracks.Field.__call__)
  };
}

// PageRaw (omr/recognize.py:956).  2-D arrays are arrays of rows: FA/FI/FS/MA/MI/MS rows are Float32Array, SA/SI/SS rows Float64Array.
function PageRaw(numbers, labels, digitIndex, FA, SA, FI, SI, ink, minStrength, verified, o) {
  o = o || {};
  var r = { numbers: numbers, labels: labels, digit_index: digitIndex, FA: FA, SA: SA, FI: FI, SI: SI, ink: ink, min_strength: minStrength,
    verified: verified === undefined ? true : verified, FS: null, SS: null, ink_source: 'default', ink_separation: 0.0, MA: null, MI: null, MS: null };
  for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) r[k] = o[k];
  return r;                  // decided_sig (an array, see decide.decisionSig) is attached by recognize() after deciding
}

function idGuessed(r) { return r.flags.indexOf('id_guessed') >= 0; }
function subjectGuessed(r) { return r.flags.indexOf('subject_guessed') >= 0; }

function needsReview(r) {                // omr/recognize.py RecognitionResult.needs_review
  return r.status === OK && (r.student_id == null || r.subject == null || hardFlags(r.flags).length > 0 ||
    r.answers.some(function (a) { return a.ambiguous || a.guessed || a.state !== 'single'; }));
}

// 3x3 inverse by Gauss-Jordan with partial pivoting (like LAPACK's getrf/getri up to rounding)
function inv3(m) {
  var a = [[m[0][0], m[0][1], m[0][2], 1, 0, 0], [m[1][0], m[1][1], m[1][2], 0, 1, 0], [m[2][0], m[2][1], m[2][2], 0, 0, 1]], i, j, k;
  for (i = 0; i < 3; i++) {
    var p = i;
    for (j = i + 1; j < 3; j++) if (Math.abs(a[j][i]) > Math.abs(a[p][i])) p = j;
    if (a[p][i] === 0) throw new Error('singular matrix');
    var t = a[i]; a[i] = a[p]; a[p] = t;
    var d = a[i][i];
    for (k = 0; k < 6; k++) a[i][k] /= d;
    for (j = 0; j < 3; j++) {
      if (j === i) continue;
      var f = a[j][i];
      if (f !== 0) for (k = 0; k < 6; k++) a[j][k] -= f * a[i][k];
    }
  }
  return [[a[0][3], a[0][4], a[0][5]], [a[1][3], a[1][4], a[1][5]], [a[2][3], a[2][4], a[2][5]]];
}

// RecognitionResult.image_points: layout mm (page space) -> source-image pixels, as an array of [x, y] (for overlays)
function imagePoints(r, ptsMm) {
  if (r.homography == null) throw new Error('no homography');
  var Hi = inv3(r.homography), out = [], ppm = r.px_per_mm;
  for (var i = 0; i < ptsMm.length; i++) {
    var x = ptsMm[i][0], y = ptsMm[i][1];
    if (r.dewarp != null) { var d = r.dewarp(x, y); x = x + d[0]; y = y + d[1]; }
    x = x * ppm; y = y * ppm;
    var w = x * Hi[2][0] + y * Hi[2][1] + Hi[2][2];       // cv2.perspectiveTransform
    if (Math.abs(w) > 2.220446049250313e-16) {
      w = 1.0 / w;
      out.push([(x * Hi[0][0] + y * Hi[0][1] + Hi[0][2]) * w, (x * Hi[1][0] + y * Hi[1][1] + Hi[1][2]) * w]);
    } else out.push([0, 0]);
  }
  return out;
}

O.types = {
  OK: OK, NO_SHEET: NO_SHEET, PARTIAL: PARTIAL, INCONSISTENT: INCONSISTENT, UNKNOWN_FORM: UNKNOWN_FORM, BAD_META: BAD_META,
  BAD_GEOMETRY: BAD_GEOMETRY, DEFAULT_PX_PER_MM: DEFAULT_PX_PER_MM, SOFT_FLAGS: SOFT_FLAGS, UNRELIABLE_FILL: UNRELIABLE_FILL,
  UNRELIABLE_STRENGTH: UNRELIABLE_STRENGTH, hardFlags: hardFlags, QuestionResult: QuestionResult, answerOf: answerOf,
  DigitResult: DigitResult, RecognitionResult: RecognitionResult, PageRaw: PageRaw, idGuessed: idGuessed, subjectGuessed: subjectGuessed,
  needsReview: needsReview, imagePoints: imagePoints
};
})(globalThis.OMR = globalThis.OMR || {});
