// omr/identity.py: physical-sheet identity, shared by the stream state machine and the database policy.
//   sheet_key = sha1(f"{serial:04d}|{subject_text}|{student_id_text}")[:16]
// Works on a result object (types.js) or on its serialised dict (same field names).
(function (O) {
'use strict';
var sha1hex = O.util.sha1hex, T = O.types;

function serialOf(r) {                   // omr/identity.py:24
  var v = r.serial;
  if (!v) return 0;
  var n = Math.trunc(Number(v));
  return isNaN(n) ? 0 : n;
}

function digitsText(value, digits) {     // omr/identity.py:31
  if (value != null && String(value).trim() !== '') return String(value).trim();
  var out = [];
  (digits || []).forEach(function (d) { out.push(d.value == null ? '?' : String(d.value)); });
  return out.join('');
}

function studentIdText(r) { return digitsText(r.student_id, r.id_digits); }
function subjectText(r) { return digitsText(r.subject, r.subject_digits); }

function pad4(n) { return ('0000' + n).slice(-4); }

function makeSheetKey(serial, idText, subject) {
  return sha1hex(pad4(Math.trunc(serial || 0)) + '|' + (subject || '') + '|' + idText).slice(0, 16);
}

function sheetKey(r) { return makeSheetKey(serialOf(r), studentIdText(r), subjectText(r)); }

function formatSerial(serial) { return serial ? 'No. ' + pad4(Math.trunc(serial)) : ''; }

// ---- commit feedback for the person scanning (server and standalone use the same rule)
var SEVERE_FLAGS = ['id_unreadable', 'id_guessed', 'id_blank', 'subject_unreadable', 'subject_guessed', 'subject_blank'];
var QUIET_FLAGS = ['serial_unreadable', 'corner_hidden', 'marker_dropped'];   // informational: no warning

// 'severe' = the sheet's identity is not reliably read; 'minor' = some answer rows need a look; 'ok' = nothing to check.
function severity(r) {                   // omr/identity.py:67
  var flags = r.flags || [], hasId = !!(r.id_digits && r.id_digits.length);
  if (flags.some(function (f) { return SEVERE_FLAGS.indexOf(f) >= 0; }) || (hasId && !r.student_id)) return 'severe';
  var answers = r.answers || [];
  var doubtful = answers.some(function (a) { return a.ambiguous || a.state === 'multi'; });
  if (doubtful || T.needsReview(r) || flags.some(function (f) { return QUIET_FLAGS.indexOf(f) < 0; })) return 'minor';
  return 'ok';
}

O.identity = { serialOf: serialOf, studentIdText: studentIdText, subjectText: subjectText, makeSheetKey: makeSheetKey, sheetKey: sheetKey,
  formatSerial: formatSerial, severity: severity, SEVERE_FLAGS: SEVERE_FLAGS, QUIET_FLAGS: QUIET_FLAGS };
})(globalThis.OMR = globalThis.OMR || {});
