// omr/revision.py: revision of a committed sheet: upgrade-only merge of a later (fused) reading into the committed one.
// A confident row is never flipped quietly; a contradiction adds the soft-hard flag "contradiction" (needs review).  Pure.
(function (O) {
'use strict';

var CONTRADICTION_FLAG = 'contradiction';
var ID_FLAGS = ['id_blank', 'id_unreadable', 'id_guessed'];
var SUBJECT_FLAGS = ['subject_blank', 'subject_unreadable', 'subject_guessed'];

function answerConfident(a) { return !(a.ambiguous || a.guessed); }          // omr/revision.py:26

function sameVal(a, b) {
  if (a.state !== b.state || a.marked.length !== b.marked.length) return false;
  for (var i = 0; i < a.marked.length; i++) if (a.marked[i] !== b.marked[i]) return false;
  return true;
}
function neq(a, b) { return (a == null ? null : a) !== (b == null ? null : b); }

function idConfident(r) { return r.student_id != null && r.flags.indexOf('id_guessed') < 0; }
function subjectConfident(r) { return r.subject != null && r.flags.indexOf('subject_guessed') < 0; }

function doubtful(r) {                   // omr/revision.py:44: anything left that a later frame could still resolve?
  if (r.id_digits.length && !idConfident(r)) return true;
  if (r.subject_digits.length && !subjectConfident(r)) return true;
  return r.answers.some(function (a) { return !answerConfident(a); });
}

// omr/revision.py:54: -> [merged result, changed rows, contradicted rows]; rows are "q<number>", "id", "subject"
function merge(base, now) {
  var changed = [], contra = [], answers = [], byNum = {};
  now.answers.forEach(function (a) { byNum[a.number] = a; });
  base.answers.forEach(function (b) {
    var n = byNum[b.number], keep = b;
    if (n !== undefined) {
      if (answerConfident(b)) {
        if (answerConfident(n) && !sameVal(n, b)) contra.push('q' + b.number);
      } else if (answerConfident(n)) {
        keep = n;
        changed.push('q' + b.number);
      }
    }
    answers.push(keep);
  });
  var out = Object.assign({}, now);
  out.answers = answers;
  var drop = ID_FLAGS.concat(SUBJECT_FLAGS, ['guessed_answers', CONTRADICTION_FLAG]);
  var flags = now.flags.filter(function (f) { return drop.indexOf(f) < 0; });
  [['id', idConfident, 'id_digits', 'student_id', ID_FLAGS], ['subject', subjectConfident, 'subject_digits', 'subject', SUBJECT_FLAGS]].forEach(function (s) {
    var name = s[0], conf = s[1], digitsAttr = s[2], valAttr = s[3], fl = s[4], src;
    if (!base[digitsAttr].length) src = now;
    else if (conf(base)) {
      src = base;
      if (conf(now) && neq(now[valAttr], base[valAttr])) contra.push(name);
    } else if (conf(now)) {
      src = now;
      changed.push(name);
    } else src = base;
    out[digitsAttr] = src[digitsAttr];
    out[valAttr] = src[valAttr];
    src.flags.forEach(function (f) { if (fl.indexOf(f) >= 0) flags.push(f); });
  });
  if (answers.some(function (a) { return a.guessed; })) flags.push('guessed_answers');
  if (contra.length || base.flags.indexOf(CONTRADICTION_FLAG) >= 0) flags.push(CONTRADICTION_FLAG);
  out.flags = flags;
  return [out, changed, contra];
}

O.revision = { CONTRADICTION_FLAG: CONTRADICTION_FLAG, answerConfident: answerConfident, idConfident: idConfident,
  subjectConfident: subjectConfident, doubtful: doubtful, merge: merge };
})(globalThis.OMR = globalThis.OMR || {});
