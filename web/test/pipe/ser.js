// serialisation of JS RecognitionResults to the JSON shape of common.ser (python) and back (for the stream replay)
var PIPE = {};
PIPE.ser = function (r) {
    var o = {};
    ['status', 'form_key', 'form_id', 'page', 'total_pages', 'serial', 'student_id', 'subject', 'flags', 'marker_ids', 'missing_corners', 'homography',
        'px_per_mm', 'quality', 'calibration', 'bubble_centers_mm', 'timing_ms', 'fused_frames', 'answers', 'id_digits', 'subject_digits',
        'marker_corners'].forEach(function (k) { o[k] = r[k]; });
    o.raw = r.raw;
    o.dewarp = r.dewarp !== null;
    if (r.dewarp && r.dewarp.field) { var f = r.dewarp.field; o.dewarp_field = { c: f.c, fn: f.fn }; }
    return JSON.stringify(o, function (k, v) { return ArrayBuffer.isView(v) ? Array.from(v) : v; });
};
// JSON (as produced above, parsed) -> result object usable by the stream session
PIPE.rebuild = function (j) {
    var T = OMR.types, r = T.RecognitionResult(j.status, j.px_per_mm);
    for (var k in j) if (k !== 'raw' && k !== 'dewarp' && k !== 'dewarp_field') r[k] = j[k];
    function rows(m, C) { return m == null ? null : m.map(function (row) { return C.from(row.map(function (x) { return x === null ? NaN : x; })); }); }
    if (j.raw) {
        var w = {}; for (var q in j.raw) w[q] = j.raw[q];
        ['FA', 'FI', 'FS', 'MA', 'MI', 'MS'].forEach(function (q) { w[q] = rows(j.raw[q], Float32Array); });
        ['SA', 'SI', 'SS'].forEach(function (q) { w[q] = rows(j.raw[q], Float64Array); });
        r.raw = w;
        r.raw.decided_sig = OMR.decide.decisionSig(r.answers, r.student_id, r.subject);
    }
    if (j.dewarp && j.dewarp_field) { var f = j.dewarp_field; r.dewarp = function (x, y) { return OMR.tracks.fieldAt(f, x, y); }; }
    return r;
};
