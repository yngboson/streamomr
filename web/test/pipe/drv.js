// jsc driver: recognise frames of a set.   jsc <src files> ser.js drv.js -- <fixtures dir> <set> <start> <end> [reps]
// prints one line "@<json>" per frame: {name, ms, res}
(function (A) {
    var dir = A[0], set = A[1], reps = +(A[3] || 1), idxs = [];
    if (A[4] === 'parity') OMR.detect.setPolicy({ flatOpen: true, hintOnly: false, candScale: 0, coarse: false });      // omr/detect.py's exact pass sequence
    var man = JSON.parse(readFile(dir + '/' + set + '.manifest.json'));
    if (A[2].indexOf(',') >= 0) idxs = A[2].split(',').map(Number);
    else { var se = A[2].split('-'); for (var q = +se[0]; q < Math.min(+se[1], man.length); q++) idxs.push(q); }
    for (var ii = 0; ii < idxs.length; ii++) {
        var i = idxs[ii];
        var m = man[i], d = readFile(dir + '/' + m.file, 'binary'), gray = { w: m.w, h: m.h, data: d };
        var best = Infinity, r = null;
        for (var k = 0; k < reps; k++) {
            var t0 = performance.now();
            r = OMR.recognize.recognize(gray);
            best = Math.min(best, performance.now() - t0);
        }
        print('@' + JSON.stringify({ name: m.name, idx: i, ms: best, res: JSON.parse(PIPE.ser(r)) }));
    }
})(arguments);
