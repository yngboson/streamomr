// per-section minimum over reps for one frame:  jsc <src> ser.js stages.js -- <fixtures dir> <set> <idx> [reps]
(function (A) {
    var dir = A[0], set = A[1], idx = +A[2], reps = +(A[3] || 8);
    var m = JSON.parse(readFile(dir + '/' + set + '.manifest.json'))[idx];
    var gray = { w: m.w, h: m.h, data: readFile(dir + '/' + m.file, 'binary') };
    OMR.recognize.profile = true;
    var best = {}, r;
    for (var i = 0; i < reps; i++) {
        r = OMR.recognize.recognize(gray);
        var tm = r.timing_ms;
        for (var k in tm) if (!(k in best) || tm[k] < best[k]) best[k] = tm[k];
    }
    print(m.name + ' ' + r.status + '  ' + Object.keys(best).filter(function (k) { return k !== 'total'; }).map(function (k) { return k.replace(/^x_/, '') + ' ' + best[k].toFixed(0); }).join(' | ') + '  || total(min) ' + best.total.toFixed(0));
})(arguments);
