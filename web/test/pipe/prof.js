// function-level profile of one recognise() call:  jsc <src> ser.js prof.js -- <fixtures dir> <set> <idx> [reps]
(function (A) {
    var dir = A[0], set = A[1], idx = +A[2], reps = +(A[3] || 3);
    var man = JSON.parse(readFile(dir + '/' + set + '.manifest.json'))[idx];
    var gray = { w: man.w, h: man.h, data: readFile(dir + '/' + man.file, 'binary') };
    var acc = {}, depth = 0;
    function wrap(obj, name, label) {
        var f = obj[name];
        if (typeof f !== 'function') return;
        obj[name] = function () {
            var t0 = performance.now(), r = f.apply(this, arguments), dt = performance.now() - t0;
            var a = acc[label] || (acc[label] = { n: 0, ms: 0 });
            a.n++; a.ms += dt;
            return r;
        };
    }
    var targets = { cv: OMR.cv, cvx: OMR.cvx, aruco: OMR.aruco, illum: OMR.illum, ink: OMR.ink, fastops: OMR.fastops, tracks: OMR.tracks, detect: OMR.detect, recognize: OMR.recognize };
    for (var m in targets) for (var k in targets[m]) if (!/^(sample|setTrace)$/.test(k)) wrap(targets[m], k, m + '.' + k);
    OMR.recognize.profile = true;
    // internal (non-exported) helpers are not visible: the exported ones above cover the heavy primitives
    var best = Infinity, bestAcc = null;
    for (var i = 0; i < reps; i++) {
        acc = {};
        var t0 = performance.now(); var r = OMR.recognize.recognize(gray); var dt = performance.now() - t0;
        if (dt < best) { best = dt; bestAcc = acc; }
    }
    print('total ' + best.toFixed(0) + ' ms  status ' + r.status);
    var rows = Object.keys(bestAcc).map(function (k) { return [k, bestAcc[k].n, bestAcc[k].ms]; }).sort(function (a, b) { return b[2] - a[2]; });
    rows.slice(0, 28).forEach(function (r) { print((r[2]).toFixed(1).padStart(8) + ' ms  x' + String(r[1]).padStart(4) + '  ' + r[0]); });
    var tm = r.timing_ms; print(JSON.stringify(Object.keys(tm).reduce(function (o, k) { o[k] = Math.round(tm[k]); return o; }, {})));
    print('recognize.js sections: ' + Object.keys(tm).filter(function (k) { return k.indexOf('x_') === 0; }).map(function (k) { return k.slice(2) + ' ' + tm[k].toFixed(0); }).join(' | '));
})(arguments);
