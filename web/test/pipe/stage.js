// stage parity: JS stages fed with Python's intermediates.  jsc <src> ser.js stage.js -- <fixtures dir> <set>_<idx>
(function (A) {
    var dir = A[0], key = A[1], O_ = OMR, cv = O_.cv, cvx = O_.cvx;
    var meta = JSON.parse(readFile(dir + '/stage/' + key + '.json'));
    var man = JSON.parse(readFile(dir + '/' + meta.set + '.manifest.json'))[meta.idx];
    function arr(name, T) { var a = meta.arrays[name], u = readFile(dir + '/stage/' + a.file, 'binary'); return { shape: a.shape, data: new T(u.buffer, u.byteOffset, u.byteLength / T.BYTES_PER_ELEMENT) }; }
    function img(name, T) { var a = arr(name, T || Uint8Array); return { w: a.shape[1], h: a.shape[0], data: a.data }; }
    function cmp(label, a, b) {
        var n = Math.min(a.length, b.length), mx = 0, c = 0, sum = 0;
        for (var i = 0; i < n; i++) { var d = Math.abs(a[i] - b[i]); if (isNaN(d)) d = (isNaN(a[i]) && isNaN(b[i])) ? 0 : 1e9; if (d > mx) mx = d; if (d > 0) c++; sum += d; }
        print(label + ': n=' + n + ' differing=' + c + ' max=' + mx.toPrecision(4) + ' mean=' + (sum / n).toPrecision(3) + (a.length !== b.length ? ' LENGTH ' + a.length + '/' + b.length : ''));
    }
    var gray = { w: man.w, h: man.h, data: readFile(dir + '/' + man.file, 'binary') };
    var H = Float64Array.from([].concat.apply([], meta.H));
    var t0 = performance.now();
    var warped = OMR.cv.warpPerspective(gray, H, 1260, 1782, { interp: 'linear', border: 'constant', value: 255 });
    print('warp ' + (performance.now() - t0).toFixed(0) + 'ms');
    var pw = img('warped');
    cmp('warped', pw.data, warped.data);
    var flat = O_.recognize.flattenHeader ? O_.recognize.flattenHeader(pw, 6.0) : null;
    if (flat) { var pf = arr('header_flat', Float32Array); cmp('header_flat', pf.data, flat.data); }
    var p2 = img('warped2');
    var t1 = performance.now(); var sm = cv.gaussianBlur(p2, 1.5); print('blur ' + (performance.now() - t1).toFixed(0) + 'ms');
    var ps = img('smooth'); cmp('smooth', ps.data, sm.data);
    var smooth = ps;   // python's smooth feeds the rest
    var groups = [[meta.cent, meta.dia[0]], [meta.icent, meta.dia[1]]];
    t1 = performance.now();
    var diag = {}, fa = O_.recognize.fitAlignment(smooth, groups, 6.0, meta.flags.missing && meta.tracks_none, meta.flags.strict, diag);
    print('fitAlignment ' + (performance.now() - t1).toFixed(0) + 'ms');
    if (meta.coef && fa.coef) { cmp('coef', [].concat.apply([], meta.coef), [].concat.apply([], fa.coef)); } else print('coef py ' + JSON.stringify(meta.coef) + ' js ' + JSON.stringify(fa.coef));
    print('align info js ' + JSON.stringify(fa.info) + '\n          py ' + JSON.stringify(meta.align_info));
    t1 = performance.now();
    var DA = O_.illum.bubbleDarkness(smooth, meta.centA, meta.dia[0], 6.0);
    print('illum A ' + (performance.now() - t1).toFixed(0) + 'ms');
    var pDA = arr('DA', Float32Array); cmp('DA', pDA.data, DA.D);
    var rel = []; for (var i = 0; i < DA.n; i++) rel.push(DA.reliable[i] ? 0 : 1);
    cmp('UA', meta.UA, rel);
    var DI = O_.illum.bubbleDarkness(smooth, meta.centI, meta.dia[1], 6.0);
    cmp('DI', arr('DI', Float32Array).data, DI.D);
    // darkness of python's D into ink
    var pD = { D: pDA.data, n: DA.n, m: DA.m };
    var inkA = O_.ink.bubbleInk(pD.D, pD.n, pD.m); cmp('inkA (from python D)', meta.inkA, inkA);
    var inkA2 = O_.ink.bubbleInk(DA.D, DA.n, DA.m); cmp('inkA (from js D)', meta.inkA, inkA2);
    var v = O_.illum.paperSamples(smooth, meta.centA, meta.dia[0], 6.0); cmp('paper samples v', [].concat.apply([], meta.v), v);
})(arguments);
