// jsc web/test/cv/test_prim.js -- web/test/fixtures [opFilter]   : cv.js primitives vs cv2 outputs (fixtures/prim)
load('web/src/cv.js'); load('web/test/cv/lib.js');
var TIMING = arguments[3] === 'time'; var LIBNAME = arguments[2] || 'cv'; if (LIBNAME === 'cvx') load('web/src/cvx.js');
var cv0 = OMR.cv, P = FX + '/prim/', man = readJSON(P + 'manifest.json'), filt = arguments[1] || '';
var cv = Object.create(cv0);
if (LIBNAME === 'cvx') {
    var X = OMR.cvx; ['gaussianBlur', 'dilate', 'erode', 'morphOpen', 'matchTemplateCcoeffNormed', 'warpPerspective', 'remap', 'remapLinear'].forEach(function (n) { cv[n] = X[n]; });
    cv.resize = function (img, w, h, mode) { return X.resize(img, w, h, mode); };
    cv.resizeBy = function (img, fx, fy, mode) { return X.resize(img, Math.max(cv0.rint(img.w * fx), 1), Math.max(cv0.rint(img.h * fy), 1), mode, fx, fy); };
}
var TY = { uint8: Uint8Array, float32: Float32Array, float64: Float64Array, int32: Int32Array };
function rd(m) { var T = TY[m.dtype]; var a = readBin(P + m.file, T); var sh = m.shape; return sh.length === 1 ? { w: sh[0], h: 1, data: a } : sh.length === 2 ? { w: sh[1], h: sh[0], data: a } : { w: sh[1], h: sh[0], c: sh[2], data: a }; }
function cmp(a, b) { var n = Math.min(a.length, b.length), mx = 0, bad = 0, sum = 0; if (a.length !== b.length) return { mx: 1e9, bad: n, n: n, rate: 1, len: [a.length, b.length] }; for (var i = 0; i < n; i++) { var d = Math.abs(a[i] - b[i]); if (d > mx) mx = d; sum += d; if (d > 0) bad++; } return { mx: mx, bad: bad, n: n, rate: bad / n }; }
var agg = {};
function note(key, r, tol, ms, cvms, desc) {
    var A = agg[key] = agg[key] || { n: 0, fail: 0, mx: 0, rateMax: 0, ms: [], cvms: [], worst: '' };
    A.n++; if (r.mx > tol) { A.fail++; if (A.fail <= 2) A.worst += ' [' + desc + ' max ' + r.mx.toFixed(4) + ' bad ' + (r.rate * 100).toFixed(3) + '%]'; }
    A.mx = Math.max(A.mx, r.mx); A.rateMax = Math.max(A.rateMax, r.rate); if (ms !== undefined) { A.ms.push(ms); A.cvms.push(cvms); }
}
function med(a) { a = a.slice().sort(function (x, y) { return x - y; }); return a.length ? a[a.length >> 1] : 0; }
man.forEach(function (c) {
    if (c.op === 'contours' || (filt && c.op !== filt)) return;
    var ins = c.in.map(rd), exp = c.out.file ? rd(c.out) : null, p = c.params, got, fn;
    switch (c.op) {
        case 'resize_by': fn = function () { return cv.resizeBy(ins[0], p.fx, p.fx, p.mode); }; break;
        case 'resize': fn = function () { return cv.resize(ins[0], p.w, p.h, p.mode); }; break;
        case 'gaussian': fn = function () { return cv.gaussianBlur(ins[0], p.sigma); }; break;
        case 'dilate': fn = function () { return cv.dilate(ins[0], p.k); }; break;
        case 'erode': fn = function () { return cv.erode(ins[0], p.k); }; break;
        case 'open': fn = function () { return cv.morphOpen(ins[0], p.k); }; break;
        case 'box': fn = function () { return cv.boxFilter(ins[0], p.kx, true, p.ky); }; break;
        case 'laplacian': fn = function () { return cv.laplacian(ins[0]); }; break;
        case 'integral': fn = function () { return cv.integral(ins[0]); }; break;
        case 'otsu': fn = function () { return cv.otsu(ins[0], p.inv).bin; }; break;
        case 'not': fn = function () { return cv.bitwiseNot(ins[0]); }; break;
        case 'divide': fn = function () { return cv.divide(ins[0], ins[1], p.scale); }; break;
        case 'addWeighted': fn = function () { return cv.addWeighted(ins[0], p.a, ins[1], p.b, p.g); }; break;
        case 'gray': fn = function () { return cv.toGray(ins[0].data, ins[0].w, ins[0].h, p.ch, p.order); }; break;
        case 'getPT': fn = function () { return { data: cv.getPerspectiveTransform(ins[0].data, ins[1].data) }; }; break;
        case 'warp': fn = function () { return cv.warpPerspective(ins[0], ins[1].data, p.w, p.h, { interp: p.interp, value: p.value }); }; break;
        case 'perspT': fn = function () { return { data: cv.perspectiveTransform(ins[0].data, ins[1].data) }; }; break;
        case 'findH': fn = function () { return { data: cv.findHomography(ins[0].data, ins[1].data) }; }; break;
        case 'remap': fn = function () { return cv.remap(ins[0], ins[1].data, ins[2].data, p.w, p.h, p.border, p.value); }; break;
        case 'remapLinear': fn = function () { return { data: cv.remapLinear(ins[0], ins[1].data, ins[2].data, 'replicate') }; }; break;
        case 'subpix': fn = function () { var q = ins[1].data, f = new Float64Array(8); for (var i = 0; i < 8; i++) f[i] = q[i]; return { data: cv.cornerSubPix(ins[0], f, p.win, 30, 0.1) }; }; break;
        case 'matchT': fn = function () { return cv.matchTemplateCcoeffNormed(ins[0], ins[1]); }; break;
        default: print('unknown op ' + c.op); return;
    }
    try { got = fn(); } catch (e) { print('ERR ' + c.op + ' ' + JSON.stringify(p) + ' ' + e + ' ' + (e.stack || '').split('\n')[0]); note(c.op, { mx: 1e9, rate: 1 }, 0, undefined, undefined, 'exception'); return; }
    var r = cmp(got.data, exp.data);
    var ms = (TIMING && ins[0].data.length > 5000) ? bench(fn, 60) : undefined;
    note(c.op + (c.note ? '(' + c.note + ')' : ''), r, (c.op === 'warp' || c.op === 'remap') ? 3 : (c.op === 'findH' ? 1e-3 : c.tol), ms, c.cv2_ms, JSON.stringify(p).slice(0, 60) + (r.len ? ' len ' + r.len : ''));
});
Object.keys(agg).sort().forEach(function (k) {
    var A = agg[k];
    print((A.fail ? 'FAIL ' : 'PASS ') + k + ' cases ' + A.n + ' fail ' + A.fail + ' maxdiff ' + A.mx.toFixed(5) + ' worst-bad-rate ' + (A.rateMax * 100).toFixed(3) + '%' + (A.ms.length ? ' | js ' + med(A.ms).toFixed(2) + 'ms cv2 ' + med(A.cvms).toFixed(2) + 'ms' : '') + A.worst);
});
