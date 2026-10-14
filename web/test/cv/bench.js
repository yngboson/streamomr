// jsc web/test/cv/bench.js -- web/test/fixtures : JS ms/call for the same operations as bench_cv2.py (reads fixtures/bench_cv2.json)
load('web/src/cv.js'); load('web/src/aruco.js'); load('web/test/cv/lib.js');
var cv = OMR.cv, ref = readJSON(FX + '/bench_cv2.json'), man = readJSON(FX + '/aruco/manifest.json');
var c = man.filter(function (c) { return c.src.indexOf('video:') === 0 && c.w === 1080 && c.h === 1920 && c.ids.length >= 3; })[0];
var g = { w: c.w, h: c.h, data: readBin(FX + '/aruco/' + c.file) };
var gf = { w: g.w, h: g.h, data: Float32Array.from(g.data) };
var small = { w: 240, h: 135, data: g.data.slice(0, 240 * 135) };
var H = Float64Array.from([1.0, 0.02, 10, 0.01, 1.1, 20, 0.00001, 0.00002, 1]);
var mx = new Float32Array(1400 * 1000), my = new Float32Array(1400 * 1000);
for (var i = 0; i < mx.length; i++) { mx[i] = (i * 7919 % 1000); my[i] = (i * 104729 % 1800); }
var bw = cv.adaptiveThreshold(g, 13, 7);
var ops = {
    'adaptiveThreshold win3': function () { cv.adaptiveThreshold(g, 3, 7); },
    'adaptiveThreshold win13': function () { cv.adaptiveThreshold(g, 13, 7); },
    'adaptiveThreshold win23': function () { cv.adaptiveThreshold(g, 23, 7); },
    'findContours LIST/NONE (win13)': function () { cv.findContours(bw, 'list', 'none'); },
    'findContours EXTERNAL/SIMPLE (win13)': function () { cv.findContours(bw, 'external', 'simple'); },
    'aruco detectMarkers (plain)': function () { OMR.aruco.detect(g); },
    'aruco detectMarkers (opened k=9)': function () { OMR.aruco.detect(cv.morphOpen(g, 9)); },
    'resize area x0.25': function () { cv.resizeBy(g, 0.25, 0.25, 'area'); },
    'resize area x0.125 (u8)': function () { cv.resize(g, g.w >> 3, g.h >> 3, 'area'); },
    'resize area x0.4 fractional': function () { cv.resizeBy(g, 0.4, 0.4, 'area'); },
    'resize linear small->full (u8)': function () { cv.resize(small, 1920, 1080, 'linear'); },
    'GaussianBlur u8 sigma 1.2': function () { cv.gaussianBlur(g, 1.2); },
    'GaussianBlur u8 sigma 3': function () { cv.gaussianBlur(g, 3); },
    'GaussianBlur f32 sigma 1.2': function () { cv.gaussianBlur(gf, 1.2); },
    'dilate k=9': function () { cv.dilate(g, 9); },
    'erode k=23': function () { cv.erode(g, 23); },
    'morphology OPEN k=9': function () { cv.morphOpen(g, 9); },
    'boxFilter f32 21x15': function () { cv.boxFilter(gf, 21, true, 15); },
    'Laplacian': function () { cv.laplacian(g); },
    'integral f64': function () { cv.integral(g); },
    'threshold otsu': function () { cv.otsu(g, true); },
    'warpPerspective linear 1400x1000': function () { cv.warpPerspective(g, H, 1400, 1000, { interp: 'linear', value: 255 }); },
    'remap linear 1400x1000': function () { cv.remap(g, mx, my, 1400, 1000, 'constant', 255); },
    'cornerSubPix 4 corners': function () { cv.cornerSubPix(g, Float64Array.from(c.corners[0]), 4, 30, 0.1); },
    'bitwise_not': function () { cv.bitwiseNot(g); }
};
print('frame ' + ref.frame);
print('operation'.padEnd(40) + 'js ms'.padStart(9) + 'cv2 ms'.padStart(9) + '  ratio');
var only = arguments[1] ? arguments[1].split(',') : null;
Object.keys(ops).forEach(function (k) {
    if (only && !only.some(function (o) { return k.indexOf(o) >= 0; })) return;
    var ms = bench(ops[k], 250), r = ref[k];
    print(k.padEnd(40) + ms.toFixed(2).padStart(9) + (r === undefined ? '' : r.toFixed(2).padStart(9) + '  ' + (ms / r).toFixed(1) + 'x'));
});
