// first-frame time with / without a warm-up of the kernels:   jsc <src> cold.js -- <idx> <warm 0|1>
var idx=+arguments[0], warm=+arguments[1];
var man = JSON.parse(readFile('web/test/fixtures/pipe/video.manifest.json'))[idx];
var gray = { w: man.w, h: man.h, data: readFile('web/test/fixtures/pipe/' + man.file, 'binary') };
if (warm) { var t0=performance.now(); OMR.recognize.warmup(+arguments[2] || 2); print('warmup ms', (performance.now()-t0).toFixed(0)); }
var r=[]; for (var i=0;i<6;i++){ var t0=performance.now(); OMR.recognize.recognize(gray); r.push((performance.now()-t0).toFixed(0)); } print('frames', r.join(' '));
