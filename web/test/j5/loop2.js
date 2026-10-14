var idx = +arguments[0] || 16, reps = +arguments[1] || 30;
var man = JSON.parse(readFile('web/test/fixtures/pipe/video.manifest.json'))[idx];
var gray = { w: man.w, h: man.h, data: readFile('web/test/fixtures/pipe/' + man.file, 'binary') };
var ts=[]; for (var i=0;i<reps;i++){ var t0=performance.now(); OMR.recognize.recognize(gray); ts.push(performance.now()-t0); }
ts.sort(function(a,b){return a-b;}); print('frame', idx, 'min', ts[0].toFixed(0), 'median', ts[reps>>1].toFixed(0), 'max', ts[reps-1].toFixed(0));
