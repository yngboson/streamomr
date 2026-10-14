var idx = +arguments[0] || 16, reps = +arguments[1] || 40;
var man = JSON.parse(readFile('web/test/fixtures/pipe/video.manifest.json'))[idx];
var gray = { w: man.w, h: man.h, data: readFile('web/test/fixtures/pipe/' + man.file, 'binary') };
for (var i=0;i<reps;i++) OMR.recognize.recognize(gray);
