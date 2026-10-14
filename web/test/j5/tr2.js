var man = JSON.parse(readFile('web/test/fixtures/pipe/video.manifest.json'));
var py = JSON.parse(readFile('web/test/fixtures/pipe/video.py.json'));
var cls = arguments[0], n = +arguments[1] || 6, picked = [];
for (var i = 0; i < man.length && picked.length < n; i += 7) if (py[man[i].name].status === cls) picked.push(i);
picked.forEach(function (i) {
    var m = man[i], gray = { w: m.w, h: m.h, data: readFile('web/test/fixtures/pipe/' + m.file, 'binary') };
    var last = performance.now(), parts = [];
    OMR.detect.setTrace(function (tag, img, r) { var t = performance.now(); parts.push(tag + ':' + img.w + 'x' + img.h + ' ids' + r.ids.length + ' rej' + r.rejected.length + ' ' + (t - last).toFixed(0) + 'ms'); last = t; });
    var t0 = performance.now(); last = t0;
    var r = OMR.recognize.recognize(gray);
    print(i + ' ' + r.status + ' total ' + (performance.now() - t0).toFixed(0) + ' | ' + parts.join(' | '));
});
