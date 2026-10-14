// jsc <src> ser.js dump.js -- <fixtures dir> <set> <idx,idx,...>  : per frame "name status ms jsonhash canvashash"
(function (A) {
    var dir = A[0], set = A[1], idxs = A[2].split(',').map(Number);
    var man = JSON.parse(readFile(dir + '/' + set + '.manifest.json'));
    function hs(s) { var h = 0; for (var i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0; return h; }
    function hb(u) { var h = 0; for (var i = 0; i < u.length; i++) h = (h * 31 + u[i]) | 0; return h; }
    idxs.forEach(function (i) {
        var m = man[i], gray = { w: m.w, h: m.h, data: readFile(dir + '/' + m.file, 'binary') };
        var t0 = performance.now(), r = OMR.recognize.recognize(gray), dt = performance.now() - t0;
        var j = JSON.parse(PIPE.ser(r)); delete j.timing_ms;
        print(m.name + ' ' + r.status + ' ' + dt.toFixed(0) + ' ' + hs(JSON.stringify(j)) + ' ' + (r.warped ? hb(r.warped.data) : 0));
    });
})(arguments);
