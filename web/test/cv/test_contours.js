// adaptiveThreshold / findContours / approxPolyDP / arcLength / contourArea / isContourConvex vs cv2 (fixtures/prim 'contours' cases)
load('web/src/cv.js'); load('web/test/cv/lib.js');
var cv = OMR.cv, P = FX + '/prim/', man = readJSON(P + 'manifest.json');
var tot = { thresholdCases: 0, thresholdBad: 0, contourSets: 0, contourSetsBad: 0, externalSets: 0, externalBad: 0, approx: 0, approxBad: 0, convexBad: 0, arcMax: 0, areaMax: 0 };
function same(a, b) { if (a.length !== b.length) return false; for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; }
man.forEach(function (c) {
    if (c.op !== 'contours') return;
    var sh = c.in[0].shape, g = { w: sh[1], h: sh[0], data: readBin(P + c.in[0].file) }, exp = readBin(P + c.at.file);
    var t = cv.adaptiveThreshold(g, c.win, 7), d = diffU8(t.data, exp);
    tot.thresholdCases++; if (d.bad) { tot.thresholdBad++; print('  threshold diff ' + c.name + ' ' + d.bad); }
    var cs = cv.findContours(t, 'list', 'none');
    tot.contourSets++;
    var ok = cs.length === c.list_none.length;
    if (ok) for (var i = 0; i < cs.length; i++) if (!same(cs[i], c.list_none[i])) { ok = false; break; }
    if (!ok) { tot.contourSetsBad++; print('  contours differ ' + c.name); }
    var ex = cv.findContours(t, 'external', 'simple'); tot.externalSets++;
    var okx = ex.length === c.ext_simple.length; if (okx) for (i = 0; i < ex.length; i++) if (!same(ex[i], c.ext_simple[i])) { okx = false; break; }
    if (!okx) { tot.externalBad++; print('  external/simple differ ' + c.name); }
    c.approx.forEach(function (a) {
        var cc = Int32Array.from(a[0]), ap = cv.approxPolyDP(cc, cc.length / 2 * 0.03, true);
        tot.approx++; if (!same(ap, a[1])) tot.approxBad++;
        tot.arcMax = Math.max(tot.arcMax, Math.abs(cv.arcLength(cc, true) - a[2]));
        tot.areaMax = Math.max(tot.areaMax, Math.abs(cv.contourArea(cc) - a[3]));
        if (cv.isContourConvex(ap) !== a[4] && same(ap, a[1])) tot.convexBad++;
    });
});
print((tot.thresholdBad + tot.contourSetsBad + tot.externalBad + tot.convexBad ? 'FAIL ' : 'PASS ') + JSON.stringify(tot));
