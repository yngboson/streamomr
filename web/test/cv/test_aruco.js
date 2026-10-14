// jsc web/src/cv.js web/src/aruco.js ... : parity of aruco.detect against cv2 on every fixture.  args: [fixtureDir] [maxCases] [filter]
load('web/src/cv.js'); load('web/src/aruco.js'); load('web/test/cv/lib.js');
var A = (typeof arguments !== 'undefined') ? arguments : [];
var man = readJSON(FX + '/aruco/manifest.json'), maxN = +(A[1] || 1e9), filt = A[2] || '', only = A[3] ? A[3].split(',').map(Number) : null; if (A[4]) { var lh = A[4].split(','); OMR.aruco.lo = +lh[0]; OMR.aruco.hi = +lh[1]; OMR.aruco.lin = +(lh[2] || 0); OMR.aruco.geq = +(lh[4] || 0); }
var stat = { cases: 0, idsOk: 0, idsBad: 0, rejOk: 0, rejBad: 0, cornerMax: 0, cornerErrs: [], jsMs: 0, cvMs: 0, nCorners: 0 };
var perSrc = {};
function key(s) { return s.split(':')[0] + (s.indexOf('@') >= 0 ? '@' : ''); }
function cmpSets(js, ex) {   // match corner sets irrespective of order by nearest centroid
    var used = {}, bad = 0;
    for (var i = 0; i < ex.length; i++) {
        var ec = [(ex[i][0] + ex[i][2] + ex[i][4] + ex[i][6]) / 4, (ex[i][1] + ex[i][3] + ex[i][5] + ex[i][7]) / 4], best = -1, bd = 1e9;
        for (var j = 0; j < js.length; j++) { if (used[j]) continue; var c = [(js[j][0] + js[j][2] + js[j][4] + js[j][6]) / 4, (js[j][1] + js[j][3] + js[j][5] + js[j][7]) / 4], d = Math.hypot(c[0] - ec[0], c[1] - ec[1]); if (d < bd) { bd = d; best = j; } }
        if (best < 0 || bd > 1.5) bad++; else used[best] = 1;
    }
    return bad + Math.max(0, js.length - ex.length);
}
var shown = 0, shownId = 0;
man.forEach(function (c, ci) {
    if (stat.cases >= maxN || (filt && c.src.indexOf(filt) < 0) || (only && only.indexOf(ci) < 0)) return;
    var g = { w: c.w, h: c.h, data: readBin(FX + '/aruco/' + c.file) };
    var t0 = now(), r = OMR.aruco.detect(g); var ms = now() - t0;
    stat.cases++; stat.jsMs += ms; stat.cvMs += c.cv2_ms;
    var S = perSrc[key(c.src)] = perSrc[key(c.src)] || { n: 0, idBad: 0, rejBad: 0, errs: [], js: 0, cv: 0 };
    S.n++; S.js += ms; S.cv += c.cv2_ms;
    // ids + corners: compare by id
    var idsOk = true, byId = {};
    r.ids.forEach(function (id, i) { (byId[id] = byId[id] || []).push(r.corners[i]); });
    var cnt = {}; c.ids.forEach(function (id) { cnt[id] = (cnt[id] || 0) + 1; });
    var jsCnt = {}; r.ids.forEach(function (id) { jsCnt[id] = (jsCnt[id] || 0) + 1; });
    if (r.ids.length !== c.ids.length) idsOk = false;
    c.ids.forEach(function (id, i) {
        var cand = byId[id];
        if (!cand) { idsOk = false; return; }
        var bestE = 1e9;
        cand.forEach(function (q) { var e = 0; for (var k = 0; k < 4; k++) e = Math.max(e, Math.hypot(q[2 * k] - c.corners[i][2 * k], q[2 * k + 1] - c.corners[i][2 * k + 1])); bestE = Math.min(bestE, e); });
        S.errs.push(bestE); stat.cornerErrs.push(bestE);
    });
    if (idsOk) { stat.idsOk++; } else { stat.idsBad++; S.idBad++; if (shownId++ < 40) print('  ID DIFF ' + c.src + ' #' + ci + ' js=' + r.ids + ' cv=' + c.ids); }
    var rb = cmpSets(r.rejected, c.rejected.map(function (x) { return x; }));
    if (rb === 0 && r.rejected.length === c.rejected.length) stat.rejOk++; else { stat.rejBad++; S.rejBad++; if (shown++ < 40) print('  REJ DIFF ' + c.src + ' #' + ci + ' js=' + r.rejected.length + ' cv=' + c.rejected.length + ' unmatched=' + rb); }
});
function q(a, p) { if (!a.length) return 0; a = a.slice().sort(function (x, y) { return x - y; }); return a[Math.min(a.length - 1, Math.floor(p * a.length))]; }
print('cases ' + stat.cases + ' idsOk ' + stat.idsOk + ' idsBad ' + stat.idsBad + ' rejOk ' + stat.rejOk + ' rejBad ' + stat.rejBad);
print('corner err px (max over 4 corners): median ' + q(stat.cornerErrs, 0.5).toFixed(4) + ' p99 ' + q(stat.cornerErrs, 0.99).toFixed(4) + ' max ' + q(stat.cornerErrs, 1).toFixed(4) + ' n=' + stat.cornerErrs.length);
print('ms/call js ' + (stat.jsMs / stat.cases).toFixed(1) + ' cv2 ' + (stat.cvMs / stat.cases).toFixed(1));
Object.keys(perSrc).forEach(function (k) { var S = perSrc[k]; print('  ' + k + ': n=' + S.n + ' idBad=' + S.idBad + ' rejBad=' + S.rejBad + ' cornerMed=' + q(S.errs, .5).toFixed(3) + ' max=' + q(S.errs, 1).toFixed(3) + ' js=' + (S.js / S.n).toFixed(1) + 'ms cv2=' + (S.cv / S.n).toFixed(1) + 'ms'); });
