// Web Worker: the JS recognizer (web/src/*.js and scanner.js are inlined in front of this file by scripts/build_web.py).
// Messages in: {type:'init'} | {type:'frame', id, buf, w, h, tx} | {type:'reset'}
// Messages out: {type:'progress', text} | {type:'ready'} | {type:'error', text} | {type:'result', id, json, ms, t}
// Same protocol as the old Pyodide worker; no download, so init answers at once.  Dev timing columns keep their old names
// (py_convert_ms / py_feed_ms / py_other_ms in frames.csv now measure the JS grey conversion / session.feed / the rest).
var scanner = null;
self.onmessage = function (e) {
  var m = e.data;
  if (m.type === 'init') {
    try {
      self.postMessage({ type: 'progress', text: '인식 코드 준비 중...' });
      scanner = new OMR.scanner.Scanner();
      self.postMessage({ type: 'ready' });
    } catch (err) { self.postMessage({ type: 'error', text: String(err && err.message || err) }); }
  } else if (m.type === 'frame') {
    var t0 = performance.now(), rx = performance.timeOrigin + t0;   // rx - m.tx = transfer + queue wait (epoch clocks)
    var t = function () { return { xfer: rx - m.tx, worker: performance.now() - t0, tx: performance.timeOrigin + performance.now() }; };
    try {
      var json = scanner.stepRgba(new Uint8Array(m.buf), m.w, m.h);
      self.postMessage({ type: 'result', id: m.id, json: json, ms: performance.now() - t0, t: t() });
    } catch (err) {
      self.postMessage({ type: 'result', id: m.id, json: JSON.stringify({ state: 'idle', hint: '', overlay: {}, error: String(err && err.message || err) }), ms: performance.now() - t0, t: t() });
    }
  } else if (m.type === 'reset' && scanner) scanner.reset();
};
