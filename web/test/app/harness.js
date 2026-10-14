// jsc web/test/app/harness.js -- <worker source file> <frames manifest json> <out file>
// Evaluates the built worker source with a fake self/postMessage, sends the init/reset/frame messages the app sends
// (frames as transferred ArrayBuffers) and prints one line per message posted back:  @{...}
(function (A) {
    var workerSrc = A[0], man = JSON.parse(readFile(A[1]));
    var posted = [], clock = 0, t0 = Date.now();
    globalThis.self = globalThis;
    globalThis.postMessage = function (m) { posted.push(m); };
    if (typeof performance === 'undefined') globalThis.performance = { now: function () { return Date.now() - t0; }, timeOrigin: t0 };
    load(workerSrc);
    if (typeof self.onmessage !== 'function') throw new Error('worker did not set self.onmessage');
    function send(m) { self.onmessage({ data: m }); }
    function flush() { var out = posted; posted = []; return out; }
    send({ type: 'init' });
    flush().forEach(function (m) { print('@' + JSON.stringify(m)); });
    send({ type: 'reset' });
    OMR.scanner.clock = function () { return clock; };     // deterministic stream time (same fake clock on the Python side)
    for (var i = 0; i < man.frames.length; i++) {
        var f = man.frames[i], u8 = readFile(f.path, 'binary');
        var buf = u8.buffer.byteLength === u8.length ? u8.buffer : u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.length);
        clock = f.t;
        send({ type: 'frame', id: i + 1, buf: buf, w: f.w, h: f.h, tx: performance.timeOrigin + performance.now() });
        flush().forEach(function (m) {
            var o = { type: m.type, id: m.id, ms: m.ms, t: m.t, json: m.json, text: m.text };
            print('@' + JSON.stringify(o));
        });
    }
})(arguments);
