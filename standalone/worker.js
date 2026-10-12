// Web Worker: Pyodide + the omr package.  Messages in: {type:'init', sources, pyodideBase} | {type:'frame', id, buf, w, h} | {type:'reset'}
// Messages out: {type:'progress', text} | {type:'ready'} | {type:'error', text} | {type:'result', id, json, ms}
let py = null, pyFrame = null, pyReset = null;
self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'init') {
    try {
      const say = text => self.postMessage({ type: 'progress', text });
      say('Pyodide 런타임 내려받는 중 (약 10MB)...');
      importScripts(m.pyodideBase + 'pyodide.js');
      py = await loadPyodide({ indexURL: m.pyodideBase });
      say('numpy / OpenCV 내려받는 중 (약 15MB)...');
      await py.loadPackage(['numpy', 'opencv-python'], { messageCallback: say, errorCallback: say });
      say('인식 코드 적재 중...');
      py.FS.mkdirTree('/pkg/omr');
      for (const [name, src] of Object.entries(m.sources)) py.FS.writeFile('/pkg/' + name, src);
      py.runPython("import sys; sys.path.insert(0, '/pkg')");
      py.runPython('import omr_web; omr_web.init()');
      const mod = py.pyimport('omr_web');
      pyFrame = mod.frame; pyReset = mod.reset;
      self.postMessage({ type: 'ready' });
    } catch (err) { self.postMessage({ type: 'error', text: String(err && err.message || err) }); }
  } else if (m.type === 'frame') {
    const t0 = performance.now();
    try {
      const json = pyFrame(new Uint8Array(m.buf), m.w, m.h);
      self.postMessage({ type: 'result', id: m.id, json, ms: performance.now() - t0 });
    } catch (err) { self.postMessage({ type: 'result', id: m.id, json: JSON.stringify({ state: 'idle', hint: '', overlay: {}, error: String(err && err.message || err) }), ms: performance.now() - t0 }); }
  } else if (m.type === 'reset' && pyReset) pyReset();
};
