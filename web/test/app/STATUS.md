# J4 status: scanner.js, worker.js, app, build (JS port)

Command: `.venv/bin/python web/test/app/run.py` (`--frames N --stride S` default 60 / 9, `--no-dev`, `--keep`); ~45 s. Exit 0 = all pass.
Files: `web/src/scanner.js` (= standalone/glue.py), `web/worker.js`, `web/app.js`, `web/dev.js`, `web/template.html`, `scripts/build_web.py`,
`web/streamomr.html` (373 KB), `web/streamomr-dev.html` (400 KB, `--dev`, git-ignored like the old dev build). decide.js only gained exports
(DIGIT_MARGIN_FROM/FULL, REL_SURE, REL_SURE_DIGIT, GUESS_GAP, GUESS_REL_GAP, relTrustAnswers, relTrustDigits, relativeMarks(md) = relRows([md])[0]); logic suite still passes.

## What changed vs standalone/
- app.js: init message is `{type:'init'}` (no sources / pyodideBase), comment text; dev.js: `pyodide: PYODIDE_BASE` -> `engine: 'js'`, comments. template.html: no
  "downloading 30 MB" line, no omr-src / PYODIDE_BASE. The OMR-DETAIL-SHARED block is untouched (byte-identical to app/static/detail.js, checked).
- worker.js: same protocol (progress / ready / error / result{id,json,ms,t{xfer,worker,tx}}); init answers at once. Frames come in as transferred RGBA buffers.
- Dev timing columns keep their names (py_convert_ms = RGBA->grey, py_feed_ms = session.feed, py_other_ms = rest) so old and new frames.csv compare.
- scanner.js: RGBA->grey inline with cv2's 15-bit coefficients (bit-identical, tested), a baseline grey JPEG encoder replaces cv2.imencode (q75; the review
  image in commit.detail), `OMR.scanner.clock` is a test hook for the stream time (default performance.now()/1000).
- Build: worker source = all web/src/*.js in load order + scanner.js + worker.js in `<script type="text/plain" id="worker-src">`, Blob URL worker; DEV flag in
  scanner.js and app.js substituted by `--dev`. No external resource, no importScripts (the shared detail block still holds two server-mode fetch('/api/..') calls, never reached standalone).

## Test results
- Both built files: 2 inline scripts each parse (jsc `new Function`), no external URLs / Pyodide loader.
- RGBA->grey identical to cv2 COLOR_RGBA2GRAY (random 97x61); JPEG encoder q75/q90: cv2.imdecode ok, reconstruction error = cv2's own (0.729 vs 0.730), size within 1 %.
- Worker harness (jsc, fake self/postMessage, init -> reset -> 60 transferred RGBA frames of segment_01.mp4, every 9th frame 0..531, fake stream clock) vs standalone/glue.py
  Scanner (same frames, same clock), prod and dev builds: status / state / hint equal 60/60; JSON key sets identical (event, overlay, ms, commit/revision summary + detail,
  dev + trace); commits at the same frames [81,180,279,369,486], revision at 513; commit summaries equal (all); detail values equal except 1 score
  (answers[43].scores[4] 0.2118 vs 0.2287: recognizer-level pixel difference, not glue); detail JPEG decoded mean abs diff vs Python's 0.06-0.08 grey levels;
  missed reports 22 = 22; overlay presence 60/60.
- Dev trace (explain) values differ only where the recognizer's fused measurements differ (disc_dark / fill within a few 0.001, fill_pct_min/max/std +-0.1): printed, not asserted.
- JS time per frame under jsc here: median ~540 ms, max ~1.9 s (Python native 150 ms).

## Not tested
- No browser run: safaridriver exists but remote automation is not enabled (needs `safaridriver --enable`, admin), so the page itself (camera, canvas grab, Blob worker creation,
  UI, dev zip download, iPhone Safari) is untested; only the worker source and syntax of the inline scripts are. Needs a manual open of web/streamomr.html (file:// and https).
- OffscreenCanvas/JPEG decoding in Safari of the in-house JPEG (valid per cv2/libjpeg decode, not per Safari).
- dev.js behaviour with the new engine string (parsed only).
