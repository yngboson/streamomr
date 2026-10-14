# J3 status: image pipeline port (detect, fastops, illum, ink, tracks, recognize) + end-to-end tests

Command: `.venv/bin/python web/test/pipe/run.py [--sets photos,devlog,video,synth] [--parity] [--no-time] [--no-stream] [--reuse] [--regen]`
(exports grey frames + Python results to `web/test/fixtures/pipe/` on first run, runs JS under jsc in parallel, compares, prints timing).
Tools: `one.py <set> <idx>` (diff one frame), `prof.py <set> <idx> [reps]` (function profile), `stage_export.py` + `stage.py` (Python intermediates fed to JS stages).
`--parity` runs detect.js with omr/detect.py's exact pass sequence (no pass-policy optimisation).

## Ported (web/src)
detect.js, fastops.js (pctRows), illum.js (method E), ink.js, tracks.js, recognize.js (everything except decide.js), cvx.js (what remains of the interim
primitives: percentile/median/lstsq/gaussianBlurRow/sample; resize/morphology/gaussian/matchTemplate/warp now come from cv.js, duplicates deleted).
Entry: `OMR.recognize.recognize(gray, {resolveLayout, pxPerMm, keepImages})` -> TYPES.md result (`dewarp` is a function with `.field`).
Load order: ... cv, aruco, cvx, fastops, ink, illum, tracks, detect, recognize.
For J4: DEV `explain()` needs `relTrustAnswers/relTrustDigits/relativeMarks` + the REL_*/GUESS_* constants exported from decide.js (not exported today).

## Parity with Python (same grey input) - default (optimised) policy
| set | frames | fields | differ | agreement |
|---|---|---|---|---|
| photos (8 jpeg + mixed) | 9 | 277 | 0 | 100 % |
| devlog 2026-10-13 | 20 | 1300 | 0 | 100 % |
| video 2026-10-14 (8 fps) | 424 | 22800 | 197 | 99.14 % |
| synthetic (35 conds, 120 frames) | 120 | 6116 | 54 | 99.12 % |
| total | 573 | 30493 | 251 | **99.18 %** |
Numbers: marker corners max 1e-5 px on photos/devlog; homography rel 1e-7; fills within 5e-4 except where a pixel-rounding tie flips (see below).
Accuracy vs truth: photos 9/9 (python 9/9); devlog single frame 900/900, ids/subjects 20/20, and fused through the JS StreamSession 900/900, 20/20 (identical
records to Python's); synthetic: silent errors 0 = Python's 0, id_wrong/subject_wrong/serial_wrong 0 = 0, but 40 answers + 1 id fewer correct (one frame, below).
`--parity` gives the same numbers within 1 field (the pass policy changes nothing measurable).

## Every known difference (all explained)
1. synth `curl_hand.sheet.q40_nosubj.8516` (53 fields, the only synthetic loss, so "JS >= Python accuracy" is NOT met on the synthetic sum): Python decodes its
   header only in the third (tracked) pass by flipping 2 cells whose margin is 0.0005. Each header pass matches Python exactly when fed Python's canvas;
   JS's canvas differs in 0.03 % of pixels by 1 grey level (cv2 5.0's warpPerspective rounding ties; neither float32 nor fixed-point models reproduce it)
   which is enough on this knife-edge frame -> bad_meta. Not a logic difference.
2. video `t015.270` (61 fields, ok -> bad_meta): marker 1 corners differ 5.9 px: J1's aruco picks another candidate of a near-duplicate group (cv2 behaviour J1 could not identify).
3. video `t051.010` (63 fields, partial -> ok): JS finds a 3rd marker (aruco/rescue differences); JS result is the better one.
4. video `t004.634`-type frames: same cause (marker found through model_decode instead of ArUco, corners 2.3 px apart) -> alignment-unverified rows differ; ~35 fields.
5. ~35 single-field flips on rows that are already ambiguous/guessed (fill within 0.01 of a threshold; pixel-rounding of cv2 resize-linear in the flattened
   copy (cv2 5.0 resize LINEAR uint8 not reproducible, tie rule depends on pixel parity) and warp rounding).
6. StreamSession on the video: Python 17 records, JS 16: follows from 2/3 above (one commit missing/extra), not a stream logic difference (stream replay on identical
   results was verified identical by J2 and on devlog here).
Numpy fp conventions that mattered and are replicated: np.arange values (start + i*((start+step)-start)), np.linspace, float32 percentile lerp, float32 `where`/fround steps.

## Optimisations (jsc, ms; this laptop was shared with other agents, so +-30 %)
| change | gain | accuracy effect |
|---|---|---|
| cvx morphology: van Herk, one copy per op | open 2MP 470 -> ~100 (then cv.js's version used) | exact (brute-force checked) |
| laplacian variance interior fast path | 107 -> 3 | none |
| rectify: coarse field interpolated inside the remap (no 2 full float maps) | 200 -> 60 | none (same weights) |
| canvas Gaussian: fused 11-tap kernel in cv.js; blurred only around the bubbles (bbox + 12 mm) | 135 -> ~60 | bit-identical where read |
| area resize: horizontal pass once per row, tap tables | 74 -> 35 each | exact |
| pass policy A: drop the opened pass on the flattened copy (never added a marker) | -1 full pass (morph+aruco, ~250) on partial/inconsistent frames | 0 id-set changes on 553 frames |
| pass policy B: with a decoded marker open only at ITS scale | -1 full pass | 0 changes except video t039.641 where Python's extra id 17 (a spurious marker on a thumb/texture, page group 4) disappears |
| pass policy C: no marker decoded -> open at the scale of the marker-like rejected candidates (1 step) instead of the ladder | no_sheet 1140 -> ~430 | 0 changes |
| coarse-to-fine detection (half size first) | -110 on ok frames | REJECTED: corners move up to 3.3 px, agreement 98.3 %; left as `policy.coarse=false` |
Evidence (Python replay of detect with pass logging, 553 frames): ok frames: 626 window searches found a marker in 1 (video); partial frames: 336 window runs + 62 flat-opened + 130 opened
calls found 1 marker in total; no_sheet/inconsistent: 0 of 60. `no_flatlast` was tried and rejected (loses a solid set), `no_complete2` rejected (loses one synthetic frame), fewer complete-kernels rejected.
Not tuned numerically anywhere (no threshold loops).

## Timing per frame class (video frames, 8 per class, best of 3, jsc vs cv2-native Python, same machine)
| class | python native | JS (jsc) |
|---|---|---|
| ok | 152 (max 222) | 592 (max 1003) |
| partial | 129 | 488 |
| no_sheet | 83 | 435 |
| inconsistent | 114 | 466 |
JS does NOT beat the phone's Pyodide numbers (0.27 ok / 0.37 partial) here: 3-4x slower than cv2-native on this Mac. Warm ok-frame profile (ms): aruco.detect 148, canvas blur ~60,
warp 63, rectify 60, area resizes 55, matchTemplate 36, adaptive threshold (in aruco) 67, illum 33, header 25, tracks.measure 21. Partial frames are now cheaper than ok frames in JS
(raw + 1 opened + flat-last + windows) instead of costlier. Largest remaining levers: aruco.detect (3 threshold windows share one pass), warp/rectify in one resampling,
detector windows reuse across frames (glue level).

## Speed round (J5)
Tools: `bench.py [n] [reps]` (JS-only per-class timing, best of 3, two throw-away warm-up frames first), `stages.py <idx> [reps]` (per-section minimum of one frame),
`prof.py` (function profile), `web/test/j5/` (`jsc.sh` runs any script with the sources loaded, `jsc.sh x.js --sample -- ...` = jsc's sampling profiler with `loop.js`,
`cmp.py <ref src dir> [n]` = **bit-identity check**: json + canvas hash of ~100 frames from all sets against a snapshot of the sources, `morph_t.js` / `at_chk.js` = brute-force
checks of the new morphology / threshold kernels). **Machine note:** this Mac is an Intel i9-9880H (2019) shared with other agents (timings +-20 %); every change below was
A/B-checked in the same load conditions. An iPhone core is substantially faster than this one for this kind of code, so the absolute jsc numbers are pessimistic.
Every change except one (none) is **bit-identical** to the code before it (cmp.py: 0 differing frames of 99, incl. canvases and all fields); the full `run.py` after all changes is
identical to the baseline: photos 100 %, devlog 100 %, video 99.136 % / 197 diffs, synth 99.117 % / 54 diffs, same 3 status differences; cv `--quick`, logic, app tests pass.

| change | gain (jsc ms, warm 1080x1920) | accuracy |
|---|---|---|
| warpPerspective: truncation instead of floor/rint on the interior, hoisted matrix entries | warp 62 -> 49 | bit-identical |
| rectify: interior fast path, `all-outside -> border` shortcut, per-run hoisting of the coarse-map values | rectify ~61 -> ~55 | bit-identical |
| resizeAreaGeneral: horizontal row computed once (4-row cache), taps padded to 3/4 and unrolled | 1.5x resize 35 -> 30, 0.29x (sharpness) 19 -> 14 | bit-identical |
| alignment image (inverted canvas at 4 px/mm): lazy, tile by tile (`cv.lazyAreaResize`), only where rings/ring scores are read; NOT + resize of the full canvas gone | align 77 -> 28 | bit-identical |
| matchTemplate: four neighbouring windows per pass (own running sum each, same order) | 13 -> 8 | bit-identical |
| illum.contrastField: k-nearest by insertion instead of 285 full sorts | illum 32 -> 13 | bit-identical |
| pctRows / flattenHeader percentiles: quick-select instead of sorting every row | -6 | bit-identical |
| morphology (erode/dilate/open, 8-bit): SWAR, 4 pixels per 32-bit word (horizontal: 4 image rows packed per word), van-Herk-style doubling | erode 9: 64 -> 25, open 9: 132 -> 49; first call 140-430 -> 64 (smaller functions tier up sooner) | bit-identical (brute-force checked) |
| 11-tap Gaussian (canvas smoothing sigma 1.5 uint8, flatten sigma 1.2 float): 4 outputs per iteration horizontally, 4 rows per iteration vertically (load-bound) | full-frame uint8 107 -> 61; blur stage 55 -> 40; float 70 -> ~30 | bit-identical |
| flattenLighting tail fused (max(bg,8), divide, blur, clip, truncate) | ~ -3 (no copies) | bit-identical |
| adaptiveThreshold: fused column-update/slide, then 32-bit word loads (little-endian, width % 4 == 0) | 20 -> 15 per window (x3 per detector call) | bit-identical (brute-force + cv suite) |
| findContours: background skipped four pixels per step | 8-10 -> 6.5-8.7 per window | bit-identical |
| scanner.js JPEG encoder (review image on commit): unrolled DCT, Uint8Array output, clz32 | 173 -> 108 per page | bit-identical bytes |
| `OMR.recognize.warmup(reps)` (new, not called by anything yet): runs the kernels once on synthetic 1080x1920 data | first three frames 728/586/405 -> 499/363/353 (reps 2, costs 2.4 s; reps 1: 592/423/354, 1.3 s) | none (no state touched) |
Tried and dropped: integral-image adaptive threshold (24 ms/window, slower), tile skipping on flat areas (only ~20 % of tiles skippable), four-accumulator
matchTemplate (superseded by the exact 4-window version), a different bilinear arithmetic form for warp (no change: the loop is gather/convert bound), warp+rectify as one
resampling (per-pixel cost of the two coordinate computations alone is about that of the two passes; would also change canvas values), word-wise contour scan inside the
detector alone (no gain), integral-image sums for blur/box (no gain).

### Final per-class table (video frames, 8 per class, best of 3, jsc, this loaded Intel Mac; before = J3's table)
| class | python native | JS before | JS now | target |
|---|---|---|---|---|
| ok | 152 | 592 | 386 (warm frames 330) | 200 |
| partial | 129 | 488 | 313 | 180 |
| no_sheet | 83 | 435 | 204 | 100 |
| inconsistent | 114 | 466 | 306 | - |
In the real WebKit worker (smoke harness, 1080x1920 ok frame): 920-1050 -> 385-390 ms steady, commit frame 1586 -> ~590 ms (JPEG 173 -> 108 and the rest).
**Targets not reached; where the time goes (ok frame, warm, ms):** detector 75 (3 adaptive thresholds x 15 + contours 20 + polygon/bit work), warp 49, rectify 55,
canvas blur 40, tracks.measure 20, alignment 28, homography+sharpness 17, illum 13, ink/ring scores 14, header 9. Every one of these is already at the loop-throughput limit
of JavaScriptCore on this CPU (a trivial pass over 2 MP costs 3-5 ms; the thresholds are 7 memory operations per pixel; bilinear gathers 20+ ns/pixel); the remaining ways down are
(1) running the three threshold windows / the warp+rectify bands in parallel workers (no SharedArrayBuffer on Pages, so frame copies ~2 ms each - the kernels are pure functions of
(image, window) and can be moved as they are), (2) a WebAssembly SIMD build of the five kernels (threshold, contour scan, bilinear gather, blur, morphology), (3) policy-level
pruning of partial/no_sheet passes (the flat probe + opened pass are 2/3 of a no_sheet frame; J3's counts say they found a marker in 0 of 60 such frames, but they matter for dim
synthetic frames, so this was NOT touched), (4) tracking markers across frames at the glue level (search the previous quad's neighbourhood first).
