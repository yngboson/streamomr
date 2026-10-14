# J1 status: cv.js + aruco.js (parity against cv2 5.0.0)

Run everything: `.venv/bin/python web/test/cv/run.py` (`--quick` = every 4th aruco case, `--bench` = timing table, `--regen` = re-export fixtures
from realasset / tests/synth.py; fixtures live in web/test/fixtures/, git-ignored). JS runs under JavaScriptCore (`jsc`), no node.
Files: `web/src/cv.js`, `web/src/aruco.js`; tests `test_contours.js`, `test_prim.js`, `test_aruco.js`, `bench.js`; exporters `export_aruco.py`, `export_prim.py`, `bench_cv2.py`.

## ArUco (aruco.detect vs cv2.aruco.ArucoDetector with omr/detect.py `_detector()` params)
Fixtures: 1144 detector calls = every grey image detect.py feeds the detector (plain / opened / flattened) for the 8 realasset photos
(+0.5/0.33 scales), mixed_ink_pencil, 20 devlog scan frames + 40 devlog failure frames, every 10th frame of devlog_2026-10-14 segment_01.mp4
(1080x1920), 60 synthetic frames (tests/synth.py: clean, tilted, blurred, dark, phone, lowres, printer, handheld, curl, glare, shadow, raking).
- Id sets equal: **1136 / 1144 (99.3%)**; real 219 calls: 218 equal, video 670: 664, synth 249: 248.
- Corners (refined, all matched markers, n=1077): median 0.0000 px, p99 0.0001 px; max 1.86 px only in the 8 id-differing calls.
- Rejected-candidate sets equal in 1102 / 1144 calls (remaining: +-1 candidate).
- Dictionary: DICT_4X4_50 codes exported from cv2 (4 rotations x 50, embedded).
Differences (all in marginal frames, e.g. partial/moving sheets): 8 calls differ. Cause found for the pattern: cv2 5.0 picks one candidate per
group of near-duplicate quads (3 threshold windows) by **longest quad perimeter** (not area; matches 2446/2450 measured groups); in ~0.2% of groups
(always window-13 vs window-23 variants) cv2 keeps the other one for a reason I could not identify. Also measured, not documented by OpenCV:
bit cell is 1 when white pixels >= half (not >), approxPolyDP uses point-to-SEGMENT distance (4.x used the line).
Timing 1080x1920 frame: aruco.detect ~100-130 ms plain (cv2 13 ms, multithreaded); on the opened image 220-400 ms incl. morphOpen
(~125 ms). Per threshold window ~18-23 ms + contours 10-17 ms. Windows are independent (could be 3 workers).

## Primitives (cv.js vs cv2; all PASS in run.py)
Exact (0 differing pixels): adaptiveThreshold, findContours (list/external, none/simple, same order), approxPolyDP (0/8573), arcLength, contourArea,
isContourConvex, integral, otsu threshold, bitwiseNot, cvtColor gray (15-bit coefficients, as cv2 5.0), divide, addWeighted, dilate/erode/morphOpen
(uint8 + float, any k incl. even), boxFilter (uint8 rounds half up), GaussianBlur uint8 (bit-exact fixed point; kernel = error-diffused 8-bit
coefficients, found by impulse responses), resize INTER_AREA integer scales (incl. edge pixels averaged over the pixels that exist) and fractional.
Within float noise: GaussianBlur float32/1-row float (<2e-4), resize float, boxFilter float, laplacian, matchTemplate TM_CCOEFF_NORMED (<5e-5),
getPerspectiveTransform, perspectiveTransform, findHomography (DLT + LM like cv2; <5e-5 abs on entries of size 100s), cornerSubPix (0.0 on tests).
Not exact: resize INTER_LINEAR uint8 upscaling: 0.5% of pixels off by 1 (SIMD fixed-point path); warpPerspective/remap (float bilinear as
cv2 5.0 does; J3's finding) differ by <=3 in 0.003% of pixels (rounding ties), nearest warp 1 level in 0.001%.
Fixed J3 finding 1: no more 1/32 fixed-point table (the Int16 overflow is gone).
Timings (single call, 1080x1920, quiet machine; this laptop ran other agents so numbers vary 1.5-3x, and one jsc process running many ops
in sequence can mis-tier the JIT - run `bench.js` with a filter arg for clean numbers): adaptiveThreshold 18-23 ms (cv2 3-4), findContours 17 (cv2 11),
resize area x0.25 6 (1.0), gaussian u8 sigma1.2 100 (1.3) / sigma3 190 (3), f32 80, dilate k9 64 (0.4), open k9 125 (0.9), boxFilter f32 22 (6),
laplacian 22 (0.7), integral 11 (3), otsu 11 (1.8), warp 1.4MP 44 (0.9), remap 37 (2), subPix 4 corners 0.08 ms.
JIT note: one loop that sees both Uint8Array and Float32Array (or Uint8Array and Int8Array) runs 5-10x slower in JavaScriptCore, so hot kernels
(gaussian, morphology, boxFilter, area resize, adaptiveThreshold) exist as one copy per element type.

## cvx.js (J3's interim copies) vs cv.js, on the same fixtures (lead: delete the duplicates, point J3 at cv.js)
- warpPerspective / remap / remapLinear: equal (both float bilinear); cv.js adds 'nearest' interp and is the one run.py covers. cvx has no nearest.
- resize (area + linear): cv.js better: cvx differs from cv2 by 1 in up to 66% of pixels for fractional area scales and misses edge handling
  (15 of 68 area cases fail, uint8 linear 8-13% of pixels off by 1); cv.js exact for area, 0.5% off-by-1 for linear upscale.
- gaussianBlur: both pass on the fixtures now; cv.js is the bit-exact uint8 path (cvx passed within its tolerance here).
- dilate/erode/morphOpen: cv.js better: cvx fails even kernel sizes (6/6 fail), equal for odd k; cv.js also has `into` (no allocation) and is faster.
- matchTemplateCcoeffNormed: cv.js better: cvx returns 0 instead of cv2's 1 for flat window+flat template; otherwise equal.
- laplacianVar: cv.js has `laplacian` (exact) + `laplacianVar` (population variance); cvx.laplacianVar not separately fixtured.
- cvx-only (not in cv.js, keep): percentile, median, mean, lstsq, gaussianBlurRow.
