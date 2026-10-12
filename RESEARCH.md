# StreamOMR accuracy research (2026-10-13)

Method: think -> search -> think, 3 cycles, then implement the top ideas one at a time, each measured against
`research-base/` (pristine copy) with the bench scripts.  Rule: keep only clear wins with no regression in wrong ID (0),
silent errors, real photos 9/9 and time.

## Baseline (research-base)
Real photos (`scripts/photo_bench.py`): 9/9 at scale 1 / 0.5 / 0.33 and rot 90, answers 109/109, max 334 ms, mean 262 ms
(24 MP).  Real photos are saturated -> gains have to be shown on the synthetic hard conditions (header curl/raking/h_all,
geom fold/curl silent errors, faint, illum --hard) and on corner accuracy (marker_bench `cerr`).
(Synthetic baselines filled in below as they finish.)

## Cycle 1 - THINK (from code + handoff failure notes)
Where the pipeline loses frames / makes errors today (README log):
* Header decode is weakest under paper shape, not light: curl 62 %, raking 62 %, h_all 52 % (header_bench).  Diagnosis
  recorded by the previous agent: only 3 markers visible -> homography extrapolated -> one end of the header off by
  2-5 mm -> `header_reg_mm` rejects.  Tracks are only used *after* a failed header read.
* geom bench: residual silent errors on fold/curl; illum --hard silent errors 2; faint OK.
* Real photos: saturated (9/9), time ~260 ms mean (budget 500 ms).

Candidates (gain / cost / risk):
1. **Marker corner refinement by edge-line fitting** (fit 4 lines to the marker's outer edges at full resolution,
   intersect) instead of `cornerSubPix` on an L-corner.  cornerSubPix on a blurred L-corner is biased inward;
   corners feed the homography, reprojection check (MAX_REPROJ_MM) and the 4th-marker prediction.  Gain: medium
   (accuracy of everything downstream, esp. the 3-marker extrapolation); cost low; risk low.  Measurable with
   marker_bench `cerr` + header/geom benches.
2. **Homography from more points**: use the interior of each marker (the 4x4 bit-cell grid corners = up to 25 more
   points per marker) or the timing-track ticks (v2 sheets: 132 ticks along the edges) for a *global* homography with
   all points, instead of 12-16 marker corners.  Gain: medium-high for 3-marker frames (no extrapolation: ticks along
   the 4th side constrain it); cost medium; risk: ticks are found on the warp already, so it's a 2-pass loop.
3. **Header soft-bit fusion across frames** (stream): accumulate per-cell darkness margins across frames of one track
   and decode the CRC on the fused soft bits (patents: Cognex "reading patterns using multiple image frames").
   Gain: stream-level header rate; cost medium (stream.py plumbing); risk low (CRC still judges).  Not measurable on
   the single-frame benches.
4. **Header read after track rectification for 3-marker frames** (instead of only after a failure): gain on curl/raking
   header; cost: tracks.measure time on those frames; risk low.
5. **Header error correction** (Reed-Solomon / BCH in a new header version): CRC-16 only detects; with ECC 1-3 bad
   cells would be corrected.  Needs a new template version (old sheets keep CRC).  Gain medium; cost medium; risk:
   wrong-layout acceptance must stay ~0 -> keep CRC over ECC.
6. **Higher-order / local alignment model** (cubic or TPS on ring offsets) for paper deformation, when many inliers.
   Gain: fold/curl interior; risk: overfitting to marked bubbles -> lattice slip (silent).  Tracks already cover much.
7. **Soft multi-frame fusion weighting** (weight frames by sharpness / ring corr instead of plain median).  Gain small.
8. **Robust homography (USAC_MAGSAC)** over marker corners: with 12-16 points from 3-4 markers, outliers are whole
   markers, not points; LSQ is fine.  Low value.
9. **Detection speed**: AprilTag-style decimation is already done (find_markers_scaled).  Low value now.
10. **Lens-distortion model** (phone wide camera k1): absorbed by quadratic ring alignment.  Low value.

## Cycle 1 - SEARCH
* OpenCV ArUco docs / OpenXR spec: SUBPIX = fast, reasonable accuracy; CONTOUR = line fitting to contour points;
  APRILTAG = slowest, most accurate.  https://docs.opencv.org/5.0/main_modules/objdetect_aruco.html ,
  https://registry.khronos.org/OpenXR/specs/1.0/man/html/XrMarkerDetectorCornerRefineMethodML.html
* TopoTag paper: AprilTag beats ArUco on small marker angles "thanks to its edge refinement" (vertices = intersections
  of fitted lines). https://arxiv.org/pdf/1908.01450
* Zea & Hanebeck (KIT, Fusion 2019) "Refined pose estimation for square markers using shape fitting": fit the polygon
  to contour pixels by nonlinear LSQ, gives corners + covariance, reduces jitter.  https://isas.iar.kit.edu/pdf/Fusion19_Zea.pdf
* STag: initial homography from square border, then *homography refinement from an inner circle* -> stable pose.
  https://arxiv.org/html/1707.06292  (=> supports idea 2: more geometry than 4 corners).
* AprilTag 3 `refine_edges`: snap quad edges to strong gradients after decimated detection - exactly our
  find_markers_scaled situation (detect on 1920 px copy, refine at 24 MP).  https://pupil-apriltags.readthedocs.io/en/latest/api.html
* Multi-frame barcode reading (Cognex patents US9734376 / US10592715 / EP2210211): accumulate per-cell 0/1 evidence over
  frames, decode after each frame; frames individually undecodable become decodable together.
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9734376  (=> supports idea 3).
* Dewarping: TPS needs control-point conditioning, overshoots; polynomial models fine for mild smooth distortion
  (Fourier Document Restoration https://arxiv.org/pdf/2203.09910 uses 9x9 TPS mesh; CVPR 2022 grid regularization
  https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_Revisiting_Document_Image_Dewarping_by_Grid_Regularization_CVPR_2022_paper.html).
  => idea 6 risky; our Coons patch from edge ticks is already a mesh method.
* USAC/MAGSAC++ (https://docs.opencv.org/4.x/de/d3e/tutorial_usac.html): helps with many outlier-contaminated matches;
  not our case (few, clean points).  => drop idea 8.

## Cycle 1 - THINK again (re-rank)
1. Edge-line corner refinement (1) - cheap, measurable with marker_bench cerr, literature-backed.
2. Tracks/ticks-assisted global geometry for 3-marker frames (2+4) - targets the documented weakest number (header curl/raking).
3. Header soft-bit multi-frame fusion (3) - stream-level, needs a stream bench.
4. Header ECC in a new version (5) - later unless 1-3 stall.
5. Higher-order alignment (6) - risky, defer.
Dropped: 8 (MAGSAC), 9 (speed), 10 (lens model).

## Cycle 2 - THINK (failure data, `hdr_diag.py` / `trk_diag.py` / `partial_diag.py` in the scratchpad)
Header/geom failures on curl, raking, h_all, broken down by cause (baseline, 40 frames each):
* **4th marker missed on curled sheets.** Example curl seed 1033: the TL marker is fully visible (a white toner streak
  breaks its border), but the 3-marker homography puts it **80 px = 1.6 marker sides** from where it really is. `_complete`
  only searches +-1.3 sides and accepts +-0.9 sides, so it never looks there.  An opening with k=5..7 on the right ROI
  decodes it right away.
* **Then curl frames with 4 markers fail geometry** (4-marker reprojection 2.3-4.3 mm because the curled corner's marker
  is not on the plane of the other three) and the timing tracks do not save them: the top track's DP chain jumps onto
  header row 1 (2.4 mm below the ticks) where the top edge curls, so the field is wrong right at the header.
* **Partial frames (2 markers)**: fold_big = the folded-away half's markers are slivers (physically unreadable, accept);
  fold_hand = 1 hidden by the hand + 1 visible marker missed (window search from the true position finds it).
* Real photos are saturated (9/9).

## Cycle 2 - SEARCH
* cornerSubPix = Forstner/Gulch gradient-orthogonality corner; gradient methods are noise-sensitive; Duda & Frese
  (BMVC 2018, localized Radon transform + box filters) robust to blur/low contrast with high subpixel accuracy:
  https://www.krakenrobotics.com/wp-content/uploads/2019/06/2018-Alex-Duda-Nov-2018.pdf ,
  https://docs.adaptive-vision.com/studio/filters/FeatureDetection/cvCornerSubPix.html .  For an *L-corner* of a
  marker (not an X/saddle corner) blur rounds the corner and pulls a gradient corner inward, while straight sides stay
  straight -> fitting lines to the sides (AprilTag/ArUco CONTOUR style, see TopoTag above) is the right tool.
* OMR timing-mark tracking with DP: no prior art found (patents only use timing tracks for skew/slip checks:
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5600117 ; camera OMR patent notes page curl + lens
  distortion as the open problem: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7573616).

## Cycle 2 - THINK again (re-rank)
1. **A: wider 4th-marker search + 2-marker completion** (new; directly from the data, cheap: only frames with 2-3 markers).
2. **B: drop the worst-fitting marker and retry** when a 4-marker frame fails geometry/header (the curled-corner marker).
3. **C: edge-line corner refinement** (literature + marker_bench cerr).
4. Header soft fusion across frames / ECC version - deferred until A-C are measured.
5. Top-track/header confusion in tracks.py DP - real but deeper (risk of re-tuning a DP); deferred.

## Cycle 3 - SEARCH (lead asked for an explicit pass before more implementation)
* Corner localization: Duda & Frese (above) and its use in chess-corners (https://docs.rs/chess-corners/) - Radon
  detector "useful in the small-cell, blur and low-contrast fixtures".  Our markers' outer corners are L-corners, so
  the side-line fit stays the better fit; the half-level crossing profile is the 1-D analogue of an erf edge model.
* Dewarping from boundaries: Coons patch from page boundary curves (Brown & Seales JCDL 2003,
  https://www.eecs.yorku.ca/~mbrown/pdf/jcdl_2003.pdf) - exactly our tracks.py model; cylinder-model book dewarping
  (https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9992471) would fit "curl" better than a bilinear
  interior, but our failure is the *edge* measurement (top track vs header row), not the interior model.  => later.
* Multi-frame soft decoding: per-cell 0/1 evidence accumulated over frames (Cognex, cycle 1) and summing LLRs of
  several copies (https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8250444); LDPC 2D barcodes with soft
  decisions beat RS Data Matrix (https://dspace.upt.ro/jspui/bitstream/123456789/610/3/BUPT_TD_Pross%20Wolfgang.pdf).
  => For our header: sum per-cell signed margins over the frames of one track, then CRC (+ the existing 1-2 flips).
  Needs stream plumbing and a stream bench; header_bench is single-frame. => "later", recorded with a design.

## Cycle 3 - THINK (final ranking)
| # | idea | evidence | status |
|---|------|----------|--------|
| 1 | A: wide / 2-marker completion of missing corner markers | curl seed 1033 off by 1.6 sides | implemented |
| 2 | B: drop worst marker and retry on bad_geometry/bad_meta | curled-corner marker off-plane | implemented |
| 3 | C: edge-line (AprilTag-style) marker corner refinement | TopoTag, Zea & Hanebeck, Duda & Frese; cerr -10..20 % | implementing |
| 4 | Header soft-bit fusion over a track | Cognex patents, LLR combining | later (design below) |
| 5 | Header ECC (BCH/RS) in a new header version | LDPC/RS barcode literature | later |
| 6 | tracks.py: keep top chain off the header row (header-row-aware DP) | trk_diag curl 1033 | later |
| 7 | cylinder curl model for the interior | book-dewarping patents | later |
| 8 | MAGSAC / lens model / speed | not our failure mode | dropped |

## Implementation results (each vs research-base, same seeds)
Bench tools in the scratchpad (not committed): `hdr_diag.py` (header failures by marker count / cause, wrong-form_key
count), `trk_diag.py`, `partial_diag.py`, `marker_bench.py --extra` with a refine method.

### KEPT A - wider / two-marker completion of missing corner markers (`omr/detect.py` `_complete`, `_search_window`, `COMPLETE_WIDE`)
* 3 markers found: the old window (1.3 / 0.9 sides) first, then a wide one (2.6 / 2.2 sides).
* 2 markers of one page found: the two predict the others through an **affine** map (two squares cannot pin a perspective)
  and each missing marker is searched in the wide window; one more marker found -> re-predict (2 -> 3 -> 4).
* Only an exact dictionary decode of the *missing* id is accepted; the reprojection / tracks / header CRC / ring checks
  downstream are unchanged.
### KEPT B - drop the worst-fitting marker and retry (`omr/recognize.py` `recognize` / `_worst_marker` / `_recognize`)
* A 4-marker frame that ends `bad_geometry` or `bad_meta` (not page/layout mismatch) is re-run once without the marker
  whose corners the 4-marker homography reprojects worst (a curled corner's marker is off the plane).  Success ->
  soft flag `marker_dropped` (same trust as `corner_hidden`).  `tests/test_recognize.py::test_three_markers_geometry_is_checked`
  now allows "read from the three genuine markers, flagged, no silent error / wrong ID" for a pasted wrong-place marker.

| bench (base -> A+B) | base | A+B |
|---|---|---|
| header curl / raking / h_all (n=40) | 82 / 88 / 65 % | **88 / 92 / 68 %** (others unchanged, 100 % clean/dark/glare/shadow) |
| header 8 lighting conds, n=120 each, wrong form_key | - | 899/960, **0 wrong** |
| geom ALL read% / clear-mark acc% / ID read% | 86.2 / 96.61 / 87.2 | **86.9 / 97.67 / 88.1** |
| geom curl clear acc% / h_all clear acc% | 93.95 / 89.87 | **98.71 / 90.41** |
| geom silent / silent-clear / ID wrong | 9 / 3 / 0 | 9 / 3 / 0 |
| illum (1020 fr) / --ink (420) / --hard (360) silent | 1 / 62 / 12 | 1 / 62 / 12 (hard read 91 -> 92 %) |
| faint on: silent faint / faint_glare | 2 / 0 | 2 / 0 (identical) |
| ID wrong (all benches) | 0 | 0 |
| real photos 1 / 0.5 / 0.33 / rot 90,180,270 | 9/9 all | 9/9 all |
| real photo time max / mean (24 MP) | 344 / 263 ms | 330 / 258 ms |
| 1080p single-thread: thumb (1 hidden) / thumb2 (2 hidden) / printer | 289 / 185 / 256 ms | 295 / 229 / 257 ms |

### REVERTED C - edge-line (AprilTag-style) corner refinement
Half-level crossing profile across each marker side at full resolution, Huber line fit, intersect (vectorised: 6 ms
per 4 markers at 24 MP).  marker_bench corner error (modules): printer 0.133 -> 0.120, blurred 0.173 -> 0.140, lowres
0.146 -> 0.125, clean 0.053 -> 0.046.  But downstream it did not help and added silent errors: illum --ink 62 -> 64,
illum 1 -> 3, faint_glare 0 -> 2 (geom 9 -> 8).  The ring alignment already absorbs < 0.1 mm of corner error; the
change only reshuffles borderline marks.  Not worth it.  (Would matter for pose estimation, which we do not use.)
### REVERTED D - one-end header registration fallback
Glare / shadow over one end of the header: register and threshold from the other end's guards only (CRC judges).
n=120 x 8 lighting conditions: 899 -> 901 / 960, 0 wrong form_key.  +0.2 % for two more CRC attempts per failing
frame: not worth the (tiny) false-accept exposure.  Remaining header failures are data cells washed out by glare
(see h_glare seed 1029) -> needs multi-frame fusion or ECC.

## Later (designs, not implemented)
1. **Header soft-bit fusion over a stream track** (top recommendation for "QR" read rate under moving glare):
   `_read_header_pass` exposes per-data-cell signed margins `(D - thr) / (B - W)` whenever contrast and registration
   are OK; the stream keeps a running sum per cell for frames whose marker quad does not jump (fusion.py's track
   rule); after each bad_meta frame decode the CRC on the summed signs (+ the existing <= 2 weakest-cell flips).  A
   decoded header from the fused bits is only a *hint*: recognize(image, header_hint=h) must re-check that the frame's
   confident cells agree with h (0 confident disagreements, >= 85 % of cells confident) before using its layout.
   Needs: recognize API + app/main.py + standalone/glue.py plumbing + a multi-frame header bench.
2. **Header ECC** (BCH(…)/RS over the data cells) as header version 3; old sheets keep CRC-only.  CRC kept on top.
3. **Header-aware top-track DP** (tracks.py): penalise a chain that lands on header row 1 (known y offset 2.4 mm);
   curl seed 1033 shows the jump.  Would turn some 4-marker curl frames from bad_geometry into reads.
4. Cylinder curl model for the interior (book-dewarping patents) when tracks show a single-axis bend.
5. 2-marker completion speed: on 2-hidden frames it costs +44 ms (1080p); could stop after the first missing marker
   fails in the wide window if streams show it matters.
