# J2 status: logic modules (JS port)

Command: `.venv/bin/python web/test/logic/run.py` (exports fixtures if missing; `--regen` re-recognises everything, `--no-video` skips
the iPhone devlog video 2026-10-14; ~5 min export, ~10 s jsc run). Fixtures: `web/test/fixtures/logic/` (gitignored, ~250 MB).

## Ported (web/src, behaviour-identical)
util.js (Python round/half-even, sha1, deepEqual), markers.js (all of omr/markers.py; DICT_4X4_50 bit table embedded, header whitening
constant embedded), layout.js (full layout engine + layout_for_header cache + from_dict), types.js (result constructors/helpers, see TYPES.md),
identity.js, revision.js, fusion.js, stream.js (StreamSession incl. revisions, hints, overlay, missed/unread episodes),
**decide.js** = the decision half of recognize.py (decide_page, _decide(_abs), _guess, _weak_pick, _rel_*, _calibrate, decision_sig):
fusion needs it, so J2 ported it (J3: do not duplicate; ports only the image-derived scoring).
numpy-2 float32 semantics are emulated with Math.fround where Python arrays are float32 (fills, darkness, medians).

## Result (last run)
markers 1085/1085, layout 479/479 (116 templates incl. error cases), decide 1296/1296 (real + 400 perturbed raws: noise, sentinels, NaN, unverified),
identity 1026/1026, revision 500/500, fusion 3282/3282 steps (every fused raw/decision/applied result), stream 8133/8133 frame events
(187 sequences, 7946 events, 350 commits, 18 revisions, with finish()/missed()/unread tails) - floats within 1e-9. Sequences: 18 synthetic sheets x14 jittered
frames (clean/handheld/curl/glare/dark/lowres/raking/pencil...), random programs, borderline contradiction/swap/hold sequences, and the real iPhone
video (every 3rd frame = 530 frames; also stride 9 = ~3 fps like the device, k=2, no-fuse, long-hold variants; 13 commits for 12 sheets).
Mutating a threshold in decide/stream/fusion makes the suite fail (checked), so the tests are not vacuous.

## Known differences / notes
- None found in outputs. Representation differences (TYPES.md): marker_corners is an array of [id, quad]; dewarp is a function; 2-D arrays are
  arrays of typed rows; tuples are arrays; decision_sig is a JSON string.
- Test-side `makeField` (test.js) stands in for J3's tracks Field when evaluating image_points of dewarped sheets.
- Not tested: legacy (form_id) sheets via resolve_layout, only the self-describing path.
