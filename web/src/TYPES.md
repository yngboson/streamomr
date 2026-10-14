# JS shapes of the recognizer results (contract between J2 logic modules and J3 recognize.js)

Plain objects, SAME snake_case field names as `omr/recognize.py`. Constructors with the Python defaults live in `web/src/types.js`
(`OMR.types.RecognitionResult(status)`, `QuestionResult`, `DigitResult`, `PageRaw`); Python properties are functions there
(`needsReview(r)`, `idGuessed(r)`, `subjectGuessed(r)`, `answerOf(a)`, `imagePoints(r, ptsMm)`) because plain objects are shallow-copied
and edited (fusion/revision), so nothing is computed at construction. `null` = Python `None`. Nothing may be `undefined`.

## RecognitionResult
| field | JS |
|---|---|
| status | string: ok / no_sheet / partial / inconsistent / unknown_form / bad_meta / bad_geometry |
| form_key, form_id, page, total_pages, serial | string/number or null |
| answers | array of QuestionResult (sorted by number) |
| student_id | string or null; id_digits: array of DigitResult |
| subject | string, `""` = form has none, null = unreadable; subject_digits: array of DigitResult |
| flags | array of strings |
| marker_ids | array of ints (sorted) |
| **marker_corners** | **array of `[id, [[x,y]x4]]` in detection (insertion) order** (Python: dict id -> quad; JS objects would reorder integer keys, and stream's overlay lists quads in dict order) |
| missing_corners | array of strings |
| homography | 3x3 nested arrays (source px -> canvas px) or null |
| px_per_mm | number |
| quality, calibration | plain objects of numbers/strings |
| bubble_centers_mm | `{answers: [q][choice] -> [x,y], id: [digit][value] -> [x,y], subject: [...]}` nested arrays (mm, rounded to 2 dp as in Python); `{}` if not computed |
| timing_ms | object |
| warped, name_crop | grey image `{w,h,data:Uint8Array}` or null |
| raw | PageRaw or null |
| fused_frames | int (0 = single frame) |
| **dewarp** | **null (flat) or a function `(x_mm, y_mm) -> [dx_mm, dy_mm]`** = `omr.tracks.Field.__call__` for one point; used only by `types.imagePoints` |

Fusion adds a non-field `trace_src = [fusedRaw, [raw per frame]]` to applied results (as Python does).

## QuestionResult
`{number, state: 'single'|'blank'|'multi', marked: [int], labels: [str], confidence, ambiguous, scores: [number], guess: int|null, guessed: bool}`
## DigitResult
`{index, state, value: int|null, confidence, ambiguous, scores: [number], guessed: bool}`

## PageRaw (per-bubble measurements; what `decide.decidePage` and fusion consume)
`{numbers: [int], labels: [[str]], digit_index: [int], FA, SA, FI, SI, ink, min_strength, verified: bool, FS|null, SS|null, ink_source: 'auto'|'default', ink_separation, MA|null, MI|null, MS|null, decided_sig}`

2-D numpy arrays are **arrays of rows**: `FA FI FS MA MI MS` rows are `Float32Array` (numpy float32 - the logic emulates float32
arithmetic with `Math.fround`, so values MUST be float32-rounded and stay float32), `SA SI SS` rows are `Float64Array` (float64 in numpy).
NaN is allowed in `MA/MI/MS` (unknowable bubble). The UNRELIABLE sentinel is fill exactly 0.5 together with strength exactly 0.3.
`decided_sig` = `OMR.decide.decisionSig(answers, student_id, subject)` (a string) set by recognize() right after `decidePage(raw)`
(fusion.trustworthy compares it with the frame's current decisions).

## Decisions
`decide.decidePage(raw) -> {answers, id_digits, student_id, flags, calibration, subject_digits, subject}` (= PageDecision).
`calibration = {ink, typ_fill, t_lo, t_hi, id_typ_fill, id_t_lo, id_t_hi, ink_auto, ink_source, ink_separation}`.

## Module map / load order
`util, markers, layout, types, identity, decide, fusion, revision, stream` (J3 may use `OMR.layout.layoutForHeader`, `OMR.markers.*`,
`OMR.decide.*`, `OMR.types.*`). J3 still owns `_fill`, `_mean_dark`, `_ink_level`, `_calibrate`-independent scoring, tracks (Field).

## Stream events
`session.feed(result, t, payload) -> {state, hint, overlay, frame_index, present, possibly_same, unread, commit?|revision?}`;
`commit = {result, payload, key, identity, t, frames, frame_index, fused_frames}`; keys/identities are nested arrays (Python tuples).
`session.finish()`, `session.missed(final)`, `session.unreadEpisodes(final)` (Python `unread_episodes`).
