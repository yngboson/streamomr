// ---- development logging (only in the dev build: scripts/build_web.py --dev)
// Logs EVERYTHING, good reads included (so a doubtful sheet can be compared with a clean one): a text log, every scan (camera
// frame, rectified sheet, annotated sheet, a zoomed crop of EVERY row, the full decision trace: fill %, strength, disc darkness,
// thresholds, rule verdicts, per-frame spread), a CSV with one line per processed frame, and snapshots of frames where a
// sheet was in view but could not be read.  Downloaded as ONE zip (no compression: the images are JPEGs already).  See README.txt in it.
const DEV_FAIL = new Set(['partial', 'bad_geometry', 'bad_meta', 'inconsistent', 'unknown_form']);   // sheet in view, not readable
const DEV_MAX_FAIL = 60, DEV_FAIL_GAP_MS = 1500, DEV_MAX_LINES = 50000;
const devT0 = Date.now(), devLines = [], devScans = [], devFails = [], devFrames = [];
const DEV_MAX_FRAMES = 30000;
// per-frame cost split (frames.csv, last 11 columns): grab = drawImage+getImageData on the main thread; frame_age = how long the
// frame waited after its grab (prefetch); xfer = postMessage main->worker; py_convert = RGBA -> gray (column keeps its Pyodide-era name); py_feed = session.feed;
// py_other = worker_ms - convert - recognize - feed (clean, JSON.stringify, commit detail JPEG); post = worker->main;
// parse = JSON.parse; gap = result received -> next frame sent (worker idle time).  total_ms of the old columns = python only.
function devTimingCols(e, tm) {
  const s = devSent || {}, ms = e.ms || {}, t = (tm && tm.m && tm.m.t) || {}, r1 = v => v == null || isNaN(v) ? '' : Math.round(v * 10) / 10;
  const other = t.worker != null ? t.worker - (ms.convert || 0) - (ms.recognize || 0) - (ms.feed || 0) : null;
  return [r1(s.grabMs), r1(s.age), r1(t.xfer), r1(ms.convert), r1(ms.feed), r1(other), r1(t.worker), r1(tm && t.tx != null ? tm.recvAt - t.tx : null), r1(tm && tm.parseMs), r1(s.gap), s.prefetched ? 1 : 0];
}
const devStat = { frames: 0, recMs: 0, status: {}, lastReport: performance.now() };
let devPrevState = null, devPrevStatus = null, devPrevMissed = '', devLastFail = 0, devBuild = new Date().toISOString();
document.title += ' [DEV]';

const dpad = (n, w = 2) => String(n).padStart(w, '0');
function devLog(kind, text) {
  if (devLines.length >= DEV_MAX_LINES) return;
  const d = new Date();
  devLines.push(`${dpad(d.getHours())}:${dpad(d.getMinutes())}:${dpad(d.getSeconds())}.${dpad(d.getMilliseconds(), 3)} +${((Date.now() - devT0) / 1000).toFixed(1)}s ${kind.padEnd(7)} ${text}`);
  devRefresh();
}
// JPEG of the frame that was just sent to the worker (the one-frame-in-flight loop leaves it in `grab` until the next send)
const devFrame = () => new Promise(res => { try { flightCanvas.toBlob(b => res(b), 'image/jpeg', 0.85); } catch (err) { res(null); } });
const devBlob = (cv, q = 0.9) => new Promise(res => cv.toBlob(res, 'image/jpeg', q));
const devQ = q => Object.entries(q || {}).map(([k, v]) => `${k}=${typeof v === 'number' ? +v.toFixed(3) : v}`).join(' ');

function devOnStart() {
  const st = track && track.getSettings ? track.getSettings() : {};
  devLog('SESSION', `start camera ${video.videoWidth}x${video.videoHeight} settings=${JSON.stringify(st)}`);
  devVideoStart();
}
function devOnStop() { devVideoStop(); }

// ---- optional camera video recording (MediaRecorder on the same camera stream; full camera resolution, independent of
// the frames the recognizer processed).  Each start..stop is one segment; segment start/end are in session seconds (+t in
// log.txt, t_s in frames.csv), so a frame row can be found in the video at (t_s - start_s).
const DEV_VIDEO_MAX_BYTES = 400e6, DEV_VIDEO_BPS = 4e6;
const devVideos = [];   // {name, mime, start_s, end_s, chunks: [Blob], bytes, done: Promise}
let devRec = null;
const devVideoOn = () => { try { return localStorage.getItem('dev-video') !== '0'; } catch (err) { return true; } };
function devVideoMime() {
  if (!window.MediaRecorder) return null;
  for (const m of ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'])
    if (!MediaRecorder.isTypeSupported || MediaRecorder.isTypeSupported(m)) return m;
  return '';
}
function devVideoStart() {
  if (!devVideoOn() || !stream) return;
  const used = devVideos.reduce((a, v) => a + v.bytes, 0);
  if (used >= DEV_VIDEO_MAX_BYTES) { devLog('VIDEO', `not recording: ${Math.round(used / 1e6)} MB already stored (limit ${DEV_VIDEO_MAX_BYTES / 1e6} MB)`); return; }
  const mime = devVideoMime();
  if (mime === null) { devLog('VIDEO', 'MediaRecorder not available in this browser'); return; }
  let rec;
  try { rec = new MediaRecorder(stream, Object.assign({ videoBitsPerSecond: DEV_VIDEO_BPS }, mime ? { mimeType: mime } : {})); }
  catch (err) { devLog('VIDEO', 'cannot record: ' + (err && err.message || err)); return; }
  const type = rec.mimeType || mime || 'video/mp4', ext = /mp4/.test(type) ? 'mp4' : 'webm';
  const seg = { name: `video/segment_${dpad(devVideos.length + 1)}.${ext}`, mime: type, start_s: +((Date.now() - devT0) / 1000).toFixed(2), end_s: null, chunks: [], bytes: 0 };
  seg.done = new Promise(res => { rec.onstop = () => { seg.end_s = +((Date.now() - devT0) / 1000).toFixed(2); devLog('VIDEO', `${seg.name} stopped at +${seg.end_s}s, ${(seg.bytes / 1e6).toFixed(1)} MB`); res(); }; });
  rec.ondataavailable = ev => {
    if (!ev.data || !ev.data.size) return;
    seg.chunks.push(ev.data); seg.bytes += ev.data.size;
    if (devVideos.reduce((a, v) => a + v.bytes, 0) >= DEV_VIDEO_MAX_BYTES && rec.state === 'recording') { devLog('VIDEO', 'size limit reached - stopping'); rec.stop(); }
    devRefresh();
  };
  rec.onerror = ev => devLog('VIDEO', 'recorder error: ' + ((ev.error && ev.error.message) || ev.error || ev));
  rec.start(1000);
  devVideos.push(seg); devRec = rec;
  devLog('VIDEO', `${seg.name} recording (${type}, ~${DEV_VIDEO_BPS / 1e6} Mbit/s) from +${seg.start_s}s`);
}
async function devVideoStop() {
  const rec = devRec; devRec = null;
  if (rec && rec.state !== 'inactive') { rec.stop(); await devVideos[devVideos.length - 1].done; }
}

function devOnResult(e, tm) {
  const d = e.dev || {}, now = performance.now();
  devStat.frames++; devStat.recMs += (e.ms && e.ms.recognize) || 0;
  if (devFrames.length < DEV_MAX_FRAMES) devFrames.push([new Date().toISOString(), ((Date.now() - devT0) / 1000).toFixed(2), e.status || (e.error ? 'error' : ''), e.state, (e.ms && e.ms.recognize) ?? '', (e.ms && e.ms.total) ?? '', `${e.w || ''}x${e.h || ''}`,
    ((e.dev || {}).markers || []).join(' '), ((e.dev || {}).missing || []).join(' '), ((e.dev || {}).flags || []).join(';'), e.commit ? 'commit' : '', JSON.stringify((e.dev || {}).quality || {}),
    ...devTimingCols(e, tm)]);
  const status = e.status || (e.error ? 'error' : '?');
  devStat.status[status] = (devStat.status[status] || 0) + 1;
  if (e.state !== devPrevState) { devLog('STATE', `${devPrevState ?? '-'} -> ${e.state}  ${e.hint || ''}`); devPrevState = e.state; }
  if (status !== devPrevStatus) {
    devLog('STATUS', `${devPrevStatus ?? '-'} -> ${status}  markers=[${(d.markers || []).join(',')}] missing=[${(d.missing || []).join(',')}] flags=[${(d.flags || []).join(',')}] ${devQ(d.quality)}`);
    devPrevStatus = status;
  }
  if (e.error) devLog('ERROR', e.error);
  const missed = (e.missed || []).map(m => `p${m.page + 1}:${m.serial || m.student_id || '?'}`).join(',');
  if (missed !== devPrevMissed) { devLog('MISSED', missed || '(none)'); devPrevMissed = missed; }
  if (DEV_FAIL.has(status) && devFails.length < DEV_MAX_FAIL && now - devLastFail > DEV_FAIL_GAP_MS) {
    devLastFail = now;
    const n = devFails.length + 1, name = `failures/${dpad(n, 4)}_${status}.jpg`;
    const rec = { n, file: name, time: new Date().toISOString(), status, state: e.state, hint: e.hint || '', dev: d, size: `${video.videoWidth}x${video.videoHeight}`, blob: null };
    devFails.push(rec);
    devFrame().then(b => { rec.blob = b; });
    devLog('FAIL', `#${n} ${status} saved as ${name}  ${devQ(d.quality)}`);
  }
  if (now - devStat.lastReport > 5000) {
    const s = devStat, hist = Object.entries(s.status).map(([k, v]) => `${k}:${v}`).join(' ');
    devLog('STATS', `${(s.frames * 1000 / (now - s.lastReport)).toFixed(1)} fps, recognize avg ${(s.recMs / Math.max(s.frames, 1)).toFixed(0)} ms, ${hist}`);
    s.frames = 0; s.recMs = 0; s.status = {}; s.lastReport = now;
  }
}

// every row of a scan (questions, ID digits, subject digits): [{kind, key, name, status, severity ('' = read cleanly), ..., g, r}]
function devRows(d) {
  const out = [];
  OmrDetail.groupsOf(d).forEach(g => g.rows.forEach((r, i) => {
    const [cls, txt] = OmrDetail.rowStatus(r);
    if (!r.centers.length) return;
    const key = g.kind === 'q' ? `q${r.q}` : `${g.kind}_${i + 1}`;
    out.push({ kind: g.kind, key, name: g.name(r, i), idx: i, status: txt, severity: cls, state: r.state, guessed: !!r.guessed, labels: r.labels || null,
               marked: r.marked || r.value, scores: r.scores, g, r });
  }));
  return out;
}
const devDoubts = d => devRows(d).filter(x => x.severity);

// readable text of one traced row (see scanner.js rowTrace (glue.py::_row_trace)): every choice with its marking % and the intermediate values
function devRowText(name, t) {
  const f = t.final, marked = f.marked || (f.value !== undefined && f.value !== null ? [f.value] : []);
  const lab = i => t.choices[i] ? t.choices[i].label : i;
  const head = `${name.padEnd(14)} final=${f.state}${marked.length ? '[' + marked.map(lab).join(',') + ']' : ''}${f.guessed ? ' (추정)' : ''} conf ${f.confidence_pct}%${f.ambiguous ? ' AMBIGUOUS' : ''}` +
    `  | abs=${t.abs.state}[${t.abs.marked.map(lab).join(',')}] conf ${t.abs.confidence_pct}%${t.abs.confident ? ' confident' : ''}${t.abs.ambiguous ? ' amb' : ''}` +
    `  | rel=${t.rel.indices.length ? t.rel.indices.map(lab).join(',') + ' ex=' + t.rel.excess.join('/') : '-'} ${t.rel.trusted ? 'ON' : 'off'}`;
  const ch = t.choices.map((c, i) => `    ${marked.includes(i) ? '●' : '○'} ${String(c.label).padEnd(2)} mark% ${String(c.fill_pct).padStart(5)} (frames ${c.fill_pct_min}..${c.fill_pct_max} sd ${c.fill_pct_std})  strength ${c.strength}  disc_dark ${c.disc_dark}  excess ${c.excess_vs_others}  ratio_to_max ${c.ratio_to_max}  t_in ${c.t_in}  -> ${c.abs_class}`);
  return [head, ...ch, ...(t.why || []).map(w => '    why: ' + w)].join('\n');
}
// trace.txt of a scan: page-level numbers, then ALL rows (clean ones too, for comparison)
function devTraceText(rec) {
  const tr = (rec.dev || {}).trace;
  if (!tr || !tr.page) return 'no trace\n';
  const p = tr.page, L = [`scan #${rec.n} ${rec.folder}  id=${rec.summary.student_id} subject=${rec.summary.subject} page=${rec.summary.page + 1}`,
    `frames fused ${p.frames_fused}  ink ${p.ink} (${p.ink_source}, separation ${p.ink_separation})  min_strength ${p.min_strength}  alignment_verified ${p.alignment_verified}`,
    `answers: typ_fill ${p.typ_fill}  t_lo ${p.t_lo}  t_hi ${p.t_hi}    id/subject: typ_fill ${p.id_typ_fill}  t_lo ${p.id_t_lo}  t_hi ${p.id_t_hi}`,
    `relative rule trust: answers(page) ${p.relative_rule_page_trust}  digits ${p.relative_rule_digit_trust}   consts ${JSON.stringify(p.consts)}`,
    'mark% = share of the bubble that is inked (the fill the decision uses; median over the fused frames, min..max/sd over those frames)', ''];
  tr.answers.forEach(t => L.push(devRowText(`Q${t.q}`, t)));
  tr.id.forEach(t => L.push(devRowText(`ID digit ${t.digit}`, t)));
  tr.subject.forEach(t => L.push(devRowText(`subject digit ${t.digit}`, t)));
  return L.join('\n') + '\n';
}

function devOnCommit(s, commit) {
  const d = commit.detail || {}, rows = d.answers ? devRows(d) : [], doubts = rows.filter(x => x.severity);
  const n = devScans.length + 1, sid = s.student_id ?? 'noid';
  const rec = { n, folder: `scans/${dpad(n, 3)}_${sid}`, time: s.time, summary: s, detail: d, dev: commit.dev || {}, frames: commit.frames, rows, doubts, frame: devFrame() };   // every scan keeps the camera frame at commit time
  devScans.push(rec);
  const list = doubts.map(x => `${x.name}[${x.status}${x.guessed ? ',추정' : ''}: ${x.scores.map(v => Math.round(v * 100)).join('/')}]`).join('  ');
  devLog('COMMIT', `#${n} ${rec.folder} id=${s.student_id} subject=${s.subject} page=${s.page + 1}/${s.total_pages} frames=${commit.frames} review=${s.needs_review} flags=[${s.flags.join(',')}] doubts=${doubts.length}${list ? '  ' + list : ''}`);
  devLog('FILES', `#${n} -> ${rec.folder}/{frame.jpg,sheet.jpg,sheet_marked.jpg,crops/*.jpg,trace.txt,scan.json}`);
  const tr = (commit.dev || {}).trace;
  if (tr && tr.page) {   // one log line per row, clean ones too
    devLog('TRACE', `#${n} ink=${tr.page.ink}(${tr.page.ink_source}) t_lo=${tr.page.t_lo} t_hi=${tr.page.t_hi} id_t_hi=${tr.page.id_t_hi} fused=${tr.page.frames_fused} verified=${tr.page.alignment_verified} rel_trust=${tr.page.relative_rule_page_trust}/${tr.page.relative_rule_digit_trust}`);
    const line = (name, t) => devLog('ROW', `#${n} ${name} final=${t.final.state}${t.final.marked ? '[' + t.final.marked.map(i => t.choices[i].label).join(',') + ']' : '=' + t.final.value} conf=${t.final.confidence_pct}%${t.final.ambiguous ? ' AMB' : ''} | ` +
      t.choices.map(c => `${c.label}:${c.fill_pct}%/s${c.strength}/d${c.disc_dark}/x${c.excess_vs_others}/${c.abs_class}`).join('  ') + (t.why && t.why.length ? ' | ' + t.why.join('; ') : ''));
    tr.answers.forEach(t => line(`Q${t.q}`, t)); tr.id.forEach(t => line(`ID${t.digit}`, t)); tr.subject.forEach(t => line(`SUBJ${t.digit}`, t));
  }
}

// a revision (later frames of an already committed sheet improved / contradict it): logged and attached to that scan
function devOnRevision(s, rv) {
  const rec = devScans[devScans.length - 1];
  if (!rec) return;
  const entry = { time: new Date().toISOString(), n: rv.n, frames_fused: rv.frames, changed: rv.changed, contradiction: rv.contradiction, prev_severity: rv.prev_severity,
    severity: rv.summary.severity, student_id: rv.summary.student_id, subject: rv.summary.subject, flags: rv.summary.flags, n_doubtful: rv.summary.n_doubtful };
  (rec.revisions = rec.revisions || []).push(entry);
  const tr = (rv.dev || {}).trace;
  if (tr) rec.revTrace = tr;
  rec.summaryRevised = rv.summary;
  devLog('REVISION', `#${rec.n} ${rec.folder} r${rv.n} frames=${rv.frames} changed=[${rv.changed.join(',')}] contradiction=[${rv.contradiction.join(',')}] severity ${rv.prev_severity} -> ${rv.summary.severity} id=${rv.summary.student_id} subject=${rv.summary.subject} flags=[${rv.summary.flags.join(',')}]`);
}

// ---- zip (stored, no compression)
const devCrcT = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function devCrc32(u8) { let c = 0xFFFFFFFF; for (let i = 0; i < u8.length; i++) c = devCrcT[(c ^ u8[i]) & 0xFF] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
function devZip(files) {   // files: [{name, data: Uint8Array}] -> Blob
  const enc = new TextEncoder(), parts = [], central = [];
  const now = new Date(), dosT = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
  const dosD = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  let off = 0;
  for (const f of files) {
    const name = enc.encode(f.name), crc = devCrc32(f.data), sz = f.data.length;
    const lh = new DataView(new ArrayBuffer(30));
    lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true); lh.setUint16(8, 0, true);
    lh.setUint16(10, dosT, true); lh.setUint16(12, dosD, true); lh.setUint32(14, crc, true); lh.setUint32(18, sz, true); lh.setUint32(22, sz, true);
    lh.setUint16(26, name.length, true); lh.setUint16(28, 0, true);
    parts.push(lh.buffer, name, f.data);
    const ch = new DataView(new ArrayBuffer(46));
    ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true); ch.setUint16(8, 0x0800, true); ch.setUint16(10, 0, true);
    ch.setUint16(12, dosT, true); ch.setUint16(14, dosD, true); ch.setUint32(16, crc, true); ch.setUint32(20, sz, true); ch.setUint32(24, sz, true);
    ch.setUint16(28, name.length, true); ch.setUint32(42, off, true);
    central.push(ch.buffer, name);
    off += 30 + name.length + sz;
  }
  const csize = central.reduce((a, b) => a + (b.byteLength ?? b.length), 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true); end.setUint32(12, csize, true); end.setUint32(16, off, true);
  return new Blob([...parts, ...central, end.buffer], { type: 'application/zip' });
}
const devText = s => new TextEncoder().encode(s);
const devBytes = async blob => new Uint8Array(await blob.arrayBuffer());
const devB64 = b64 => Uint8Array.from(atob(b64), c => c.charCodeAt(0));
const devLoad = src => new Promise((ok, no) => { const im = new Image(); im.onload = () => ok(im); im.onerror = no; im.src = src; });
const devCsv = rows => '﻿' + rows.map(r => r.map(csvCell).join(',')).join('\n') + '\n';

async function devExport() {
  const btn = $('dev-zip'); btn.disabled = true; btn.textContent = '만드는 중...';
  try {
    const wasRecording = !!devRec;
    if (wasRecording) await devVideoStop();   // close the running segment so it is complete in the zip

    const files = [], idx = [['n', 'time', 'folder', 'student_id', 'subject', 'page', 'total_pages', 'needs_review', 'flags', 'n_doubtful_rows', 'n_blank', 'n_multi', 'doubt_items', 'frame', 'sheet', 'sheet_marked', 'trace', 'scan_json', 'revisions', 'revised_rows', 'contradiction_rows', 'revised_student_id', 'revised_severity']];
    for (const rec of devScans) {
      const s = rec.summary, d = rec.detail, f = rec.folder, names = [];
      const put = (name, data) => { files.push({ name: `${f}/${name}`, data }); names.push(name); };
      if (d.img) {
        put('sheet.jpg', devB64(d.img));
        {
          const img = await devLoad('data:image/jpeg;base64,' + d.img), groups = OmrDetail.groupsOf(d);
          put('sheet_marked.jpg', await devBytes(await devBlob(OmrDetail.drawSheet(img, d.ppm, groups))));
          for (const x of rec.rows) {   // zoomed crop of EVERY row (clean ones too): with rings, and the plain pixels
            put(`crops/${x.key}.jpg`, await devBytes(await devBlob(OmrDetail.drawCrop(img, d.ppm, x.g, x.r, null, true))));
            put(`crops/${x.key}_raw.jpg`, await devBytes(await devBlob(OmrDetail.drawCrop(img, d.ppm, x.g, x.r, null, false))));
          }
        }
      }
      const fb = rec.frame ? await rec.frame : null;
      if (fb) put('frame.jpg', await devBytes(fb));
      const { img, ...detailNoImg } = d;
      put('trace.txt', devText(devTraceText(rec)));
      const doubts = rec.rows.map(({ g, r, ...x }) => ({ ...x, crop: `crops/${x.key}.jpg`, crop_raw: `crops/${x.key}_raw.jpg` }));
      put('scan.json', devText(JSON.stringify({ n: rec.n, folder: f, time: rec.time, summary: s, frames_fused: rec.frames, rows: doubts, thresholds: { t_lo: d.t_lo, t_hi: d.t_hi, id_t_lo: d.id_t_lo, id_t_hi: d.id_t_hi },
        dev: rec.dev, revisions: rec.revisions || [], summary_after_revisions: rec.summaryRevised || null, trace_after_revisions: rec.revTrace || null, files: names.filter(x => x !== 'scan.json'), detail: detailNoImg }, null, 1)));
      idx.push([rec.n, rec.time, f, s.student_id, s.subject, s.page + 1, s.total_pages, s.needs_review, s.flags.join(';'), rec.doubts.length, s.n_blank, s.n_multi,
        rec.doubts.map(x => x.key + ':' + x.status).join(';'), names.includes('frame.jpg') ? `${f}/frame.jpg` : '', `${f}/sheet.jpg`, names.includes('sheet_marked.jpg') ? `${f}/sheet_marked.jpg` : '', `${f}/trace.txt`, `${f}/scan.json`,
        (rec.revisions || []).length, [...new Set((rec.revisions || []).flatMap(x => x.changed))].join(';'), [...new Set((rec.revisions || []).flatMap(x => x.contradiction))].join(';'),
        rec.summaryRevised ? rec.summaryRevised.student_id : '', rec.summaryRevised ? rec.summaryRevised.severity : '']);
    }
    const fr = [['n', 'file', 'time', 'status', 'state', 'hint', 'frame_size', 'markers', 'missing_corners', 'flags', 'quality']];
    for (const x of devFails) {
      if (x.blob) files.push({ name: x.file, data: await devBytes(x.blob) });
      const d = x.dev || {};
      fr.push([x.n, x.blob ? x.file : '', x.time, x.status, x.state, x.hint, x.size, (d.markers || []).join(' '), (d.missing || []).join(' '), (d.flags || []).join(';'), JSON.stringify(d.quality || {})]);
    }
    const videos = [];
    for (const v of devVideos) {
      if (!v.bytes) continue;
      files.push({ name: v.name, data: await devBytes(new Blob(v.chunks, { type: v.mime })) });
      videos.push({ file: v.name, mime: v.mime, start_s: v.start_s, end_s: v.end_s, mb: +(v.bytes / 1e6).toFixed(1) });
    }
    const st = track && track.getSettings ? track.getSettings() : {};
    const session = { videos, build: devBuild, exported: new Date().toISOString(), page: location.href, userAgent: navigator.userAgent, platform: navigator.platform,
      screen: `${screen.width}x${screen.height}@${window.devicePixelRatio}`, camera: { video: `${video.videoWidth}x${video.videoHeight}`, settings: st },
      engine: 'js', scans: devScans.length, doubtful_scans: devScans.filter(r => r.doubts.length).length, failure_frames: devFails.length, frame_rows: devFrames.length, log_lines: devLines.length };
    files.unshift({ name: 'README.txt', data: devText(DEV_README) }, { name: 'session.json', data: devText(JSON.stringify(session, null, 1)) },
                  { name: 'log.txt', data: devText(devLines.join('\n') + '\n') }, { name: 'scans.csv', data: devText(devCsv(idx)) }, { name: 'failures.csv', data: devText(devCsv(fr)) },
                  { name: 'frames.csv', data: devText(devCsv([['time', 't_s', 'status', 'state', 'recognize_ms', 'total_ms', 'frame', 'markers', 'missing_corners', 'flags', 'event', 'quality', 'grab_ms', 'frame_age_ms', 'xfer_ms', 'py_convert_ms', 'py_feed_ms', 'py_other_ms', 'worker_ms', 'post_ms', 'parse_ms', 'gap_ms', 'prefetched'], ...devFrames])) });
    const a = document.createElement('a'); a.href = URL.createObjectURL(devZip(files));
    a.download = 'streamomr-dev_' + new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-') + '.zip';
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 30000);
    devLog('EXPORT', `zip with ${files.length} files`);
    if (wasRecording && running) devVideoStart();   // keep recording after an export taken mid-session
  } catch (err) { devLog('ERROR', 'export failed: ' + (err && err.message || err)); alert('ZIP 생성 실패: ' + (err && err.message || err)); }
  btn.disabled = false; btn.textContent = '로그 ZIP 다운로드';
}

const DEV_README = `StreamOMR 개발용 로그 (standalone dev build) - 잘 읽힌 것까지 전부 기록

log.txt          시간순 텍스트 로그. 줄 머리: 시각 +세션경과초 종류.
                 STATE 상태 전이 / STATUS 인식 상태(ok, partial, bad_meta ...)가 바뀔 때 마커·품질 수치 /
                 FAIL 시트가 보이는데 읽지 못한 프레임 저장 / COMMIT 스캔 확정 / TRACE 그 스캔의 페이지 단위 값(잉크, 임계값) /
                 ROW 모든 문항·학번·과목 자리마다 한 줄: 최종 판정, 선지별 마킹% / 강도(s) / 원 어두움(d) / 다른 선지 대비 초과(x) / 절대규칙 분류 /
                 FILES 그 스캔의 파일 위치 / REVISION 확정 뒤 같은 시트의 다음 프레임들이 결과를 고침(바뀐 행, 모순 행, 심각도 변화 - 확정 줄 아래에 붙음) / MISSED 놓친 시트 / STATS 5초마다 fps·인식 시간·상태 분포 / ERROR
session.json     브라우저, 기기, 카메라 설정, 집계
scans.csv        스캔 목록 (n, 폴더, 수험번호, 과목, 플래그, 애매한 항목, 파일 경로) - log.txt의 #n 과 같은 번호
frames.csv       처리한 모든 프레임 한 줄씩 (상태, 인식 시간, 마커 id, 가려진 모서리, 품질 수치) 최대 ${DEV_MAX_FRAMES}줄
scans/NNN_<수험번호>/   (모든 스캔, 잘 읽힌 것도 동일하게 저장)
  trace.txt        사람이 읽는 판정 과정: 페이지 값(잉크, t_lo/t_hi, 신뢰 여부) + 모든 행의 선지별 값 (비교용)
  scan.json        revisions[]: 확정 뒤 수정 기록(바뀐 행·모순 행·융합 프레임 수), summary_after_revisions: 마지막 수정 뒤 요약 (summary는 확정 당시 값)
                 같은 내용을 기계가 읽는 형태 (dev.trace) + 요약 + 모든 행 목록 + 버블 좌표(mm)
  sheet.jpg        인식에 쓴 보정(정렬)된 시트 전체
  sheet_marked.jpg 링(초록=인식, 빨강=복수, 주황=애매/공란)과 문제 행 박스를 그린 시트
  frame.jpg        확정 순간의 카메라 프레임
  crops/<q12|id_3|subject_1>.jpg   모든 행의 확대 크롭 (링 포함) / ..._raw.jpg 는 링 없는 원본 픽셀
failures/        시트가 보이는데 읽지 못한 프레임 JPEG (최대 ${DEV_MAX_FAIL}장, ${DEV_FAIL_GAP_MS}ms 간격)
video/segment_NN.mp4 (또는 .webm)  '영상도 저장'을 켠 경우 카메라 영상 원본(시작~중지 한 구간마다 파일 하나, 약 4 Mbit/s).
                 session.json의 videos[]에 start_s/end_s(세션 경과초)가 있음: frames.csv의 t_s 행은 영상의 (t_s - start_s)초 지점.
failures.csv     각 실패 프레임의 상태, 마커 id, 가려진 모서리, 품질 수치(header_reg_mm, meta_contrast ...)

용어: 마킹%(mark%) = 버블 원 면적 중 잉크로 칠해진 비율(fill). 판정에 쓰는 값이며 합쳐진 여러 프레임의 중앙값, 괄호에 프레임별 min..max, 표준편차.
      strength = 칠한 픽셀의 진하기 / 페이지 잉크 수준. disc_dark = 원 전체의 평균 어두움(상대 규칙 입력). excess = 같은 문항 다른 선지 중앙값 대비 초과분.
      ratio_to_max = 이 선지 fill / 문항 내 최대 fill. t_in = 마킹으로 인정되는 최소 fill. abs_class: in(마킹) / out(비어 있음) / grey(애매).
      conf = 행 단위 신뢰도(선지별 확률이 아님).
`;

// ---- panel
(function devPanel() {
  const card = document.createElement('div'); card.className = 'card';
  card.innerHTML = '<h1>개발용 로그 <span class="tag warn">DEV</span></h1><div class="mut small" id="dev-stats"></div>' +
    '<div class="row" style="margin:8px 0"><button id="dev-zip">로그 ZIP 다운로드</button><button id="dev-clear" class="secondary">로그 비우기</button>' +
    '<label class="small" style="display:flex;align-items:center;gap:6px"><input type="checkbox" id="dev-video"> 영상도 저장 (다음 시작부터)</label></div>' +
    '<pre id="dev-tail" class="mut small" style="white-space:pre-wrap;word-break:break-all;max-height:160px;overflow:auto;margin:0"></pre>';
  document.querySelector('.scan').appendChild(card);
  $('dev-zip').onclick = devExport;
  $('dev-video').checked = devVideoOn();
  $('dev-video').onchange = ev => { try { localStorage.setItem('dev-video', ev.target.checked ? '1' : '0'); } catch (err) {}
    if (!ev.target.checked) devVideoStop(); else if (running && !devRec) devVideoStart(); };
  $('dev-clear').onclick = () => { if (confirm('로그와 저장된 스캔/실패 프레임을 모두 비울까요?')) { devLines.length = 0; devScans.length = 0; devFails.length = 0; devFrames.length = 0; devVideos.splice(0, devRec ? devVideos.length - 1 : devVideos.length); devLog('SESSION', 'log cleared'); } };
  devLog('SESSION', `dev build ${devBuild}  UA=${navigator.userAgent}`);
})();
function devRefresh() {
  const t = $('dev-tail'); if (!t) return;
  const vmb = devVideos.reduce((a, v) => a + v.bytes, 0) / 1e6;
  $('dev-stats').textContent = `스캔 ${devScans.length}건 (애매 ${devScans.filter(r => r.doubts.length).length}) · 실패 프레임 ${devFails.length} · 로그 ${devLines.length}줄` +
    (devVideos.length ? ` · 영상 ${devVideos.length}구간 ${vmb.toFixed(0)} MB${devRec ? ' (녹화 중)' : ''}` : '');
  t.textContent = devLines.slice(-8).join('\n'); t.scrollTop = t.scrollHeight;
}
