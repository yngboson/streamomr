const $ = id => document.getElementById(id);
const video = $('video'), ov = $('ov'), stage = $('stage'), ctx = ov.getContext('2d');
const grab = document.createElement('canvas');
const gctx = grab.getContext('2d', { willReadFrequently: true });
const COLORS = ['rgba(150,150,150,.55)', '#2ecc71', '#ff5252', '#f5a524'];   // empty, marked, multi, ambiguous
const TARGET_FPS = 8;
let stream = null, track = null, audio = null, running = false, ready = false;
let busy = false, lastOv = null, lastState = 'idle', lastSend = 0, torchOn = false, frameId = 0;
let sentDims = '-', frames = 0, fpsT = performance.now(), fps = 0, rtt = 0;
const scans = [];   // in-memory results of this page load

function beep() {
  if (!audio) return;
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = 1040; o.connect(g); g.connect(audio.destination);
  g.gain.setValueAtTime(0.2, audio.currentTime); g.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + 0.18);
  o.start(); o.stop(audio.currentTime + 0.2);
}
function flash() { stage.classList.add('flash'); setTimeout(() => stage.classList.remove('flash'), 350); }
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function subjLabel(subject) { return subject === null || subject === undefined ? '과목? 미확인' : subject === '' ? '' : '과목 ' + subject; }

// ---- results list
function addScan(s, detail) {
  scans.push(s);
  const ans = s.answers.map(a => a.marked || '-').join(' ');
  const li = document.createElement('li');
  li.style.flexDirection = 'column';
  const flags = s.flags.length ? ` <span class="tag warn">${esc(s.flags.join(','))}</span>` : '';
  li.innerHTML = `<span><b>${esc(subjLabel(s.subject))}</b> ${s.subject === '' ? '' : '&middot;'} <b>${esc(s.student_id ?? '(번호 없음)')}</b> &middot; ${s.page + 1}${s.total_pages ? '/' + s.total_pages : ''}페이지` +
    (s.needs_review ? ' <span class="badge needs_review">확인 필요</span>' : '') + flags + `</span>` +
    `<span class="mut small" style="word-break:break-all">${esc(ans)}</span>`;
  if (detail && detail.img) { li.onclick = () => showDetail(s, detail); li.title = '탭하면 인식 상세'; li.innerHTML += '<span class="mut small">탭하여 인식 상세 보기 &rsaquo;</span>'; }
  $('recent').prepend(li);
  $('count').textContent = scans.length + '건'; $('csv').disabled = false;
}
const csvCell = v => { const s = v === null || v === undefined ? '' : String(v); return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
$('csv').onclick = () => {
  const nq = Math.max(0, ...scans.flatMap(s => s.answers.map(a => a.q)));
  const head = ['time', 'subject', 'student_id', 'page', 'total_pages', 'serial', 'form_key', 'n_questions', 'n_blank', 'n_multi', 'n_doubtful', 'needs_review', 'flags'];
  for (let q = 1; q <= nq; q++) head.push('q' + q);
  const rows = [head].concat(scans.map(s => {
    const byQ = {}; s.answers.forEach(a => { byQ[a.q] = a.marked; });
    const r = [s.time, s.subject, s.student_id, s.page + 1, s.total_pages, s.serial, s.form_key, s.n_questions, s.n_blank, s.n_multi, s.n_doubtful, s.needs_review, s.flags.join(';')];
    for (let q = 1; q <= nq; q++) r.push(byQ[q] ?? '');
    return r;
  }));
  const blob = new Blob(['﻿' + rows.map(r => r.map(csvCell).join(',')).join('\n') + '\n'], { type: 'text/csv;charset=utf-8' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'streamomr_' + new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-') + '.csv';
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};
$('reset').onclick = () => { if (worker) worker.postMessage({ type: 'reset' }); };

// ---- per-scan detail: the rectified sheet cropped around each question, with the doubtful choices marked
const D_COL = { on: '#2ecc71', bad: '#ff5252', doubt: '#f5a524', off: 'rgba(120,120,120,.7)' };
const cl = n => '①②③④⑤⑥⑦⑧⑨'[n] || String(n + 1);
// per-choice verdict: on = counted as marked, bad = counted wrongly (multi) / doubtful, doubt = grey zone, off = empty
function verdicts(r, lo, hi, kind) {
  const marked = kind === 'q' ? r.marked : (r.state === 'blank' ? [] : r.scores.map((v, i) => i).filter(i => r.scores[i] === Math.max(...r.scores) && r.scores[i] >= lo));
  return r.scores.map((v, i) => {
    const isMarked = marked.includes(i);
    if (r.state === 'multi' && isMarked) return 'bad';
    if (isMarked) return r.doubt || v < hi ? 'doubt' : 'on';
    return v >= lo && v < hi && r.state !== 'blank' ? 'doubt' : (v >= lo && r.doubt ? 'doubt' : 'off');
  });
}
function rowStatus(r) {
  if (r.state === 'multi') return ['bad', '복수 마킹'];
  if (r.state === 'blank' && r.doubt) return ['doubt', '애매 (옅은 마킹?)'];
  if (r.state === 'blank') return ['doubt', '공란'];
  if (r.doubt) return ['doubt', '애매'];
  return ['', '정상'];
}
function chipsFor(r, vs, kind) {
  return '<div class="chips">' + r.scores.map((v, i) => `<span class="chip ${vs[i] === 'off' ? '' : vs[i]}">${kind === 'q' ? esc(r.labels[i] ?? cl(i)) : i} &middot; ${Math.round(v * 100)}%</span>`).join('') + '</div>';
}
function showDetail(s, d) {
  const img = new Image();
  img.onload = () => renderDetail(s, d, img);
  img.src = 'data:image/jpeg;base64,' + d.img;
}
// whole rectified sheet in one canvas: a ring on every bubble, a box + number around each doubtful row
function drawSheet(img, d, groups) {
  const cv = document.createElement('canvas'); cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  const c = cv.getContext('2d'), ppm = d.ppm; c.drawImage(img, 0, 0);
  groups.forEach(({ rows, kind, lo, hi }) => rows.forEach(r => {
    if (!r.centers.length) return;
    const [cls] = rowStatus(r), vs = verdicts(r, lo, hi, kind), rMM = bubbleR(r);
    r.centers.forEach(([x, y], i) => {
      c.lineWidth = (vs[i] === 'off' ? 3 : 10) * ppm / 6; c.strokeStyle = D_COL[vs[i]];
      c.beginPath(); c.arc(x * ppm, y * ppm, rMM * ppm * 1.25, 0, 6.2832); c.stroke();
    });
    if (!cls) return;
    const xs = r.centers.map(p => p[0]), ys = r.centers.map(p => p[1]);
    const x0 = (Math.min(...xs) - rMM - (kind === 'q' ? 6 : 1.5)) * ppm, y0 = (Math.min(...ys) - rMM - 1) * ppm;
    c.lineWidth = 8 * ppm / 6; c.strokeStyle = D_COL[cls]; c.strokeRect(x0, y0, (Math.max(...xs) + rMM + 2) * ppm - x0, (Math.max(...ys) + rMM + 1) * ppm - y0);
  }));
  return cv;
}
function bubbleR(r) {
  return Math.min(...r.centers.slice(1).map((c, i) => Math.hypot(c[0] - r.centers[i][0], c[1] - r.centers[i][1])), 8) * 0.38;
}
function renderDetail(s, d, img) {
  $('dtitle').textContent = `${subjLabel(s.subject)} ${s.student_id ?? '(번호 없음)'} · ${s.page + 1}페이지`;
  const body = $('dbody'); body.innerHTML = '';
  const q = { rows: d.answers, kind: 'q', lo: d.t_lo ?? 0.2, hi: d.t_hi ?? 0.5, name: r => `${r.q}번` };
  const idLo = d.id_t_lo ?? d.t_lo ?? 0.2, idHi = d.id_t_hi ?? d.t_hi ?? 0.5;
  const groups = [q, { rows: d.id, kind: 'id', lo: idLo, hi: idHi, name: (r, i) => `학번 ${i + 1}번째 자리` },
                  { rows: d.subject, kind: 'subject', lo: idLo, hi: idHi, name: (r, i) => `과목 코드 ${i + 1}번째 자리` }];
  if (s.flags.length) body.insertAdjacentHTML('beforeend', `<div class="small" style="margin:6px 0">플래그: <span class="tag warn">${esc(s.flags.join(', '))}</span></div>`);
  const cv = drawSheet(img, d, groups); cv.className = 'sheet'; body.appendChild(cv);
  body.insertAdjacentHTML('beforeend', '<div class="mut small">초록=인식됨 · 빨강=복수 마킹 · 주황=애매/공란 · 회색=빈 칸. 확대해서 보세요.</div>');
  const list = document.createElement('div'); let n = 0;
  groups.forEach(g => g.rows.forEach((r, i) => {
    const [cls, txt] = rowStatus(r);
    if (!cls || !r.centers.length) return;
    n++;
    const box = document.createElement('div'); box.className = 'qrow ' + cls;
    box.innerHTML = `<b>${g.name(r, i)}</b> <span class="${cls === 'bad' ? 'err' : 'mut'}">${txt}</span>` + chipsFor(r, verdicts(r, g.lo, g.hi, g.kind), g.kind);
    list.appendChild(box);
  }));
  list.insertAdjacentHTML('afterbegin', `<b>${n ? '확인할 항목 ' + n + '개' : '문제 있는 항목 없음'}</b>`);
  body.appendChild(list);
  $('detail').hidden = false;
}
$('dclose').onclick = () => { $('detail').hidden = true; };
$('detail').onclick = ev => { if (ev.target === $('detail')) $('detail').hidden = true; };

// ---- overlay (same drawing as the server-based scanner)
function draw() {
  const dpr = window.devicePixelRatio || 1, cw = stage.clientWidth, ch = stage.clientHeight;
  if (ov.width !== Math.round(cw * dpr) || ov.height !== Math.round(ch * dpr)) { ov.width = Math.round(cw * dpr); ov.height = Math.round(ch * dpr); }
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, ov.width, ov.height);
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !lastOv) return;
  const s = Math.min(cw / vw, ch / vh), ox = (cw - vw * s) / 2, oy = (ch - vh * s) / 2;
  ctx.setTransform(dpr * s, 0, 0, dpr * s, dpr * ox, dpr * oy);
  const lw = 2 / s;
  if (lastOv.sheet) { ctx.strokeStyle = lastState === 'detecting' ? '#f5a524' : '#4dabf7'; ctx.lineWidth = lw; poly(lastOv.sheet); }
  ctx.strokeStyle = '#2ecc71'; ctx.lineWidth = lw * 1.5;
  (lastOv.markers || []).forEach(poly);
  const r = lastOv.bubble_r || 4;
  (lastOv.bubbles || []).forEach(([x, y, c]) => {
    ctx.fillStyle = COLORS[c]; ctx.beginPath(); ctx.arc(x, y, c ? r * 0.8 : r * 0.25, 0, 6.2832); ctx.fill();
  });
}
function poly(p) { ctx.beginPath(); p.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath(); ctx.stroke(); }
function renderLoop() { draw(); requestAnimationFrame(renderLoop); }

// ---- camera
async function listCameras() {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  const cur = track && track.getSettings().deviceId;
  $('cam').innerHTML = devs.map((d, i) => `<option value="${esc(d.deviceId)}" ${d.deviceId === cur ? 'selected' : ''}>${esc(d.label || 'Camera ' + (i + 1))}</option>`).join('');
}
async function openCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  const dev = $('cam').value;
  const video_c = { width: { ideal: 1920 }, height: { ideal: 1080 } };
  if (dev) video_c.deviceId = { exact: dev }; else video_c.facingMode = { ideal: 'environment' };   // first start: rear camera
  stream = await navigator.mediaDevices.getUserMedia({ video: video_c, audio: false });
  track = stream.getVideoTracks()[0];
  video.srcObject = stream; await video.play();
  // Safari can resolve play() before the first frame: videoWidth/Height stay 0 until metadata and a frame arrive
  for (let i = 0; i < 50 && !video.videoWidth; i++) await new Promise(r => setTimeout(r, 100));
  await listCameras();
  const caps = track.getCapabilities ? track.getCapabilities() : {};
  $('torch').hidden = !caps.torch; torchOn = false;
}
$('torch').onclick = async () => {
  torchOn = !torchOn;
  try { await track.applyConstraints({ advanced: [{ torch: torchOn }] }); } catch (e) { torchOn = false; }
};
$('cam').onchange = () => { if (running) openCamera().catch(showErr); };
video.addEventListener('resize', () => stage.classList.toggle('portrait', video.videoHeight > video.videoWidth));

// ---- worker (Pyodide) and the one-frame-in-flight loop
let worker = null;
function startWorker() {
  const src = $('worker-src').textContent;
  worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
  worker.onmessage = ev => {
    const m = ev.data;
    if (m.type === 'progress') { $('loadmsg').textContent = m.text; }
    else if (m.type === 'ready') { ready = true; $('loading').hidden = true; $('start').disabled = false; $('hint').textContent = '시작을 누르세요'; }
    else if (m.type === 'error') { $('loadtitle').textContent = '불러오기 실패'; $('loadmsg').textContent = m.text; $('hint').textContent = '불러오기 실패: ' + m.text; }
    else if (m.type === 'result') onResult(JSON.parse(m.json), m.ms);
  };
  worker.postMessage({ type: 'init', sources: JSON.parse($('omr-src').textContent), pyodideBase: PYODIDE_BASE });
}
function onResult(e, wms) {
  rtt = performance.now() - lastSend; busy = false; frames++;
  lastOv = e.overlay; lastState = e.state;
  $('state').textContent = e.state; $('state').className = 'state ' + e.state;
  $('hint').textContent = e.error || e.hint || '';
  const mm = $('missed'), ms = e.missed || [];
  mm.hidden = !ms.length;
  mm.textContent = ms.length ? '놓친 시트 있음: ' + ms.map(m => (m.serial ? 'No. ' + String(m.serial).padStart(4, '0') : '번호 없음 ' + (m.student_id || '?')) + ` (${m.page + 1}페이지)` + (m.possibly_same_as ? ` - ${m.possibly_same_as}와 같은 시트일 수 있음` : '')).join(', ') + ' - 다시 보여 주세요' : '';
  if (e.commit) {
    const s = e.commit.summary; s.time = new Date().toISOString();
    beep(); flash(); addScan(s, e.commit.detail);
    $('res').textContent = `${subjLabel(s.subject)} · ${s.student_id ?? '(번호 없음)'}`;
    if (navigator.vibrate) navigator.vibrate(60);
  }
  const now = performance.now();
  if (now - fpsT > 1000) { fps = frames * 1000 / (now - fpsT); frames = 0; fpsT = now; }
  $('stats').textContent = `${fps.toFixed(1)} fps - 프레임당 ${rtt.toFixed(0)} ms (인식 ${e.ms ? e.ms.recognize : '?'} ms) - 카메라 ${video.videoWidth}x${video.videoHeight} - 처리 ${sentDims}`;
  pump();
}
function pump() {
  if (!running || busy) return;
  const wait = Math.max(0, 1000 / TARGET_FPS - (performance.now() - lastSend));
  busy = true;
  setTimeout(sendFrame, wait);
}
function sendFrame() {
  if (!running) { busy = false; return; }
  // Never process a frame before the real size is known (a 0x0 canvas gives Safari a blank/1x1 image).
  // Frames are used exactly as the camera delivers them: never mirrored (any selfie flip is CSS-only).
  const st = track && track.getSettings ? track.getSettings() : {};
  const w = video.videoWidth || st.width || 0, h = video.videoHeight || st.height || 0;
  if (!w || !h) { setTimeout(sendFrame, 100); return; }   // stay busy and retry
  grab.width = w; grab.height = h;
  gctx.drawImage(video, 0, 0, w, h);
  const data = gctx.getImageData(0, 0, w, h).data;      // RGBA
  sentDims = `${w}x${h}`;
  lastSend = performance.now();
  worker.postMessage({ type: 'frame', id: ++frameId, buf: data.buffer, w, h }, [data.buffer]);
}
function showErr(e) { $('hint').textContent = '카메라 오류: ' + (e && e.message || e) + (location.protocol === 'http:' && location.hostname !== 'localhost' ? ' (휴대폰은 https 필요)' : ''); }

$('start').onclick = async () => {
  if (running) { running = false; if (stream) stream.getTracks().forEach(t => t.stop()); $('start').textContent = '시작'; return; }
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)(); if (audio.state === 'suspended') await audio.resume();
    await openCamera(); running = true; busy = false; $('start').textContent = '중지';
    worker.postMessage({ type: 'reset' });
    stage.classList.toggle('portrait', video.videoHeight > video.videoWidth);
    pump();
  } catch (e) { showErr(e); }
};

if (!navigator.mediaDevices) $('hint').textContent = '카메라를 사용할 수 없습니다. https 또는 localhost 로 접속하세요.';
renderLoop();
startWorker();
