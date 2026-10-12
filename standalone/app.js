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
function addScan(s) {
  scans.push(s);
  const ans = s.answers.map(a => a.marked || '-').join(' ');
  const li = document.createElement('li');
  li.style.flexDirection = 'column';
  const flags = s.flags.length ? ` <span class="tag warn">${esc(s.flags.join(','))}</span>` : '';
  li.innerHTML = `<span><b>${esc(subjLabel(s.subject))}</b> ${s.subject === '' ? '' : '&middot;'} <b>${esc(s.student_id ?? '(번호 없음)')}</b> &middot; ${s.page + 1}${s.total_pages ? '/' + s.total_pages : ''}페이지` +
    (s.needs_review ? ' <span class="badge needs_review">확인 필요</span>' : '') + flags + `</span>` +
    `<span class="mut small" style="word-break:break-all">${esc(ans)}</span>`;
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
    beep(); flash(); addScan(s);
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
