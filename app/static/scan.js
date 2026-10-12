'use strict';
const $ = id => document.getElementById(id);
const video = $('video'), ov = $('ov'), stage = $('stage'), ctx = ov.getContext('2d');
const grab = document.createElement('canvas');
const COLORS = ['rgba(150,150,150,.55)', '#2ecc71', '#ff5252', '#f5a524'];   // empty, marked, multi, ambiguous
const TARGET_FPS = 8;
let stream = null, track = null, ws = null, audio = null, running = false;
let busy = false, lastOv = null, lastState = 'idle', lastSend = 0, torchOn = false;
let sentDims = '-', frames = 0, fpsT = performance.now(), fps = 0, rtt = 0;

function beep() {
  if (!audio) return;
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = 1040; o.connect(g); g.connect(audio.destination);
  g.gain.setValueAtTime(0.2, audio.currentTime); g.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + 0.18);
  o.start(); o.stop(audio.currentTime + 0.2);
}
function flash() { stage.classList.add('flash'); setTimeout(() => stage.classList.remove('flash'), 350); }
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

// subject: digits, "" = form without subject code, null/undefined = not readable
function subjLabel(subject) { return subject === null || subject === undefined ? '과목? 미확인' : subject === '' ? '' : '과목 ' + subject; }
function addScan(kind, subject, studentId, page, prepend = true) {
  const li = document.createElement('li');
  li.innerHTML = `<span><b>${esc(subjLabel(subject))}</b> ${subject === '' ? '' : '&middot;'} <b>${esc(studentId ?? '(번호 없음)')}</b> &middot; ${page + 1}페이지</span><span class="badge ${esc(kind || 'rejected')}">${esc(kind || '?')}</span>`;
  prepend ? $('recent').prepend(li) : $('recent').append(li);
}
async function loadRecent() {
  $('recent').innerHTML = '';
  const id = $('exam').value;
  try {
    const url = id ? `/api/exams/${id}/scans?limit=15` : '/api/scans/recent?limit=15';
    (await (await fetch(url)).json()).forEach(s => addScan(s.kind, s.subject, s.student_id, s.page, false));
  } catch (e) {}
}

// ---- links to the review pages + conflict / unassigned counts for the selected exam
async function refreshCounts() {
  const id = $('exam').value, show = (el, on) => { el.hidden = !on; };
  show($('dashlink'), !!id); show($('conflink'), false); show($('unlink'), false);
  if (!id) return;
  $('dashlink').href = `/static/exam.html?exam_id=${id}`;
  try {
    const e = await (await fetch(`/api/exams/${id}`)).json();
    $('conflink').href = `/static/conflicts.html?exam_id=${id}`; $('conflink').textContent = `충돌 ${e.n_conflicts}`; show($('conflink'), e.n_conflicts > 0);
    $('unlink').href = `/static/unassigned.html?exam_id=${id}`; $('unlink').textContent = `미할당 ${e.n_unassigned}`; show($('unlink'), e.n_unassigned > 0);
  } catch (err) {}
}

// ---- overlay: map video pixels to the letterboxed element rect
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
  if (dev) video_c.deviceId = { exact: dev }; else video_c.facingMode = { ideal: 'environment' };
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
$('exam').onchange =() => { loadRecent(); refreshCounts(); if (running) connect(); };

// ---- websocket loop (one frame in flight)
function connect() {
  if (ws) { ws.onclose = null; ws.close(); }
  const qs = [];
  if ($('exam').value) qs.push(`exam_id=${$('exam').value}`);
  const dbg = new URLSearchParams(location.search).get('debug_frames');   // ?debug_frames=20 -> server keeps the last 20 frames
  if (dbg) qs.push(`debug_frames=${encodeURIComponent(dbg)}`);
  const q = qs.length ? '?' + qs.join('&') : '';
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/scan${q}`);
  ws.onopen = () => { busy = false; pump(); };
  ws.onclose = () => { $('hint').textContent = '연결 끊김 - 다시 연결 중'; setTimeout(() => running && connect(), 1500); };
  ws.onmessage = ev => {
    const e = JSON.parse(ev.data);
    rtt = performance.now() - lastSend; busy = false; frames++;
    lastOv = e.overlay; lastState = e.state;
    $('state').textContent = e.state; $('state').className = 'state ' + e.state;
    $('hint').textContent = e.error || e.hint || '';
    const mm = $('missed'), ms = e.missed || [];
    mm.hidden = !ms.length;
    mm.textContent = ms.length ? '놓친 시트 있음: ' + ms.map(m => (m.serial ? 'No. ' + String(m.serial).padStart(4, '0') : '번호 없음 ' + (m.student_id || '?')) + ` (${m.page + 1}페이지)` + (m.possibly_same_as ? ` - ${m.possibly_same_as}와 같은 시트일 수 있음` : '')).join(', ') + ' - 다시 보여 주세요' : '';
    if (e.commit) {
      const o = e.commit.outcome, s = e.commit.summary;
      beep(); flash();
      addScan(o.kind, o.subject === undefined ? s.subject : o.subject, o.student_id ?? s.student_id, o.page ?? s.page);
      $('res').textContent = `${subjLabel(o.subject === undefined ? s.subject : o.subject)} · ${o.student_id ?? s.student_id ?? '(번호 없음)'}`;
      refreshCounts();
      if (navigator.vibrate) navigator.vibrate(60);
    }
    const now = performance.now();
    if (now - fpsT > 1000) { fps = frames * 1000 / (now - fpsT); frames = 0; fpsT = now; }
    $('stats').textContent = `${fps.toFixed(1)} fps - 왕복 ${rtt.toFixed(0)} ms - 서버 ${e.ms ? e.ms.server : '?'} ms - 카메라 ${video.videoWidth}x${video.videoHeight} - 전송 ${sentDims} - 서버 수신 ${e.w ? e.w + 'x' + e.h : '?'}`;
    pump();
  };
}
function pump() {
  if (!running || busy || !ws || ws.readyState !== 1) return;
  const wait = Math.max(0, 1000 / TARGET_FPS - (performance.now() - lastSend));
  setTimeout(sendFrame, wait);
  busy = true;
}
function sendFrame() {
  if (!running || !ws || ws.readyState !== 1) { busy = false; return; }
  // Never send a frame before the real size is known (a 0x0 canvas gives Safari a blank/1x1 image).
  // Frames are sent exactly as the camera delivers them: never mirrored (any selfie flip is CSS-only).
  const st = track && track.getSettings ? track.getSettings() : {};
  const w = video.videoWidth || st.width || 0, h = video.videoHeight || st.height || 0;
  if (!w || !h) { setTimeout(sendFrame, 100); return; }   // stay busy and retry
  grab.width = w; grab.height = h;
  grab.getContext('2d').drawImage(video, 0, 0, w, h);
  sentDims = `${w}x${h}`;
  grab.toBlob(b => {
    if (!b || ws.readyState !== 1) { busy = false; return; }
    lastSend = performance.now(); ws.send(b);
  }, 'image/jpeg', 0.85);
}
function showErr(e) { $('hint').textContent = '카메라 오류: ' + (e && e.message || e) + (location.protocol === 'http:' && location.hostname !== 'localhost' ? ' (휴대폰은 https 필요)' : ''); }

$('start').onclick = async () => {
  if (running) { running = false; if (ws) { ws.onclose = null; ws.close(); } if (stream) stream.getTracks().forEach(t => t.stop()); $('start').textContent = '시작'; return; }
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)(); if (audio.state === 'suspended') await audio.resume();
    await openCamera(); running = true; $('start').textContent = '중지'; connect();
    stage.classList.toggle('portrait', video.videoHeight > video.videoWidth);
  } catch (e) { showErr(e); }
};

(async () => {
  try {
    const ex = await (await fetch('/api/exams')).json();
    ex.forEach(e => $('exam').add(new Option(`#${e.id} ${e.title}${e.subject_label ? ' (' + e.subject_label + ')' : ''}`, e.id)));
    const want = new URLSearchParams(location.search).get('exam_id'); if (want) $('exam').value = want;
    loadRecent(); refreshCounts();
  } catch (e) {}
  if (!navigator.mediaDevices) $('hint').textContent = '카메라를 사용할 수 없습니다. https 또는 localhost 로 접속하세요.';
  renderLoop();
})();
