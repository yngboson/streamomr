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

/* OMR-DETAIL-SHARED BEGIN  (identical text lives in standalone/app.js; tests/test_review_api.py keeps them in sync)
 * Scan detail view shared by the server UI and the standalone page: the whole rectified sheet with a coloured ring
 * per bubble (green read / red multi / orange doubtful-or-blank), boxes around problem rows, a problem list with
 * per-choice darkness, and - on tap of a problem row or one of its choices - a zoomed crop of that row from the
 * full-resolution sheet (cropped client-side from the same image, so it is as crisp as the stored sheet).
 * d = {ppm, page_w?, t_lo, t_hi, id_t_lo, id_t_hi, answers:[{q,state,marked,labels,scores,doubt,guessed?,centers}],
 *      id:[{state,value,doubt,guessed?,scores,centers}], subject:[...]}  (centers: mm on the sheet, per choice)  */
const OmrDetail = (() => {
  const COL = { on: '#2ecc71', bad: '#ff5252', doubt: '#f5a524', off: 'rgba(120,120,120,.7)' };
  const CSS = `
.od-modal { position:fixed; inset:0; background:rgba(0,0,0,.8); z-index:50; overflow:auto; padding:10px; -webkit-overflow-scrolling:touch; }
.od-modal[hidden] { display:none; }
.od-box { background:var(--card,#1c2128); color:var(--fg,#e8eaed); border-radius:10px; padding:12px; max-width:720px; margin:0 auto; }
.od-box h2 { margin:0; font-size:17px; }
.od-head { display:flex; justify-content:space-between; align-items:center; gap:8px; }
.od-sheet { width:100%; display:block; border-radius:4px; background:#fff; margin:6px 0; cursor:pointer; }
.od-row { border:1px solid #2a313b; border-radius:8px; padding:8px; margin:8px 0; }
.od-row.bad { border-color:var(--bad,#ff5252); } .od-row.doubt { border-color:var(--warn,#f5a524); }
.od-row.flash { outline:2px solid #4dabf7; }
.od-top { display:flex; flex-wrap:wrap; gap:6px 10px; align-items:center; cursor:pointer; min-height:32px; }
.od-tag { font-size:12px; padding:0 7px; border-radius:5px; background:#4dabf7; color:#000; font-weight:700; }
.od-chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.od-chip { padding:6px 12px; min-height:36px; border-radius:8px; background:#2a313b; color:inherit; border:2px solid transparent; font:inherit; font-size:14px; cursor:pointer; }
.od-chip.on { background:#2ecc71; color:#000; } .od-chip.bad { background:#ff5252; color:#000; font-weight:700; } .od-chip.doubt { background:#f5a524; color:#000; font-weight:700; }
.od-chip.sel { border-color:#fff; box-shadow:0 0 0 2px #4dabf7; }
.od-zoom { margin-top:8px; }
.od-zoom canvas { display:block; width:100%; max-height:62vh; object-fit:contain; background:#fff; border-radius:6px; }
.od-zoom .od-tools { display:flex; gap:10px; align-items:center; margin-top:4px; font-size:13px; color:var(--mut,#9aa0a6); }
.od-zoom button { font:inherit; font-size:13px; padding:4px 10px; border-radius:6px; border:1px solid #3a4250; background:#2a313b; color:inherit; cursor:pointer; }
.od-mut { color:var(--mut,#9aa0a6); font-size:13px; }
.od-err { color:var(--bad,#ff5252); }
`;
  const cl = n => '①②③④⑤⑥⑦⑧⑨'[n] || String(n + 1);
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // per-choice verdict: on = counted as marked, bad = counted wrongly (multi), doubt = grey zone / doubtful, off = empty
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
  function bubbleR(r) {
    const gaps = r.centers.slice(1).map((c, i) => Math.hypot(c[0] - r.centers[i][0], c[1] - r.centers[i][1]));
    return Math.min(...gaps, 8) * 0.38;
  }
  const choiceName = (r, kind, i) => kind === 'q' ? String(r.labels[i] ?? cl(i)) : String(i);
  function guessText(r, kind) {
    if (!r.guessed) return '';
    const v = kind === 'q' ? (r.marked && r.marked.length ? r.marked.map(i => choiceName(r, kind, i)).join(',') : null) : (r.value ?? null);
    return `<span class="od-tag" title="판독이 불확실해 가장 그럴듯한 값으로 추정한 항목">추정${v !== null ? ' ' + esc(v) : ''}</span>`;
  }
  function chipsFor(r, vs, kind) {
    return '<div class="od-chips">' + r.scores.map((v, i) =>
      `<button type="button" class="od-chip ${vs[i] === 'off' ? '' : vs[i]}" data-c="${i}">${esc(choiceName(r, kind, i))} &middot; ${Math.round(v * 100)}%</button>`).join('') + '</div>';
  }
  function groupsOf(d) {
    const lo = d.t_lo ?? 0.2, hi = d.t_hi ?? 0.5, idLo = d.id_t_lo ?? lo, idHi = d.id_t_hi ?? hi;
    return [{ rows: d.answers || [], kind: 'q', lo, hi, name: r => `${r.q}번` },
            { rows: d.id || [], kind: 'id', lo: idLo, hi: idHi, name: (r, i) => `학번 ${i + 1}번째 자리` },
            { rows: d.subject || [], kind: 'subject', lo: idLo, hi: idHi, name: (r, i) => `과목 코드 ${i + 1}번째 자리` }];
  }
  // bounding box (mm) of a row's bubbles
  function rowBox(r, kind) {
    const rMM = bubbleR(r), xs = r.centers.map(p => p[0]), ys = r.centers.map(p => p[1]);
    return [Math.min(...xs) - rMM - (kind === 'q' ? 6 : 1.5), Math.min(...ys) - rMM - 1, Math.max(...xs) + rMM + 2, Math.max(...ys) + rMM + 1];
  }

  // whole rectified sheet in one canvas: a ring on every bubble, a box around each problem row
  function drawSheet(img, ppm, groups) {
    const cv = document.createElement('canvas'); cv.width = img.naturalWidth; cv.height = img.naturalHeight;
    const c = cv.getContext('2d'); c.drawImage(img, 0, 0);
    groups.forEach(({ rows, kind, lo, hi }) => rows.forEach(r => {
      if (!r.centers.length) return;
      const [cls] = rowStatus(r), vs = verdicts(r, lo, hi, kind), rMM = bubbleR(r);
      r.centers.forEach(([x, y], i) => {
        c.lineWidth = (vs[i] === 'off' ? 3 : 10) * ppm / 6; c.strokeStyle = COL[vs[i]];
        c.beginPath(); c.arc(x * ppm, y * ppm, rMM * ppm * 1.25, 0, 6.2832); c.stroke();
      });
      if (!cls) return;
      const [x0, y0, x1, y1] = rowBox(r, kind).map(v => v * ppm);
      c.lineWidth = 8 * ppm / 6; c.strokeStyle = COL[cls]; c.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }));
    return cv;
  }

  // zoomed crop of one row from the full-resolution sheet; `sel` = tapped choice (highlighted), rings optional
  function drawCrop(img, ppm, g, r, sel, rings) {
    const rMM = bubbleR(r), xs = r.centers.map(p => p[0]), ys = r.centers.map(p => p[1]);
    const q = g.kind === 'q';
    let X0 = Math.min(...xs) - rMM - (q ? 11 : 7), X1 = Math.max(...xs) + rMM + (q ? 4 : 7);
    let Y0 = Math.min(...ys) - rMM - 5, Y1 = Math.max(...ys) + rMM + (q ? 3 : 4);
    X0 = Math.max(0, X0 * ppm); Y0 = Math.max(0, Y0 * ppm);
    X1 = Math.min(img.naturalWidth, X1 * ppm); Y1 = Math.min(img.naturalHeight, Y1 * ppm);
    const sw = X1 - X0, sh = Y1 - Y0, k = Math.max(1, Math.min(4, 760 / sw, 900 / sh));
    const cv = document.createElement('canvas'); cv.width = Math.round(sw * k); cv.height = Math.round(sh * k);
    const c = cv.getContext('2d'); c.imageSmoothingEnabled = true; c.imageSmoothingQuality = 'high';
    c.drawImage(img, X0, Y0, sw, sh, 0, 0, cv.width, cv.height);
    const vs = verdicts(r, g.lo, g.hi, g.kind), s = ppm * k, fs = Math.max(12, 3.2 * s);
    c.font = `bold ${fs}px system-ui,sans-serif`; c.textAlign = 'center'; c.textBaseline = 'bottom'; c.lineJoin = 'round';
    r.centers.forEach(([x, y], i) => {
      const px = x * s - X0 * k, py = y * s - Y0 * k, on = i === sel;
      if (rings) {
        c.lineWidth = (on ? 0.9 : vs[i] === 'off' ? 0.25 : 0.5) * s; c.strokeStyle = on ? '#4dabf7' : COL[vs[i]];
        c.beginPath(); c.arc(px, py, rMM * s * (on ? 1.4 : 1.25), 0, 6.2832); c.stroke();
      }
      const t = choiceName(r, g.kind, i), ty = py - rMM * s * 1.45;
      c.lineWidth = fs / 4; c.strokeStyle = '#000'; c.strokeText(t, px, ty);
      c.fillStyle = on ? '#4dabf7' : '#fff'; c.fillText(t, px, ty);
    });
    return cv;
  }

  let root = null;
  function ensureModal() {
    if (root) return root;
    const st = document.createElement('style'); st.textContent = CSS; document.head.appendChild(st);
    root = document.createElement('div'); root.className = 'od-modal'; root.hidden = true;
    root.innerHTML = '<div class="od-box"><div class="od-head"><h2 class="od-title"></h2><button type="button" class="od-close secondary">닫기</button></div><div class="od-body"></div></div>';
    document.body.appendChild(root);
    const close = () => { root.hidden = true; document.body.style.overflow = ''; };
    root.querySelector('.od-close').onclick = close;
    root.onclick = ev => { if (ev.target === root) close(); };
    document.addEventListener('keydown', ev => { if (ev.key === 'Escape' && !root.hidden) close(); });
    return root;
  }

  function render(o, img) {
    const m = ensureModal(), body = m.querySelector('.od-body'), d = o.d;
    const ppm = d.page_w ? img.naturalWidth / d.page_w : d.ppm;
    m.querySelector('.od-title').textContent = o.title || '';
    body.innerHTML = '';
    const groups = groupsOf(d);
    if (o.flags && o.flags.length) body.insertAdjacentHTML('beforeend', `<div class="od-mut" style="margin:6px 0">플래그: ${esc(o.flags.join(', '))}</div>`);
    const cv = drawSheet(img, ppm, groups); cv.className = 'od-sheet'; body.appendChild(cv);
    body.insertAdjacentHTML('beforeend', '<div class="od-mut">초록=인식됨 · 빨강=복수 마킹 · 주황=애매/공란 · 회색=빈 칸. 문제 항목(박스)을 누르면 그 부분을 확대해서 보여 줍니다.</div>');
    const list = document.createElement('div'), items = [];
    groups.forEach(g => g.rows.forEach((r, i) => {
      const [cls, txt] = rowStatus(r);
      if (!cls || !r.centers.length) return;
      const vs = verdicts(r, g.lo, g.hi, g.kind), box = document.createElement('div');
      box.className = 'od-row ' + cls;
      box.innerHTML = `<div class="od-top"><b>${esc(g.name(r, i))}</b> <span class="${cls === 'bad' ? 'od-err' : 'od-mut'}">${txt}</span> ${guessText(r, g.kind)}<span class="od-mut">▾ 확대</span></div>` +
        chipsFor(r, vs, g.kind) + '<div class="od-zoom" hidden></div>';
      const zoom = box.querySelector('.od-zoom'); let rings = true, sel = null;
      const show = (s, toggle) => {
        if (toggle && !zoom.hidden && sel === s) { zoom.hidden = true; box.querySelectorAll('.od-chip').forEach(b => b.classList.remove('sel')); return; }
        items.forEach(it => { if (it.box !== box) it.zoom.hidden = true; });
        sel = s; zoom.hidden = false; zoom.innerHTML = '';
        zoom.appendChild(drawCrop(img, ppm, g, r, sel, rings));
        zoom.insertAdjacentHTML('beforeend', '<div class="od-tools"><button type="button">링 ' + (rings ? '끄기' : '켜기') + '</button><span>원본 시트의 확대 크롭 · 푸른 링 = 누른 선택지</span></div>');
        zoom.querySelector('button').onclick = () => { rings = !rings; show(sel, false); };
        box.querySelectorAll('.od-chip').forEach(b => b.classList.toggle('sel', sel !== null && +b.dataset.c === sel));
      };
      box.querySelector('.od-top').onclick = () => show(null, true);
      box.querySelectorAll('.od-chip').forEach(b => { b.onclick = () => show(+b.dataset.c, true); });
      list.appendChild(box); items.push({ box, zoom, g, r, show });
    }));
    list.insertAdjacentHTML('afterbegin', `<b>${items.length ? '확인할 항목 ' + items.length + '개' : '문제 있는 항목 없음'}</b>`);
    body.appendChild(list);
    // tapping a boxed row on the sheet opens the same zoom (nearest choice highlighted)
    cv.onclick = ev => {
      const b = cv.getBoundingClientRect(), x = (ev.clientX - b.left) / b.width * cv.width / ppm, y = (ev.clientY - b.top) / b.height * cv.height / ppm;
      let best = null, bd = 1e9;
      items.forEach(it => {
        const [x0, y0, x1, y1] = rowBox(it.r, it.g.kind);
        if (x < x0 - 2 || x > x1 + 2 || y < y0 - 2 || y > y1 + 2) return;
        let nearest = 0, nd = 1e9;
        it.r.centers.forEach(([cx, cy], i) => { const dd = Math.hypot(cx - x, cy - y); if (dd < nd) { nd = dd; nearest = i; } });
        if (nd < bd) { bd = nd; best = [it, nearest]; }
      });
      if (!best) return;
      best[0].show(best[1], false);
      best[0].box.scrollIntoView({ behavior: 'smooth', block: 'center' });
      best[0].box.classList.add('flash'); setTimeout(() => best[0].box.classList.remove('flash'), 900);
    };
    m.hidden = false; m.scrollTop = 0; document.body.style.overflow = 'hidden';
    const f = o.focusQ ? items.find(it => it.g.kind === 'q' && it.r.q === o.focusQ) : null;
    if (f) { f.show(null, false); f.box.scrollIntoView({ block: 'center' }); }
  }

  // o = {title, flags, src (image URL or data: URI of the full-resolution rectified sheet), d}
  function open(o) {
    const img = new Image();
    img.onload = () => render(o, img);
    img.onerror = () => { const m = ensureModal(); m.querySelector('.od-title').textContent = o.title || '';
      m.querySelector('.od-body').innerHTML = '<p class="od-err">정렬된 시트 이미지를 불러오지 못했습니다.</p>'; m.hidden = false; };
    img.src = o.src;
  }
  // server UI: fetch /api/scans/{id}/detail and show it with the stored full-resolution sheet
  async function openScan(scanId, title, focusQ) {
    let j;
    try {
      const r = await fetch(`/api/scans/${scanId}/detail`);
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || ('오류 ' + r.status));
      j = await r.json();
    } catch (e) {
      const m = ensureModal(); m.querySelector('.od-title').textContent = title || `스캔 #${scanId}`;
      m.querySelector('.od-body').innerHTML = `<p class="od-err">${esc(e.message || e)}</p>`; m.hidden = false; return;
    }
    const subj = j.subject_code === null || j.subject_code === undefined ? '과목? 미확인' : j.subject_code === '' ? '' : '과목 ' + j.subject_code;
    open({ title: title || `스캔 #${scanId} · ${subj} ${j.student_id ?? '(번호 없음)'} · ${(j.page ?? 0) + 1}페이지`, flags: j.flags,
           src: `/api/scans/${scanId}/image/warped`, d: j, focusQ });
  }
  // any element with data-detail="<scan id>" [data-q="<question>"] opens the detail of that scan
  document.addEventListener('click', ev => {
    const a = ev.target.closest && ev.target.closest('[data-detail]');
    if (!a) return;
    ev.preventDefault(); ev.stopPropagation();
    openScan(+a.dataset.detail, undefined, a.dataset.q ? +a.dataset.q : undefined);
  }, true);
  return { open, openScan };
})();
/* OMR-DETAIL-SHARED END */

function showDetail(s, d) {
  OmrDetail.open({ title: `${subjLabel(s.subject)} ${s.student_id ?? '(번호 없음)'} · ${s.page + 1}페이지`.trim(), flags: s.flags,
                   src: 'data:image/jpeg;base64,' + d.img, d });
}

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
