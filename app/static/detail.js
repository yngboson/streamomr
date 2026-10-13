/* OMR-DETAIL-SHARED BEGIN  (identical text lives in standalone/app.js; tests/test_review_api.py keeps them in sync)
 * Scan detail view shared by the server UI and the standalone page: the whole rectified sheet with a coloured ring
 * per bubble (green read / red multi / orange doubtful-or-blank), boxes around problem rows, a problem list with
 * per-choice darkness, and - on tap of a problem row or one of its choices - a zoomed crop of that row from the
 * full-resolution sheet (cropped client-side from the same image, so it is as crisp as the stored sheet).
 * With o.onFix(fix) given, every problem row also gets 정정 buttons: the person picks the right choice / digit and
 * onFix({kind:'q'|'id'|'subject', index, q, choice}) (choice null = blank) applies it; the row turns green (수정됨).
 * d = {ppm, page_w?, t_lo, t_hi, id_t_lo, id_t_hi, answers:[{q,state,marked,labels,scores,doubt,guessed?,centers}],
 *      id:[{state,value,doubt,guessed?,scores,centers}], subject:[...]}  (centers: mm on the sheet, per choice)  */
const OmrDetail = (() => {
  const COL = { on: '#2ecc71', bad: '#ff5252', doubt: '#f5a524', off: 'rgba(120,120,120,.7)', fixed: '#2ecc71' };
  const CSS = `
.od-modal { position:fixed; inset:0; background:rgba(0,0,0,.8); z-index:50; overflow:auto; padding:10px; -webkit-overflow-scrolling:touch; }
.od-modal[hidden] { display:none; }
.od-box { background:var(--card,#1c2128); color:var(--fg,#e8eaed); border-radius:10px; padding:12px; max-width:720px; margin:0 auto; }
.od-box h2 { margin:0; font-size:17px; }
.od-head { display:flex; justify-content:space-between; align-items:center; gap:8px; }
.od-sheet { width:100%; display:block; border-radius:4px; background:#fff; margin:6px 0; cursor:pointer; }
.od-row { border:1px solid #2a313b; border-radius:8px; padding:8px; margin:8px 0; }
.od-row.bad { border-color:var(--bad,#ff5252); } .od-row.doubt { border-color:var(--warn,#f5a524); }
.od-row.fixed { border-color:#2ecc71; }
.od-fix { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-top:8px; padding-top:8px; border-top:1px dashed #2a313b; }
.od-fix b { font-size:13px; margin-right:2px; }
.od-fix button { min-width:44px; min-height:40px; font:inherit; font-size:15px; font-weight:700; border-radius:8px; border:1px solid #3a4250; background:#11151a; color:inherit; cursor:pointer; }
.od-fix button.cur { background:#2ecc71; color:#000; }
.od-fix button:disabled { opacity:.5; }
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
    if (r.fixed) return r.scores.map((v, i) => (kind === 'q' ? r.marked.includes(i) : r.value === i) ? 'on' : 'off');
    return r.scores.map((v, i) => {
      const isMarked = marked.includes(i);
      if (r.state === 'multi' && isMarked) return 'bad';
      if (isMarked) return r.doubt || v < hi ? 'doubt' : 'on';
      return v >= lo && v < hi && r.state !== 'blank' ? 'doubt' : (v >= lo && r.doubt ? 'doubt' : 'off');
    });
  }
  function rowStatus(r) {
    if (r.fixed) return ['fixed', '수정됨'];
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
  // 정정 buttons: every choice (answers) or digit 0-9 (ID / subject), plus 공란
  function fixBar(r, kind) {
    const cur = kind === 'q' ? (r.state === 'single' ? r.marked[0] : null) : (r.state !== 'blank' ? r.value : null);
    const opts = r.scores.map((v, i) => `<button type="button" data-v="${i}" class="${r.fixed && cur === i ? 'cur' : ''}">${esc(choiceName(r, kind, i))}</button>`).join('');
    return `<div class="od-fix"><b>${r.fixed ? '다시 정정' : '정정'}</b>${opts}<button type="button" data-v="" class="${r.fixed && cur === null ? 'cur' : ''}">공란</button></div>`;
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
        chipsFor(r, vs, g.kind) + '<div class="od-zoom" hidden></div>' + (o.onFix ? fixBar(r, g.kind) : '');
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
      box.querySelectorAll('.od-fix button').forEach(b => { b.onclick = async () => {
        const choice = b.dataset.v === '' ? null : +b.dataset.v;
        box.querySelectorAll('.od-fix button').forEach(x => { x.disabled = true; });
        let err = null;
        try { const res = await o.onFix({ kind: g.kind, index: i, q: r.q, choice }); if (res && res.error) err = res.error; }
        catch (e) { err = e.message || String(e); }
        if (err) { box.querySelectorAll('.od-fix button').forEach(x => { x.disabled = false; });
          box.querySelector('.od-fix').insertAdjacentHTML('beforeend', `<span class="od-err">${esc(err)}</span>`); return; }
        if (g.kind === 'q') Object.assign(r, { state: choice === null ? 'blank' : 'single', marked: choice === null ? [] : [choice] });
        else Object.assign(r, { state: choice === null ? 'blank' : 'single', value: choice });
        Object.assign(r, { doubt: false, guessed: false, fixed: true });
        render(Object.assign(o, { focus: [g.kind, i] }), img);
      }; });
      list.appendChild(box); items.push({ box, zoom, g, r, show, i });
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
    const ff = o.focus ? items.find(it => it.g.kind === o.focus[0] && it.i === o.focus[1]) : null;
    o.focus = null;
    if (ff) { ff.box.scrollIntoView({ block: 'center' }); ff.box.classList.add('flash'); setTimeout(() => ff.box.classList.remove('flash'), 900); return; }
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
           src: `/api/scans/${scanId}/image/warped`, d: j, focusQ, onFix: fix => serverFix(scanId, j, fix) });
  }
  // server corrections: an answer -> override on the submission; an ID / subject digit -> re-assign / re-route the scan
  const post = async (url, body) => {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) return { error: (await r.json().catch(() => ({}))).detail || ('오류 ' + r.status) };
    return r.json().catch(() => ({}));
  };
  function digitsWith(rows, index, choice) {
    const ds = rows.map((r, i) => i === index ? choice : (r.state === 'blank' ? null : r.value));
    return ds.some(v => v === null || v === undefined) ? null : ds.join('');
  }
  async function serverFix(scanId, j, fix) {
    if (fix.kind === 'q') {
      const sub = j.submission_id;
      if (!sub) return { error: '이 스캔은 아직 학생에 배정되지 않았습니다. 학번을 먼저 정정하세요.' };
      const out = await post(`/api/submissions/${sub}/override`, { question: fix.q, marked: fix.choice === null ? [] : [fix.choice], note: '스캔 화면에서 정정' });
      document.dispatchEvent(new CustomEvent('omr-fixed', { detail: { scanId, fix } }));
      return out;
    }
    const rows = fix.kind === 'id' ? j.id : j.subject, v = digitsWith(rows, fix.index, fix.choice);
    if (v === null) {   // other digits still unknown: just remember this one until the number is complete
      rows[fix.index] = Object.assign(rows[fix.index], { state: fix.choice === null ? 'blank' : 'single', value: fix.choice });
      return {};
    }
    const out = fix.kind === 'id' ? await post(`/api/scans/${scanId}/assign`, { student_id: v })
                                  : await post(`/api/scans/${scanId}/route`, { subject: v });
    if (!out.error && out.submission_id) j.submission_id = out.submission_id;
    if (!out.error) document.dispatchEvent(new CustomEvent('omr-fixed', { detail: { scanId, fix, value: v, outcome: out } }));
    return out;
  }
  // any element with data-detail="<scan id>" [data-q="<question>"] opens the detail of that scan
  document.addEventListener('click', ev => {
    const a = ev.target.closest && ev.target.closest('[data-detail]');
    if (!a) return;
    ev.preventDefault(); ev.stopPropagation();
    openScan(+a.dataset.detail, undefined, a.dataset.q ? +a.dataset.q : undefined);
  }, true);
  return { open, openScan, drawSheet, drawCrop, groupsOf, rowStatus, verdicts };
})();
/* OMR-DETAIL-SHARED END */
