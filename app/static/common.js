'use strict';
/* Shared helpers for the review pages (not used by the scanner page). */
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const qs = new URLSearchParams(location.search);
const EXAM_ID = (() => {
  let v = qs.get('exam_id');
  try { if (v) localStorage.setItem('omr_exam', v); else v = localStorage.getItem('omr_exam'); } catch (e) {}
  return v ? Number(v) : null;
})();
const STATUS_KO = {complete:'완료', incomplete:'불완전', conflict:'충돌', needs_review:'확인 필요'};
const KIND_KO = {new:'신규', duplicate:'중복(동일)', conflict:'충돌', unassigned:'미할당', rejected:'거부', unchanged:'변경 없음', moved:'옮김'};
const SOURCE_KO = {stream:'카메라', upload:'업로드'};

function errText(body) {
  try {
    const j = typeof body === 'string' ? JSON.parse(body) : body;
    const d = j.detail;
    if (Array.isArray(d)) return d.map(x => ((x.loc || []).slice(1).join('.') + ': ' + x.msg)).join(' / ');
    if (d) return String(d);
  } catch (e) {}
  return String(body);
}
async function api(url, opts) {
  let r;
  try {
    r = await fetch(url, opts);
  } catch (e) { throw new Error('서버에 연결할 수 없습니다'); }
  const text = await r.text();
  if (!r.ok) throw new Error(errText(text) || ('오류 ' + r.status));
  try { return JSON.parse(text); } catch (e) { return text; }
}
const post = (url, body) => api(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body ?? {})});
const put = (url, body) => api(url, {method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body ?? {})});
const fmtTime = iso => { try { return new Date(iso).toLocaleString('ko-KR'); } catch (e) { return iso || ''; } };
const badge = (cls, text) => `<span class="badge ${esc(cls)}">${esc(text)}</span>`;
const statusBadge = s => badge(s, STATUS_KO[s] || s);
const num = v => (v === null || v === undefined) ? '' : (Number.isInteger(v) ? String(v) : String(Math.round(v * 100) / 100));
/* "B" / "B,C" / "-" for a list of 0-based choice indices */
function labelsOf(marked, labels) {
  if (!marked) return '-';
  if (!marked.length) return '무응답';
  return marked.map(i => labels[i] ?? String(i + 1)).join(',');
}
function img(scanId, kind, w, extra) {
  return `/api/scans/${scanId}/image/${kind}?` + (w ? 'w=' + w : '') + (extra ? '&' + extra : '');
}

async function renderNav(current, exam) {
  const el = document.querySelector('header.nav');
  if (!el) return;
  const e = EXAM_ID ? `exam_id=${EXAM_ID}` : '';
  const link = (href, text, key, extra = '') =>
    `<a href="${href}" class="${current === key ? 'cur' : ''}">${text}${extra}</a>`;
  let info = exam;
  if (!info && EXAM_ID) { try { info = await api('/api/exams/' + EXAM_ID); } catch (err) {} }
  let h = `<span class="brand">StreamOMR</span>` + link('/static/index.html', '양식·과목', 'home') +
          link('/static/scan.html' + (e ? '?' + e : ''), '스캔', 'scan');
  if (EXAM_ID && info) {
    h += link(`/static/exam.html?${e}`, '대시보드', 'exam') +
         link(`/static/conflicts.html?${e}`, '충돌', 'conflicts', info.n_conflicts ? `<span class="cnt">${info.n_conflicts}</span>` : '') +
         link(`/static/unassigned.html?${e}`, info.subject === null ? '과목 지정' : '미할당', 'unassigned', info.n_unassigned ? `<span class="cnt warn">${info.n_unassigned}</span>` : '') +
         link(`/static/key.html?${e}`, '정답키', 'key') +
         `<span class="mut small">${esc(info.title)}${info.subject_label ? ' (' + esc(info.subject_label) + ')' : ''}</span>`;
  }
  el.innerHTML = h;
}
function needExam() {
  if (EXAM_ID) return true;
  document.querySelector('main').innerHTML = '<div class="card">과목이 선택되지 않았습니다. <a href="/static/index.html">과목 목록</a>에서 선택하세요.</div>';
  return false;
}
function showMsg(id, text, ok) {
  const el = $(id); if (!el) return;
  el.textContent = text || ''; el.className = ok ? 'okmsg' : 'err';
}
