// Answer-sheet maker window (mobile-first full-screen sheet), the JS counterpart of the Python app's "답안지 양식 만들기" page.
// Needs web/src/{util,markers,layout,sheet}.js (inlined before this file) and `$` from app.js.
(function () {
const O = globalThis.OMR, mk = O.markers, ly = O.layout, sh = O.sheet;
const root = $('mk');
if (!root) return;
const f = {   // field definitions: key, label, hint, min, max, default (null = optional, empty means automatic)
  questions: { min: 1, max: 1023, v: 45 }, choices: { min: 2, max: 10, v: 5 },
  id_digits: { min: 1, max: 12, v: 8 }, subject_digits: { min: 0, max: 3, v: 2 },
  used: { min: 1, max: 1023, v: null }, pages: { min: 1, max: 16, v: null },
  copies: { min: 1, max: 500, v: 30 }, start: { min: 1, max: mk.SERIAL_MAX, v: 1 },
};
let L = null, page = 0, timer = 0, lastPdf = null;

function stepper(key) {
  const d = f[key], el = $('mk-' + key);
  el.inputMode = 'numeric';
  el.value = d.v === null ? '' : d.v;
  const set = v => { el.value = v; changed(); };
  const cur = () => el.value === '' ? null : Math.round(Number(el.value));
  const clamp = v => Math.max(d.min, Math.min(d.max, v));
  $('mk-' + key + '-m').onclick = () => { const c = cur(); if (c === null) return; set(c <= d.min && d.v === null ? '' : clamp(c - 1)); };
  $('mk-' + key + '-p').onclick = () => { const c = cur(); set(clamp(c === null ? (key === 'used' ? Number($('mk-questions').value) || d.min : d.min) : c + 1)); };
  el.addEventListener('input', changed);
  el.addEventListener('blur', () => { const c = cur(); if (c !== null && (!isFinite(c) || c < d.min || c > d.max)) el.value = clamp(isFinite(c) ? c : d.min); changed(); });
}
function val(key) { const v = $('mk-' + key).value.trim(); return v === '' ? null : Math.round(Number(v)); }

function template() {
  const choices = val('choices') || 5;
  const style = mk.LABEL_STYLES.indexOf($('mk-labels').value);
  return ly.Template({ title: $('mk-title').value.trim(), questions: val('questions') || 45, choices, id_digits: val('id_digits') || 8,
    subject_digits: val('subject_digits') ?? 2, pages: val('pages'), used: val('used'),
    choice_labels: mk.choiceLabels(style < 0 ? 0 : style, choices), min_bubble_mm: Number($('mk-bubble').value) });
}
function serialsList() {
  if (!$('mk-numbered').checked) return null;
  const n = val('copies') || 1, s = val('start') || 1, out = [];
  for (let i = 0; i < n; i++) out.push(s + i);
  return out;
}
function changed() {
  $('mk-numopts').hidden = !$('mk-numbered').checked;
  clearTimeout(timer); timer = setTimeout(refresh, 250);
  $('mk-result').hidden = true;
}
function refresh() {
  const err = $('mk-err'), go = $('mk-go');
  try {
    const t = template();
    if (t.used && t.used > t.questions) throw new Error('실제 사용 문항이 문항 수보다 많습니다');
    L = ly.buildLayout(t);
    if ($('mk-numbered').checked) {
      const s = serialsList(), last = s[s.length - 1];
      if (last > mk.SERIAL_MAX) throw new Error(`일련번호는 ${mk.SERIAL_MAX}까지입니다 (시작 ${s[0]} + ${s.length}부 = ${last})`);
    }
  } catch (e) { L = null; err.textContent = String(e.message || e); err.hidden = false; go.disabled = true; $('mk-info').textContent = ''; $('mk-tabs').innerHTML = ''; return; }
  err.hidden = true; go.disabled = false;
  $('mk-info').innerHTML = `<b>${ly.describe(L)}</b><br><span class="mut small">${L.form_key} · 쪽당 ${L.questions_per_page.join('/')}문항</span>`;
  if (page >= L.total_pages) page = 0;
  const tabs = $('mk-tabs'); tabs.innerHTML = '';
  if (L.total_pages > 1) for (let i = 0; i < L.total_pages; i++) {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = (i + 1) + '쪽';
    b.className = 'secondary' + (i === page ? ' on' : ''); b.onclick = () => { page = i; refresh(); }; tabs.appendChild(b);
  }
  const cv = $('mk-pv'), w = Math.min(900, Math.round(($('mk-pvbox').clientWidth || 340) * Math.min(2, window.devicePixelRatio || 1)));
  const s = serialsList();
  sh.drawCanvas(cv, L, page, s ? s[0] : 0, w);
}

async function make() {
  if (!L) return;
  const go = $('mk-go'), msg = $('mk-prog');
  go.disabled = true; msg.hidden = false; msg.textContent = 'PDF 만드는 중...';
  await new Promise(r => setTimeout(r, 30));
  try {
    const serials = serialsList();
    const bytes = await sh.makePdf(L, serials, 'OMR form ' + L.form_key);
    const name = `form_${L.form_key}` + (serials ? `_no${String(serials[0]).padStart(4, '0')}-${String(serials[serials.length - 1]).padStart(4, '0')}` : '') + '.pdf';
    if (lastPdf) URL.revokeObjectURL(lastPdf.url);
    const blob = new Blob([bytes], { type: 'application/pdf' });
    lastPdf = { url: URL.createObjectURL(blob), blob, name };
    $('mk-fileinfo').textContent = `${name} · ${(bytes.length / 1024).toFixed(0)} KB · ${L.total_pages * (serials ? serials.length : 1)}쪽` + (serials ? ` · No. ${String(serials[0]).padStart(4, '0')} ~ ${String(serials[serials.length - 1]).padStart(4, '0')}` : '');
    $('mk-open').href = lastPdf.url; $('mk-save').href = lastPdf.url; $('mk-save').download = name;
    $('mk-share').hidden = !(navigator.canShare && navigator.canShare({ files: [new File([blob], name, { type: 'application/pdf' })] }));
    $('mk-result').hidden = false;
    if (serials) {   // numbers are issued: continue after them next time (per form)
      try { localStorage.setItem('streamomr.nextSerial.' + L.form_key, String(serials[serials.length - 1] + 1)); } catch (e) {}
    }
    msg.hidden = true;
    $('mk-result').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  } catch (e) { msg.textContent = 'PDF를 만들 수 없습니다: ' + (e.message || e); }
  go.disabled = !L;
}

function open() {
  root.hidden = false; document.body.style.overflow = 'hidden';
  if (L) try { const n = localStorage.getItem('streamomr.nextSerial.' + L.form_key); if (n) $('mk-start').value = n; } catch (e) {}
  refresh();
}
function close() { root.hidden = true; document.body.style.overflow = ''; }

Object.keys(f).forEach(stepper);
['mk-title', 'mk-labels', 'mk-bubble', 'mk-numbered'].forEach(id => $(id).addEventListener('input', changed));
$('mk-numbered').addEventListener('change', () => {
  if ($('mk-numbered').checked && L) try { const n = localStorage.getItem('streamomr.nextSerial.' + L.form_key); if (n) $('mk-start').value = n; } catch (e) {}
  changed();
});
$('mk-go').onclick = make;
$('mk-x').onclick = close;
$('mk-share').onclick = async () => {
  try { await navigator.share({ files: [new File([lastPdf.blob], lastPdf.name, { type: 'application/pdf' })], title: lastPdf.name }); } catch (e) {}
};
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !root.hidden) close(); });
$('make').onclick = open;
window.addEventListener('resize', () => { if (!root.hidden) { clearTimeout(timer); timer = setTimeout(refresh, 150); } });
})();
