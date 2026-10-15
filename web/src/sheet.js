// Answer-sheet maker (main thread only, not part of the recognizer): Layout -> drawing ops -> canvas preview / vector PDF.
// Mirrors omr/render_pdf.py: the Layout (web/src/layout.js) decides every position, nothing here makes geometry decisions.
// PDF: graphics are vector; text is rasterised once per distinct string at 600 dpi into 1-bit stencil masks (a PDF cannot
// use the phone's system font without embedding it, and embedding a CJK font would add megabytes).  A page of the form is
// one shared Form XObject, so N numbered copies only add a few bytes each.
(function (O) {
'use strict';
var mk = O.markers, ly = O.layout;
var MM_PT = 72 / 25.4;
var TEXT_DPI = 600, TEXT_PX_MM = TEXT_DPI / 25.4;
var FONT = '"Apple SD Gothic Neo","Noto Sans CJK KR","Noto Sans KR","Malgun Gothic","Nanum Gothic",system-ui,sans-serif';

// ---- ops of one page (optionally with a serial number); coordinates in mm, origin top-left
function pageOps(L, pageIndex, serial, withSerial) {
  var pg = L.pages[pageIndex], ops = [];
  function rect(r, fill, stroke) { ops.push({ t: 'rect', r: r, fill: fill, stroke: !!stroke }); }
  function text(t) { ops.push({ t: 'text', text: t.text, x: t.x, y: t.y, size: t.size_pt, align: t.align, gray: t.gray }); }
  pg.columns.forEach(function (col) { col.bands.forEach(function (b) { rect(b.rect, b.gray, false); }); });
  rect(pg.info_box, null, true); rect(pg.answer_box, null, true); rect(pg.name_box, null, true);
  pg.markers.forEach(function (m) {
    var bits = mk.markerBits(m.id), k = bits.length, cs = m.rect[2] / k;
    rect(m.rect, 0, false);
    for (var r = 0; r < k; r++) {
      var x = 0;
      while (x < k) {
        if (bits[r][x]) { x++; continue; }
        var x2 = x;
        while (x2 < k && !bits[r][x2]) x2++;
        rect([m.rect[0] + x * cs, m.rect[1] + r * cs, (x2 - x) * cs, cs], 1, false);
        x = x2;
      }
    }
  });
  pg.meta_cells.forEach(function (c) { if (c.bit) rect(c.rect, 0, false); });
  pg.ticks.forEach(function (t) { rect(t.rect, 0, false); });
  if (withSerial && serial) {
    var bits = mk.encodeSerial(serial), cells = ly.serialCellRects();
    for (var i = 0; i < cells.length; i++) if (bits[i]) rect(cells[i], 0, false);
    text(ly.serialLabel(serial));
  }
  [pg.id_grid, pg.subject_grid].forEach(function (g) {
    if (!g) return;
    g.columns.forEach(function (col) {
      rect(col.writein, null, true);
      col.bubbles.forEach(function (b) { ops.push({ t: 'circle', cx: b.cx, cy: b.cy, r: g.bubble_diameter / 2 }); });
    });
  });
  pg.columns.forEach(function (col) {
    col.questions.forEach(function (q) { q.bubbles.forEach(function (b) { ops.push({ t: 'circle', cx: b.cx, cy: b.cy, r: pg.bubble_diameter / 2 }); }); });
  });
  var subj = pg.subject_grid ? [pg.subject_label].concat(pg.subject_grid.row_labels) : [];
  [pg.id_label, pg.name_label].concat(pg.texts, pg.id_grid.row_labels, subj).forEach(text);
  pg.columns.forEach(function (col) { col.headers.forEach(text); col.questions.forEach(function (q) { text(q.num_text); }); });
  return ops;
}

// ---- canvas preview
function drawCanvas(canvas, L, pageIndex, serial, widthPx) {
  var s = widthPx / L.page_w;
  canvas.width = Math.round(L.page_w * s); canvas.height = Math.round(L.page_h * s);
  var c = canvas.getContext('2d');
  c.fillStyle = '#fff'; c.fillRect(0, 0, canvas.width, canvas.height);
  c.lineWidth = Math.max(1, L.stroke_mm * s);
  function gray(g) { var v = Math.round(g * 255); return 'rgb(' + v + ',' + v + ',' + v + ')'; }
  pageOps(L, pageIndex, serial, true).forEach(function (o) {
    if (o.t === 'rect') {
      var r = o.r;
      if (o.fill !== null) { c.fillStyle = gray(o.fill); c.fillRect(r[0] * s, r[1] * s, r[2] * s, r[3] * s); }
      if (o.stroke) { c.strokeStyle = '#000'; c.strokeRect(r[0] * s, r[1] * s, r[2] * s, r[3] * s); }
    } else if (o.t === 'circle') {
      c.strokeStyle = '#000'; c.beginPath(); c.arc(o.cx * s, o.cy * s, o.r * s, 0, 2 * Math.PI); c.stroke();
    } else {
      c.fillStyle = gray(o.gray); c.font = (o.size / MM_PT * s) + 'px ' + FONT;
      c.textAlign = o.align === 'c' ? 'center' : o.align === 'r' ? 'right' : 'left'; c.textBaseline = 'alphabetic';
      c.fillText(o.text, o.x * s, o.y * s);
    }
  });
}

// ---- text -> 1-bit stencil (PDF ImageMask: sample 0 paints)
function textStencil(text, sizePt) {
  var px = sizePt / MM_PT * TEXT_PX_MM, cv = document.createElement('canvas'), c = cv.getContext('2d');
  c.font = px + 'px ' + FONT;
  var w = Math.ceil(c.measureText(text).width) + 4, asc = Math.ceil(px * 1.05) + 2, desc = Math.ceil(px * 0.4) + 2;
  cv.width = w; cv.height = asc + desc;
  c = cv.getContext('2d');
  c.fillStyle = '#fff'; c.fillRect(0, 0, w, cv.height);
  c.fillStyle = '#000'; c.font = px + 'px ' + FONT; c.textBaseline = 'alphabetic'; c.textAlign = 'left';
  c.fillText(text, 2, asc);
  var d = c.getImageData(0, 0, w, cv.height).data, rb = (w + 7) >> 3, out = new Uint8Array(rb * cv.height);
  out.fill(0xFF);
  for (var y = 0; y < cv.height; y++) for (var x = 0; x < w; x++)
    if (d[(y * w + x) * 4] < 140) out[y * rb + (x >> 3)] &= ~(0x80 >> (x & 7));      // dark pixel -> bit 0 -> painted
  return { w: w, h: cv.height, data: out, ox: 2 / TEXT_PX_MM, asc: asc / TEXT_PX_MM, width: (w - 4) / TEXT_PX_MM, hmm: cv.height / TEXT_PX_MM };
}

// ---- PDF writer
var enc = new TextEncoder();
function num(v) { var s = (+v).toFixed(3); return s.indexOf('.') < 0 ? s : s.replace(/0+$/, '').replace(/\.$/, ''); }
function concat(parts) {
  var n = 0, i; for (i = 0; i < parts.length; i++) n += parts[i].length;
  var out = new Uint8Array(n), o = 0; for (i = 0; i < parts.length; i++) { out.set(parts[i], o); o += parts[i].length; }
  return out;
}
async function deflate(bytes) {
  if (typeof CompressionStream === 'undefined') return null;
  try {
    var cs = new CompressionStream('deflate'), w = cs.writable.getWriter();
    w.write(bytes); w.close();
    return new Uint8Array(await new Response(cs.readable).arrayBuffer());
  } catch (e) { return null; }
}

// ops -> PDF content stream text; `img(text,size)` registers a stencil and returns its resource name
function contentStream(L, ops, img) {
  var H = L.page_h, out = [], K = 0.5522847498;
  function X(x) { return num(x * MM_PT); }
  function Y(y) { return num((H - y) * MM_PT); }
  out.push(num(L.stroke_mm * MM_PT) + ' w 0 G 0 g');
  var curFill = 0;
  function fillG(g) { if (g !== curFill) { out.push(num(g) + ' g'); curFill = g; } }
  ops.forEach(function (o) {
    if (o.t === 'rect') {
      var r = o.r, p = X(r[0]) + ' ' + Y(r[1] + r[3]) + ' ' + num(r[2] * MM_PT) + ' ' + num(r[3] * MM_PT) + ' re';
      if (o.fill !== null) { fillG(o.fill); out.push(p + ' f'); } else if (o.stroke) out.push(p + ' S');
    } else if (o.t === 'circle') {
      var cx = o.cx * MM_PT, cy = (H - o.cy) * MM_PT, r2 = o.r * MM_PT, k = K * r2;
      out.push(num(cx + r2) + ' ' + num(cy) + ' m ' +
        num(cx + r2) + ' ' + num(cy + k) + ' ' + num(cx + k) + ' ' + num(cy + r2) + ' ' + num(cx) + ' ' + num(cy + r2) + ' c ' +
        num(cx - k) + ' ' + num(cy + r2) + ' ' + num(cx - r2) + ' ' + num(cy + k) + ' ' + num(cx - r2) + ' ' + num(cy) + ' c ' +
        num(cx - r2) + ' ' + num(cy - k) + ' ' + num(cx - k) + ' ' + num(cy - r2) + ' ' + num(cx) + ' ' + num(cy - r2) + ' c ' +
        num(cx + k) + ' ' + num(cy - r2) + ' ' + num(cx + r2) + ' ' + num(cy - k) + ' ' + num(cx + r2) + ' ' + num(cy) + ' c S');
    } else {
      var st = img(o.text, o.size), left = o.align === 'c' ? o.x - st.width / 2 : o.align === 'r' ? o.x - st.width : o.x;
      fillG(o.gray);
      out.push('q ' + num(st.w / TEXT_PX_MM * MM_PT) + ' 0 0 ' + num(st.hmm * MM_PT) + ' ' + X(left - st.ox) + ' ' + Y(o.y + (st.hmm - st.asc)) + ' cm /' + st.name + ' Do Q');
    }
  });
  return enc.encode(out.join('\n'));
}

// opts: {serials: [..] | null}; returns Uint8Array
async function makePdf(L, serials, title) {
  var objs = [], stencils = {}, nStencil = 0;
  function add(dict, data, filter) { objs.push({ dict: dict, data: data || null, filter: filter || null }); return objs.length; }     // object number = index+1
  function img(text, size) {
    var key = size + '|' + text, st = stencils[key];
    if (!st) {
      st = stencils[key] = textStencil(text, size); st.name = 'T' + (++nStencil);
      st.obj = add('/Type /XObject /Subtype /Image /Width ' + st.w + ' /Height ' + st.h + ' /ImageMask true /BitsPerComponent 1 /Decode [0 1]', st.data, true);
    }
    return st;
  }
  var sets = serials && serials.length ? serials : [0];
  // reserve catalog + pages objects first (numbers 1, 2)
  add('', null); add('', null);
  // shared page bodies (everything but the serial) as Form XObjects
  var forms = L.pages.map(function (pg, i) {
    var body = contentStream(L, pageOps(L, i, 0, false), img);
    return { body: body, obj: 0, name: 'F' + i };
  });
  forms.forEach(function (f) { f.obj = add('', f.body, true); });
  var pageW = L.page_w * MM_PT, pageH = L.page_h * MM_PT;
  function xobjDict() {
    var s = '/XObject <<'; forms.forEach(function (f) { s += ' /' + f.name + ' ' + f.obj + ' 0 R'; });
    Object.keys(stencils).forEach(function (k) { s += ' /' + stencils[k].name + ' ' + stencils[k].obj + ' 0 R'; });
    return s + ' >>';
  }
  // pages: content = form Do + serial strip & label
  var pageContents = [];
  sets.forEach(function (serial) { L.pages.forEach(function (pg, i) { pageContents.push({ i: i, serial: serial }); }); });
  var serialOpsCache = {};
  function serialOnly(serial) {
    if (serialOpsCache[serial]) return serialOpsCache[serial];
    var o = [], bits = mk.encodeSerial(serial), cells = ly.serialCellRects();
    for (var k = 0; k < cells.length; k++) if (bits[k]) o.push({ t: 'rect', r: cells[k], fill: 0, stroke: false });
    var t = ly.serialLabel(serial);
    o.push({ t: 'text', text: t.text, x: t.x, y: t.y, size: t.size_pt, align: t.align, gray: t.gray });
    return (serialOpsCache[serial] = o);
  }
  var pageStreams = pageContents.map(function (pc) {
    var extra = pc.serial ? contentStream(L, serialOnly(pc.serial), img) : new Uint8Array(0);
    return concat([enc.encode('/F' + pc.i + ' Do\n'), extra]);
  });
  var pageObjs = pageStreams.map(function (s) { return add('', s, true); });
  var pageDictObjs = pageObjs.map(function (co) { return add('', null); });
  // fill in the placeholders now that all XObjects (incl. stencils created by the serial labels) are known
  var res = xobjDict();
  pageDictObjs.forEach(function (n, k) {
    objs[n - 1].dict = '/Type /Page /Parent 2 0 R /MediaBox [0 0 ' + num(pageW) + ' ' + num(pageH) + '] /Resources << ' + res + ' >> /Contents ' + pageObjs[k] + ' 0 R';
  });
  forms.forEach(function (f) {
    objs[f.obj - 1].dict = '/Type /XObject /Subtype /Form /BBox [0 0 ' + num(pageW) + ' ' + num(pageH) + '] /Resources << ' + res + ' >>';
  });
  objs[0].dict = '/Type /Catalog /Pages 2 0 R';
  objs[1].dict = '/Type /Pages /Count ' + pageDictObjs.length + ' /Kids [' + pageDictObjs.map(function (n) { return n + ' 0 R'; }).join(' ') + ']';
  var info = add('/Title (' + String(title || 'OMR form').replace(/[^\x20-\x7e]/g, '').replace(/[()\\]/g, '') + ') /Producer (StreamOMR)', null);
  // serialise
  var parts = [enc.encode('%PDF-1.5\n%\xe2\xe3\xcf\xd3\n')], offsets = [], pos = parts[0].length;
  for (var i = 0; i < objs.length; i++) {
    var ob = objs[i], head, data = ob.data;
    if (data && ob.filter) {
      var z = await deflate(data);
      if (z && z.length < data.length) { data = z; head = ob.dict + ' /Filter /FlateDecode'; } else head = ob.dict;
    } else head = ob.dict;
    var chunk = data ? concat([enc.encode((i + 1) + ' 0 obj\n<< ' + head + ' /Length ' + data.length + ' >>\nstream\n'), data, enc.encode('\nendstream\nendobj\n')])
                     : enc.encode((i + 1) + ' 0 obj\n<< ' + head + ' >>\nendobj\n');
    offsets.push(pos); parts.push(chunk); pos += chunk.length;
  }
  var xr = 'xref\n0 ' + (objs.length + 1) + '\n0000000000 65535 f \n';
  offsets.forEach(function (o) { xr += ('0000000000' + o).slice(-10) + ' 00000 n \n'; });
  xr += 'trailer\n<< /Size ' + (objs.length + 1) + ' /Root 1 0 R /Info ' + info + ' 0 R >>\nstartxref\n' + pos + '\n%%EOF\n';
  parts.push(enc.encode(xr));
  return concat(parts);
}

O.sheet = { pageOps: pageOps, drawCanvas: drawCanvas, makePdf: makePdf };
})(globalThis.OMR = globalThis.OMR || {});
