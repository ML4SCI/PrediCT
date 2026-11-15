/* PrediCT Studio — one-page result report.
 *
 * Turns the result that is already open in the viewer into a one-page A4
 * report (score, slice-by-slice table, conclusions, method) and opens the
 * browser's print dialog, where "Save as PDF" makes the document.
 *
 * Why it is separate: app.js has already loaded run.json, slices.json and both
 * CSVs into `state`, so the report needs nothing new from the server. It
 * only reads that state; it is loaded after app.js and app.js does not know it
 * exists.
 *
 * Does NOT: write any file, call any new server route, recompute a score, or
 * change `state`. Every number is copied from the result files; the only
 * arithmetic is summing per-slice values to CHECK them against run.json, and
 * a disagreement is printed on the report instead of being hidden.
 *
 * Usage: the two icons in the Argument title bar (#report-btn, #report-csv-btn) call exportReport() / exportCsv().
 *        They are not in the header, which is already full at 100% browser zoom.
 *        buildReportModel() is pure (no DOM), so tests/report_check.mjs runs it in Node.
 */
(function () {
  'use strict';

  const PAGE_HEIGHT_MM = 277;       // A4 297 mm minus the 10 mm @page margin top and bottom
  const MM_TO_PX = 96 / 25.4;       // CSS reference pixel
  const TWO_COLUMN_FROM = 18;       // more slice rows than this: table is split in two side by side
  const TOL_SLICE = 0.01;           // lesions.csv vs slices.json, per slice (CSV is written with full precision)
  const TOL_TOTAL = 0.05;           // sum of slices vs run.json total
  const KEY_IMAGES = 2;             // thumbnails on the page; a third would not leave room for 16+ table rows

  // Print-safe tier colours. The tier NAMES come from app.js (TIER_SCHEMES).
  const TIER_HEX = {
    ZERO: '#6b7280', MILD: '#2e7d32', MODERATE: '#b7791f',
    'MOD-HIGH': '#c2570c', SEVERE: '#c0392b', EXTENSIVE: '#7b1e1e',
  };

  const fmt = (v, d = 1) => (v == null || v === '' || Number.isNaN(Number(v))) ? '—' : Number(v).toFixed(d);
  const esc = s => String(s ?? '—').replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const sum = (xs, f) => xs.reduce((a, x) => a + f(x), 0);
  const max = xs => (xs.length ? Math.max(...xs) : null);

  // ── data → plain report model (no DOM) ──────────────────────────────────
  /* in: { run, slices, lesions, lesions3d, tierSchemes, params, study, user, now }
   *   slices    slices.json rows; lesions  lesions.csv rows; lesions3d  lesions_3d.csv rows
   *   tierSchemes  app.js TIER_SCHEMES ({4: [[maxInclusive, name]…], 6: …})
   *   params    app.js paramGroups() — the method block, already read from run.json
   * Throws when run.json has no total: a report with a guessed total is worse than none. */
  function buildReportModel(inp) {
    const { run, slices, lesions, lesions3d, tierSchemes } = inp;
    if (!run || typeof run.agatston_total !== 'number') throw new Error('run.json has no agatston_total.');
    const total = run.agatston_total;
    const tierIn = (t, n) => tierSchemes[n].find(([m]) => t <= m)[1];
    const warnings = [];

    const cat4 = run.risk_category ? String(run.risk_category).toUpperCase() : null;
    if (cat4 && cat4 !== tierIn(total, 4)) warnings.push(`run.json says ${cat4} but the 4-tier reading of ${fmt(total)} is ${tierIn(total, 4)}.`);

    const byslice = new Map();
    lesions.forEach(l => byslice.set(l.slice_idx, [...(byslice.get(l.slice_idx) || []), l]));

    const rows = slices.filter(s => s.has_calcium).sort((a, b) => a.idx - b.idx).map(s => {
      const on = byslice.get(s.idx) || [], inc = on.filter(l => l.included);
      const fromCsv = sum(inc, l => l.agatston);
      if (Math.abs(fromCsv - s.slice_score) > TOL_SLICE)
        warnings.push(`Slice ${s.idx}: lesions.csv sums to ${fmt(fromCsv, 2)}, slices.json says ${fmt(s.slice_score, 2)}.`);
      return {
        idx: s.idx, z_mm: s.z_mm, n_included: inc.length, n_all: on.length,
        area_mm2: sum(inc, l => l.area_mm2), peak_hu: max(inc.map(l => l.peak_hu)),
        weight: max(inc.map(l => l.density_weight)), agatston: s.slice_score,
      };
    });

    const totals = {
      n_included: sum(rows, r => r.n_included), n_all: sum(rows, r => r.n_all),
      area_mm2: sum(rows, r => r.area_mm2), peak_hu: max(rows.map(r => r.peak_hu).filter(v => v != null)),
      weight: max(rows.map(r => r.weight).filter(v => v != null)), agatston: sum(rows, r => r.agatston),
    };
    if (Math.abs(totals.agatston - total) > TOL_TOTAL)
      warnings.push(`Slice scores sum to ${fmt(totals.agatston, 2)} but run.json total is ${fmt(total, 2)}.`);
    if (run.n_lesions != null && run.n_lesions !== totals.n_included)
      warnings.push(`run.json counts ${run.n_lesions} scored components, the slice table has ${totals.n_included}.`);

    const ex = lesions.filter(l => !l.included);
    const withheld = {
      n: ex.length, area_mm2: sum(ex, l => l.area_mm2),
      would_add: sum(ex, l => l.area_mm2 * l.density_weight),
    };
    const g3 = lesions3d.filter(g => g.included).sort((a, b) => b.total_agatston - a.total_agatston);
    // The two highest-scoring slices, shown in slice order: the images that best explain the total.
    const keys = [...rows].sort((a, b) => b.agatston - a.agatston).slice(0, KEY_IMAGES).map(r => r.idx).sort((a, b) => a - b);

    return {
      study: inp.study, user: inp.user, now: inp.now, model: run.model_id, run_date: run.date,
      total, cat4, tier6: tierIn(total, 6), rows, totals, withheld, n_lesions_3d: g3.length,
      n_slices: slices.length, key_slices: keys, warnings, params: inp.params || [],
      conclusions: conclusions({ run, total, cat4, tier6: tierIn(total, 6), rows, totals, withheld, g3, nSlices: slices.length }),
    };
  }

  function conclusions({ run, total, cat4, tier6, rows, totals, withheld, g3, nSlices }) {
    const out = [];
    const ledger = withheld.n
      ? `${withheld.n} component${withheld.n > 1 ? 's were' : ' was'} withheld (below the minimum area of ` +
        `${fmt(run.min_area_mm2)} mm², or density weight 0): ${fmt(withheld.area_mm2, 2)} mm² in total, ` +
        `which would add ${fmt(withheld.would_add)} if admitted.`
      : 'No component was withheld.';
    if (!rows.length) {
      out.push(`No calcium was scored: ${nSlices} slices were predicted and none holds a component that counts.`);
      out.push(ledger);
      return out;
    }
    const top = rows.reduce((b, r) => r.agatston > b.agatston ? r : b);
    const pct = v => total > 0 ? `${fmt(100 * v / total, 0)}%` : '—';
    out.push(`Agatston score ${fmt(total)}: ${cat4 || '—'} on the 4-tier scale, ${tier6} on the 6-tier scale.`);
    out.push(`${totals.n_included} scored components form ${g3.length} 3D lesion${g3.length === 1 ? '' : 's'} on ` +
      `${rows.length} of ${nSlices} slices (z ${fmt(rows[0].z_mm)}–${fmt(rows[rows.length - 1].z_mm)} mm).`);
    out.push(`Largest contribution: slice ${top.idx} (z ${fmt(top.z_mm)} mm) with ${fmt(top.agatston)}, ${pct(top.agatston)} of the total` +
      (g3.length ? `; largest lesion ${g3[0].lesion_3d_key} with ${fmt(g3[0].total_agatston)} (${pct(g3[0].total_agatston)}) over ${g3[0].n_slices} slice(s).` : '.'));
    out.push(`Highest peak density ${fmt(totals.peak_hu, 0)} HU, density weight ${totals.weight ?? '—'} of 4.`);
    out.push(ledger);
    if (run.output === 'coverage') out.push('Areas are sums of per-voxel coverage, so partial voxels enter at their own fraction.');
    else if (run.output === 'binary') out.push(`Areas are voxel counts above the ${fmt(run.threshold, 2)} threshold; boundary voxels count fully or not at all.`);
    return out;
  }

  // ── model → one A4 HTML document ────────────────────────────────────────
  const CSS = `
    @page { size: A4; margin: 10mm; }
    * { box-sizing: border-box; }
    html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    body { margin: 10mm; font: 9pt/1.35 "Segoe UI", system-ui, Arial, sans-serif; color: #1b1b1b; }
    @media print { body { margin: 0; } }
    .page { width: 190mm; }
    header { display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 2px solid #1b1b1b; padding-bottom: 2mm; }
    h1 { font-size: 15pt; margin: 0; letter-spacing: .02em; }
    h1 small { display: block; font-size: 8pt; font-weight: 400; color: #555; letter-spacing: 0; }
    .meta { text-align: right; font-size: 8pt; color: #444; } .meta b { color: #1b1b1b; }
    h2 { font-size: 8.5pt; text-transform: uppercase; letter-spacing: .08em; margin: 4mm 0 1.5mm; color: #555; }
    .top { display: grid; grid-template-columns: 50mm 1fr auto; gap: 5mm; margin-top: 3mm; align-items: stretch; }
    .score { border: 1px solid #bbb; padding: 3mm; align-self: start; } .score .n { font-size: 30pt; font-weight: 700; line-height: 1.05; }
    .score .t { font-size: 11pt; font-weight: 700; } .score .t2 { font-size: 8pt; color: #555; }
    .stats { display: grid; grid-template-columns: auto 1fr; gap: .8mm 4mm; align-content: start; font-size: 8.5pt; }
    .stats span:nth-child(odd) { color: #555; }
    .keys { display: flex; gap: 2.5mm; align-items: flex-start; } .keys > div { width: 38mm; }
    .key { position: relative; background: #000; aspect-ratio: var(--ar, 1); border-radius: 2px; overflow: hidden; }
    .key img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; image-rendering: pixelated; }
    .key.broken::after { content: "key-slice image unavailable"; position: absolute; inset: 0; display: grid; place-items: center; color: #aaa; font-size: 7pt; }
    .key.broken img { display: none; }
    .cap { font-size: 7pt; color: #555; margin-top: .8mm; }
    .tables { display: grid; gap: 4mm; } .tables.two { grid-template-columns: 1fr 1fr; }
    table { width: 100%; border-collapse: collapse; font-size: 8.5pt; }
    th { text-align: right; font-weight: 600; border-bottom: 1.5px solid #1b1b1b; padding: .8mm 1.5mm; }
    td { text-align: right; padding: .6mm 1.5mm; border-bottom: .5px solid #ddd; font-variant-numeric: tabular-nums; }
    th:first-child, td:first-child { text-align: left; } td.b { font-weight: 700; }
    tfoot td { border-top: 1.5px solid #1b1b1b; border-bottom: 0; font-weight: 700; }
    .none { font-size: 8pt; color: #555; margin-top: 1mm; }
    ul { margin: 0; padding-left: 4.5mm; } li { margin-bottom: .8mm; }
    .method { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3mm; font-size: 7.5pt; }
    .method h4 { margin: 0 0 .5mm; font-size: 7.5pt; color: #555; text-transform: uppercase; letter-spacing: .06em; }
    .method dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 0 2mm; } .method dt { color: #666; } .method dd { margin: 0; word-break: break-word; }
    .warn { border: 1.5px solid #c0392b; color: #8e2418; background: #fdecea; padding: 1.5mm 2.5mm; margin-top: 3mm; font-size: 8pt; }
    .disc { margin-top: 4mm; border-top: 1px solid #bbb; padding-top: 1.5mm; font-size: 7.5pt; color: #444; }
    @media screen { body { background: #e5e5e5; } .page { background: #fff; padding: 0; } }`;

  function rowHtml(r) {
    return `<tr><td>${r.idx}</td><td>${fmt(r.z_mm)}</td><td>${r.n_included} / ${r.n_all}</td>` +
      `<td>${fmt(r.area_mm2, 2)}</td><td>${r.peak_hu == null ? '—' : fmt(r.peak_hu, 0)}</td>` +
      `<td>${r.weight ?? '—'}</td><td class="b">${fmt(r.agatston)}</td></tr>`;
  }

  function tableHtml(rows, foot) {
    return `<table><thead><tr><th>Slice</th><th>z (mm)</th><th>Lesions</th><th>Area (mm²)</th>` +
      `<th>Peak HU</th><th>Weight</th><th>Agatston</th></tr></thead><tbody>${rows.map(rowHtml).join('')}</tbody>` +
      (foot ? `<tfoot><tr><td>Total</td><td></td><td>${foot.n_included} / ${foot.n_all}</td><td>${fmt(foot.area_mm2, 2)}</td>` +
        `<td>${foot.peak_hu == null ? '—' : fmt(foot.peak_hu, 0)}</td><td>${foot.weight ?? '—'}</td><td>${fmt(foot.agatston)}</td></tr></tfoot>` : '') +
      `</table>`;
  }

  /* img: { aspect, slices: [{ idx, ct, mask }] } URLs of the key slices' PNGs (the viewer's own files), or null. */
  function renderReportHtml(m, img) {
    const color = TIER_HEX[m.cat4] || '#1b1b1b';
    const half = Math.ceil(m.rows.length / 2), two = m.rows.length > TWO_COLUMN_FROM;
    const tables = !m.rows.length ? '<p class="none">No slice holds a scored component.</p>'
      : two ? `<div class="tables two">${tableHtml(m.rows.slice(0, half))}${tableHtml(m.rows.slice(half), m.totals)}</div>`
        : `<div class="tables">${tableHtml(m.rows, m.totals)}</div>`;
    const keyBox = img && img.slices.length ? `<div class="keys">${img.slices.map(s => {
      const r = m.rows.find(x => x.idx === s.idx);
      return `<div><div class="key" style="--ar:${img.aspect}">` +
        `<img src="${esc(s.ct)}" alt="" onerror="this.parentNode.classList.add('broken')">` +
        `<img src="${esc(s.mask)}" alt="" onerror="this.parentNode.classList.add('broken')"></div>` +
        `<div class="cap">Slice ${r.idx} · z ${fmt(r.z_mm)} mm · score ${fmt(r.agatston)}</div></div>`;
    }).join('')}</div>` : '<div></div>';
    const nSl = m.rows.length;
    return `<!doctype html><html><head><meta charset="utf-8"><title>${esc(`PrediCT_${m.study}_${m.model}`)}</title><style>${CSS}</style></head><body><div class="page">
<header><h1>Coronary Calcium Report<small>PrediCT Studio · automated Agatston scoring</small></h1>
<div class="meta">Study <b>${esc(m.study)}</b> · Model <b>${esc(m.model)}</b><br>Run ${esc(fmtDate(m.run_date))} · Report ${esc(fmtDate(m.now))} · ${esc(m.user)}</div></header>
${m.warnings.length ? `<div class="warn"><b>Check before use — numbers disagree:</b><ul>${m.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul></div>` : ''}
<div class="top">
 <div class="score"><div class="n">${fmt(m.total)}</div><div class="t" style="color:${color}">${esc(m.cat4 || '—')}</div>
  <div class="t2">4-tier · 6-tier: ${esc(m.tier6)}</div></div>
 <div class="stats"><span>Scored components</span><span>${m.totals.n_included} of ${m.totals.n_included + m.withheld.n}</span>
  <span>3D lesions</span><span>${m.n_lesions_3d}</span><span>Slices with calcium</span><span>${nSl} of ${m.n_slices}</span>
  <span>Peak density</span><span>${m.totals.peak_hu == null ? '—' : fmt(m.totals.peak_hu, 0) + ' HU'}</span>
  <span>Total area</span><span>${fmt(m.totals.area_mm2, 2)} mm²</span><span>Withheld</span><span>${m.withheld.n}</span></div>
 ${keyBox}
</div>
<h2>Slice by slice</h2>${tables}
${m.rows.length ? `<div class="none">The other ${m.n_slices - nSl} slices hold no scored calcium. Lesions = counted / all components found on the slice.</div>` : ''}
<h2>Conclusions</h2><ul>${m.conclusions.map(c => `<li>${esc(c)}</li>`).join('')}</ul>
<h2>Method (read from run.json)</h2><div class="method">${m.params.map(([g, kv]) =>
      `<div><h4>${esc(g)}</h4><dl>${kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl></div>`).join('')}</div>
<div class="disc">Automated research output, not a diagnosis. Calcium is scored on 3 mm slices and is not attributed to individual vessels.
It must be reviewed by a qualified clinician. Every figure above was read from the result files of this run.</div>
</div></body></html>`;
  }

  // The slice table as CSV: one self-describing row per slice (study and model on every row),
  // then a TOTAL row. A study name is typed by users, so a leading = + - @ is defused
  // (spreadsheet formula injection) and separators are quoted.
  function buildCsv(m) {
    const cell = v => {
      let s = v == null ? '' : String(v);
      if (/^[=+\-@\t\r]/.test(s) && typeof v === 'string') s = `'${s}`;
      return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const r2 = v => (v == null ? '' : Number(v).toFixed(2));
    const line = (slice, z, r) => [m.study, m.model, slice, z, r.n_included, r.n_all, r2(r.area_mm2),
      r.peak_hu ?? '', r.weight ?? '', r2(r.agatston)].map(cell).join(',');
    return ['study,model,slice_idx,z_mm,lesions_counted,lesions_all,area_mm2,peak_hu,max_density_weight,agatston',
      ...m.rows.map(r => line(r.idx, r.z_mm, r)), line('TOTAL', '', m.totals)].join('\r\n') + '\r\n';
  }

  const fmtDate = iso => { const d = new Date(iso); return iso && !isNaN(d) ? d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '—'; };

  if (typeof module !== 'undefined' && module.exports) {   // Node (tests): no DOM below this line
    module.exports = { buildReportModel, renderReportHtml, buildCsv };
    return;
  }

  // ── browser glue: build the page in a hidden iframe and print it ────────
  let frame = null;

  // The open result as a report model, or null (with a message) when it cannot be built.
  function openResultModel() {
    if (!HAS_STUDY || !state.run) { toast('Open a result first, then export.', 'error'); return null; }
    try {
      return buildReportModel({
        run: state.run, slices: state.slices, lesions: state.lesions, lesions3d: state.lesions3d,
        tierSchemes: TIER_SCHEMES, params: paramGroups(), study: STUDY, user: ME ? ME.username : null,
        now: new Date().toISOString(),
      });
    } catch (e) { toast(`Report not created: ${e.message}`, 'error'); return null; }
  }

  async function exportReport() {
    const model = openResultModel();
    if (!model) return;
    const abs = u => new URL(u, location.href).href;
    const img = {
      aspect: state.imgW && state.imgH ? `${state.imgW} / ${state.imgH}` : '1',
      slices: model.key_slices.map(i => ({ idx: i, ct: abs(ctUrl(i)), mask: abs(maskUrl(i)) })),
    };
    if (frame) frame.remove();
    frame = document.createElement('iframe');
    frame.title = 'PrediCT report';
    // Off-screen but laid out at A4 width: a display:none frame prints blank in some browsers.
    frame.style.cssText = 'position:fixed;left:-10000px;top:0;width:210mm;height:297mm;border:0';
    const loaded = new Promise(res => { frame.onload = res; });
    frame.srcdoc = renderReportHtml(model, img);
    document.body.append(frame);
    await loaded;

    const doc = frame.contentDocument;
    await Promise.all([...doc.images].map(i => i.complete ? null : new Promise(r => { i.onload = i.onerror = r; })));
    if (doc.querySelector('.page').offsetHeight > PAGE_HEIGHT_MM * MM_TO_PX)
      toast('This report is longer than one page and will print on two.', 'error');
    frame.contentWindow.focus();
    frame.contentWindow.print();
  }

  // The slice table as a CSV the browser downloads from memory (no server round trip).
  function exportCsv() {
    const model = openResultModel();
    if (!model) return;
    const name = `PrediCT_${model.study}_${model.model}_slices.csv`.replace(/[^\w.-]+/g, '_');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob(['\ufeff', buildCsv(model)], { type: 'text/csv;charset=utf-8' }));
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  for (const [id, fn] of [['report-btn', exportReport], ['report-csv-btn', exportCsv]]) {
    const btn = document.getElementById(id);
    if (!btn) continue;
    if (HAS_STUDY) btn.onclick = fn; else btn.style.display = 'none';
  }
})();
