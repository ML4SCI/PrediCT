/* PrediCT Studio — one-page clinical result report.
 *
 * Turns the result that is already open in the viewer into a modern one-page A4
 * medical report card (PrediCT Studio header, hero score card with risk spectrum,
 * axial Z-distribution bar, key-slice CT overlays, lesion-wise table including
 * all scored and withheld components, clinical conclusions, and pipeline method)
 * and opens the browser's print dialog, where "Save as PDF" makes the document.
 *
 * Why it is separate: app.js has already loaded run.json, slices.json and both
 * CSVs into `state`, so the report needs nothing new from the server. It
 * only reads that state; it is loaded after app.js and app.js does not know it
 * exists.
 *
 * Does NOT: write any file, call any new server route, recompute a score, or
 * change `state`. Every number is copied from the result files; the only
 * arithmetic is summing per-lesion values to CHECK them against slices.json and
 * run.json, and a disagreement is printed on the report instead of being hidden.
 *
 * Usage: the two icons in the Argument title bar (#report-btn, #report-csv-btn)
 *        call exportReport() / exportCsv(). They are not in the header, which is
 *        already full at 100% browser zoom.
 *        buildReportModel() is pure (no DOM), so tests/report_check.mjs runs it in Node.
 */
(function () {
  'use strict';

  const PAGE_HEIGHT_MM = 277;       // A4 297 mm minus the 10 mm @page margin top and bottom
  const MM_TO_PX = 96 / 25.4;       // CSS reference pixel
  const TWO_COLUMN_FROM = 12;       // more lesion rows than this: split into two side-by-side tables
  const TOL_SLICE = 0.01;           // lesions.csv vs slices.json, per slice
  const TOL_TOTAL = 0.05;           // sum of included lesions vs run.json total
  const KEY_IMAGES = 2;             // top-scoring slices shown as thumbnails
  const TIER_SCALE_MAX = 1400;      // matches app.js TIER_SCALE_MAX for the risk spectrum bar

  // Print-safe clinical palette per tier. Tier NAMES come from app.js (TIER_SCHEMES).
  const TIER_HEX = {
    ZERO: '#64748b', MILD: '#15803d', MODERATE: '#b45309',
    'MOD-HIGH': '#c2410c', SEVERE: '#b91c1c', EXTENSIVE: '#7f1d1d',
  };
  const TIER_BG = {
    ZERO: '#f1f5f9', MILD: '#dcfce7', MODERATE: '#fef3c7',
    'MOD-HIGH': '#ffedd5', SEVERE: '#fee2e2', EXTENSIVE: '#fecaca',
  };

  const fmt = (v, d = 1) => (v == null || v === '' || Number.isNaN(Number(v))) ? '—' : Number(v).toFixed(d);
  const esc = s => String(s ?? '—').replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const sum = (xs, f) => xs.reduce((a, x) => a + f(x), 0);
  const max = xs => (xs.length ? Math.max(...xs) : null);

  // ── data → plain report model (no DOM) ──────────────────────────────────
  /* in: { run, slices, lesions, lesions3d, tierSchemes, params, study, user, now }
   *   slices     slices.json rows; lesions  lesions.csv rows; lesions3d  lesions_3d.csv rows
   *   tierSchemes  app.js TIER_SCHEMES ({4: [[maxInclusive, name]…], 6: …})
   *   params     app.js paramGroups() — the method block, already read from run.json
   * Throws when run.json has no total: a report with a guessed total is worse than none. */
  function buildReportModel(inp) {
    const { run, slices, lesions, lesions3d, tierSchemes } = inp;
    if (!run || typeof run.agatston_total !== 'number') throw new Error('run.json has no agatston_total.');
    const total = run.agatston_total;
    const tierIn = (t, n) => tierSchemes[n].find(([m]) => t <= m)[1];
    const warnings = [];

    const cat4 = run.risk_category ? String(run.risk_category).toUpperCase() : null;
    if (cat4 && cat4 !== tierIn(total, 4))
      warnings.push(`run.json says ${cat4} but the 4-tier reading of ${fmt(total)} is ${tierIn(total, 4)}.`);

    const sliceMap = new Map(slices.map(s => [s.idx, s]));
    const bySlice = new Map();
    lesions.forEach(l => bySlice.set(l.slice_idx, [...(bySlice.get(l.slice_idx) || []), l]));

    const calcSlices = slices.filter(s => s.has_calcium).sort((a, b) => a.idx - b.idx);
    calcSlices.forEach(s => {
      const inc = (bySlice.get(s.idx) || []).filter(l => l.included);
      const fromCsv = sum(inc, l => l.agatston);
      if (Math.abs(fromCsv - s.slice_score) > TOL_SLICE)
        warnings.push(`Slice ${s.idx}: lesions.csv sums to ${fmt(fromCsv, 2)}, slices.json says ${fmt(s.slice_score, 2)}.`);
    });

    // Order lesions the same way as the Instrument tab (02 Instrument): by 3D lesion, then slice.
    const ordered = [...lesions].sort((a, b) =>
      (a.lesion_3d_id ?? a.lesion_id ?? 0) - (b.lesion_3d_id ?? b.lesion_id ?? 0) ||
      a.slice_idx - b.slice_idx ||
      (a.lesion_id ?? 0) - (b.lesion_id ?? 0));

    let prevKey = null;
    const rows = ordered.map(l => {
      const key = l.lesion_3d_key || `L${String(l.lesion_3d_id ?? l.lesion_id ?? '').padStart(3, '0')}`;
      const first = key !== prevKey;
      prevKey = key;
      const sm = sliceMap.get(l.slice_idx);
      const inc = Boolean(l.included);
      return {
        idx: l.slice_idx,
        z_mm: l.z_mm != null ? l.z_mm : (sm ? sm.z_mm : null),
        lesion_id: l.lesion_id,
        lesion_3d_key: key,
        first_in_3d: first,
        included: inc,
        area_mm2: l.area_mm2,
        peak_hu: l.peak_hu,
        weight: l.density_weight,
        agatston: inc ? l.agatston : 0,
      };
    });

    const incRows = rows.filter(r => r.included);
    const totals = {
      n_included: incRows.length,
      n_all: rows.length,
      area_mm2: sum(incRows, r => r.area_mm2),
      peak_hu: max(incRows.map(r => r.peak_hu).filter(v => v != null)),
      weight: max(incRows.map(r => r.weight).filter(v => v != null)),
      agatston: sum(incRows, r => r.agatston),
    };
    if (Math.abs(totals.agatston - total) > TOL_TOTAL)
      warnings.push(`Included lesions sum to ${fmt(totals.agatston, 2)} but run.json total is ${fmt(total, 2)}.`);
    if (run.n_lesions != null && run.n_lesions !== totals.n_included)
      warnings.push(`run.json counts ${run.n_lesions} scored components, the lesion table has ${totals.n_included}.`);

    const ex = lesions.filter(l => !l.included);
    const withheld = {
      n: ex.length,
      area_mm2: sum(ex, l => l.area_mm2),
      would_add: sum(ex, l => l.area_mm2 * l.density_weight),
    };
    const g3 = lesions3d.filter(g => g.included).sort((a, b) => b.total_agatston - a.total_agatston);

    // The two highest-scoring slices, shown in slice order: the images that best explain the total.
    const keys = [...calcSlices].sort((a, b) => b.slice_score - a.slice_score)
      .slice(0, KEY_IMAGES).map(s => s.idx).sort((a, b) => a - b);

    // Axial profile across the scan for the Z-distribution visual.
    const zProfile = slices.map(s => ({
      idx: s.idx, z_mm: s.z_mm, score: s.slice_score || 0, has_calcium: Boolean(s.has_calcium),
    }));

    return {
      study: inp.study, user: inp.user, now: inp.now, model: run.model_id, run_date: run.date,
      total, cat4, tier6: tierIn(total, 6), rows, calc_slices: calcSlices, z_profile: zProfile,
      totals, withheld, n_lesions_3d: g3.length, n_slices: slices.length, key_slices: keys,
      warnings, params: inp.params || [],
      conclusions: conclusions({
        run, total, cat4, tier6: tierIn(total, 6), calcSlices, totals, withheld, g3, nSlices: slices.length,
      }),
    };
  }

  function conclusions({ run, total, cat4, tier6, calcSlices, totals, withheld, g3, nSlices }) {
    const out = [];
    const ledger = withheld.n
      ? `${withheld.n} sub-threshold component${withheld.n > 1 ? 's were' : ' was'} withheld (< ${fmt(run.min_area_mm2)} mm² or weight 0): ` +
        `${fmt(withheld.area_mm2, 2)} mm² total (${fmt(withheld.would_add)} Agatston if admitted).`
      : 'No sub-threshold component was withheld.';
    if (!calcSlices.length) {
      out.push(`No coronary calcium was scored across ${nSlices} evaluated slices.`);
      out.push(ledger);
      return out;
    }
    const top = calcSlices.reduce((b, s) => s.slice_score > b.slice_score ? s : b);
    const pct = v => total > 0 ? `${fmt(100 * v / total, 0)}%` : '—';
    out.push(`Total Agatston score ${fmt(total)} places this study in the ${cat4 || '—'} risk category (4-tier) / ${tier6} (6-tier).`);
    out.push(`${totals.n_included} scored components group into ${g3.length} 3D lesion${g3.length === 1 ? '' : 's'} across ` +
      `${calcSlices.length} of ${nSlices} slices (z ${fmt(calcSlices[0].z_mm)}–${fmt(calcSlices[calcSlices.length - 1].z_mm)} mm).`);
    out.push(`Peak burden on slice ${top.idx} (z ${fmt(top.z_mm)} mm) contributing ${fmt(top.slice_score)} (${pct(top.slice_score)})` +
      (g3.length ? `; largest 3D lesion ${g3[0].lesion_3d_key} contributes ${fmt(g3[0].total_agatston)} (${pct(g3[0].total_agatston)}) over ${g3[0].n_slices} slice(s).` : '.'));
    out.push(`Maximum attenuation ${fmt(totals.peak_hu, 0)} HU (density weight ${totals.weight ?? '—'} of 4). ${ledger}`);
    if (run.output === 'coverage') out.push('Soft coverage model: partial boundary voxels contribute proportionally to lesion area.');
    else if (run.output === 'binary') out.push(`Binary segmentation model: voxel counts thresholded at ${fmt(run.threshold, 2)}.`);
    return out;
  }

  // ── visuals (pure SVG/HTML, no external requests) ───────────────────────
  // 1. Risk spectrum bar showing where the patient's total sits across the 4 clinical bounds.
  function renderSpectrumHtml(total, color) {
    const pct = Math.max(0, Math.min(100, (total / TIER_SCALE_MAX) * 100));
    const t100 = (100 / TIER_SCALE_MAX) * 100;
    const t400 = (400 / TIER_SCALE_MAX) * 100;
    return `<div class="spec">` +
      `<div class="spec-track">` +
      `<i class="spec-fill" style="width:${pct.toFixed(1)}%;background:${color}"></i>` +
      `<u style="left:${t100.toFixed(1)}%"></u><u style="left:${t400.toFixed(1)}%"></u>` +
      `<b class="spec-pin" style="left:${pct.toFixed(1)}%;border-color:${color}"></b>` +
      `</div>` +
      `<div class="spec-labels"><span>0 Zero</span><span>1–100 Mild</span><span>101–400 Mod</span><span>&gt;400 Severe</span></div>` +
      `</div>`;
  }

  // 2. Axial cranio-caudal (Z) calcium distribution bar across all slices of the scan.
  function renderAxialChartHtml(m, color) {
    const prof = m.z_profile || [];
    if (!prof.length) return '';
    const maxSc = Math.max(...prof.map(s => s.score), 1);
    const keySet = new Set(m.key_slices || []);
    const W = 520, H = 32, padB = 9, barH = H - padB - 3;
    const step = W / prof.length;
    const bw = Math.max(1.8, Math.min(8, step * 0.72));
    const bars = prof.map((s, i) => {
      const x = (i + 0.5) * step - bw / 2;
      if (!s.has_calcium || s.score <= 0) {
        return `<rect x="${x.toFixed(1)}" y="${(barH + 1).toFixed(1)}" width="${bw.toFixed(1)}" height="1.5" fill="#cbd5e1"/>`;
      }
      const h = Math.max(3, (s.score / maxSc) * barH);
      const y = barH + 2 - h;
      const fill = keySet.has(s.idx) ? color : '#0f172a';
      return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="1" fill="${fill}"/>`;
    }).join('');
    const z0 = fmt(prof[0].z_mm, 0), z1 = fmt(prof[prof.length - 1].z_mm, 0);
    return `<div class="axial-card">` +
      `<div class="axial-head"><span>Axial Calcium Distribution (Cranio-Caudal Z-Profile)</span>` +
      `<span class="axial-sub">${m.calc_slices.length} of ${m.n_slices} slices · z ${z0} to ${z1} mm · peak slice ${fmt(maxSc)} Agatston</span></div>` +
      `<svg viewBox="0 0 ${W} ${H}" class="axial-svg" preserveAspectRatio="none">${bars}` +
      `<text x="0" y="${H - 1}" class="ax-t">sl 0 (${z0} mm)</text>` +
      `<text x="${W}" y="${H - 1}" text-anchor="end" class="ax-t">sl ${prof[prof.length - 1].idx} (${z1} mm)</text>` +
      `</svg></div>`;
  }

  // ── model → one A4 HTML document ────────────────────────────────────────
  const CSS = `
    @page { size: A4; margin: 9mm 10mm; }
    * { box-sizing: border-box; }
    html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    body { margin: 9mm 10mm; font: 8.5pt/1.35 "Inter", "Segoe UI", system-ui, -apple-system, Arial, sans-serif; color: #0f172a; }
    @media print { body { margin: 0; } }
    .page { width: 190mm; }

    /* Brand Header */
    .hdr { display: flex; justify-content: space-between; align-items: center; padding-bottom: 2.5mm; border-bottom: 2px solid #0f172a; }
    .brand { display: flex; align-items: center; gap: 3mm; }
    .brand-mark { width: 9.5mm; height: 9.5mm; border-radius: 2mm; background: #0f172a; color: #fff; display: grid; place-items: center; flex-shrink: 0; }
    .brand-mark svg { width: 5.8mm; height: 5.8mm; }
    .brand-name { font-size: 16pt; font-weight: 800; letter-spacing: -0.02em; color: #0f172a; line-height: 1; }
    .brand-name span { color: #2563eb; }
    .brand-sub { font-size: 7.5pt; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin-top: 0.8mm; }
    .hdr-meta { display: grid; grid-template-columns: repeat(2, auto); gap: 0.6mm 4.5mm; text-align: right; font-size: 7.8pt; color: #475569; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 1.8mm; padding: 1.8mm 3mm; }
    .hdr-meta b { color: #0f172a; font-weight: 600; }

    /* Hero Summary Row */
    .hero { display: grid; grid-template-columns: 55mm 1fr auto; gap: 2.8mm; margin-top: 2.8mm; align-items: stretch; }
    .score-card { border: 1px solid #cbd5e1; border-top: 3.5px solid var(--tier-c, #0f172a); border-radius: 2mm; padding: 2.2mm 2.8mm; background: #ffffff; display: flex; flex-direction: column; justify-content: space-between; }
    .sc-kicker { font-size: 6.8pt; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; }
    .sc-main { display: flex; align-items: baseline; justify-content: space-between; gap: 2mm; margin: 0.6mm 0; }
    .sc-val { font-size: 25pt; font-weight: 800; letter-spacing: -0.03em; line-height: 1; color: #0f172a; font-variant-numeric: tabular-nums; }
    .sc-badge { font-size: 7.8pt; font-weight: 800; padding: 0.6mm 2mm; border-radius: 99px; letter-spacing: 0.04em; }
    .sc-sub { font-size: 7pt; color: #64748b; margin-bottom: 1mm; }
    .spec-track { position: relative; height: 2.2mm; background: #e2e8f0; border-radius: 99px; overflow: visible; }
    .spec-fill { display: block; height: 100%; border-radius: 99px; }
    .spec-track u { position: absolute; top: 0; bottom: 0; width: 1px; background: #ffffff; text-decoration: none; }
    .spec-pin { position: absolute; top: -0.9mm; width: 4mm; height: 4mm; margin-left: -2mm; border-radius: 50%; background: #fff; border: 2px solid #0f172a; }
    .spec-labels { display: flex; justify-content: space-between; font-size: 6.2pt; color: #64748b; margin-top: 1mm; }

    /* Key Metrics Card */
    .metrics-card { border: 1px solid #e2e8f0; border-radius: 2mm; background: #f8fafc; padding: 2.4mm 3.2mm; display: grid; grid-template-columns: repeat(3, auto); justify-content: space-between; gap: 2mm 3mm; align-content: space-between; min-width: 0; }
    .m-item { display: flex; flex-direction: column; }
    .m-lbl { font-size: 6.4pt; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; white-space: nowrap; }
    .m-val { font-size: 10.5pt; font-weight: 800; color: #0f172a; font-variant-numeric: tabular-nums; line-height: 1.2; white-space: nowrap; margin-top: 0.4mm; }
    .m-val small { font-size: 7pt; font-weight: 500; color: #64748b; }

    /* Key Slice Images */
    .keys { display: flex; gap: 2mm; align-items: stretch; }
    .key-card { width: 29.5mm; border: 1px solid #cbd5e1; border-radius: 2mm; padding: 1.1mm; background: #fff; display: flex; flex-direction: column; justify-content: space-between; }
    .key { position: relative; background: #000; aspect-ratio: var(--ar, 1); border-radius: 1.2mm; overflow: hidden; }
    .key img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; image-rendering: pixelated; }
    .key.broken::after { content: "image unavailable"; position: absolute; inset: 0; display: grid; place-items: center; color: #94a3b8; font-size: 6.5pt; }
    .key.broken img { display: none; }
    .cap { font-size: 6.6pt; color: #334155; margin-top: 1mm; line-height: 1.2; text-align: center; font-variant-numeric: tabular-nums; }
    .cap b { color: #0f172a; }

    /* Axial Distribution Strip */
    .axial-card { margin-top: 2.5mm; border: 1px solid #e2e8f0; border-radius: 1.8mm; background: #f8fafc; padding: 1.5mm 3mm 1.2mm; }
    .axial-head { display: flex; justify-content: space-between; align-items: baseline; font-size: 7pt; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #475569; margin-bottom: 0.8mm; }
    .axial-sub { font-weight: 500; text-transform: none; letter-spacing: 0; color: #64748b; }
    .axial-svg { width: 100%; height: 8.5mm; display: block; }
    .ax-t { font-size: 7px; fill: #64748b; font-family: inherit; }

    /* Section Headings */
    .sec-head { display: flex; justify-content: space-between; align-items: baseline; gap: 3mm; margin: 2.8mm 0 1.2mm; border-bottom: 1px solid #cbd5e1; padding-bottom: 0.6mm; }
    .sec-head h2 { font-size: 8pt; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; margin: 0; color: #0f172a; white-space: nowrap; }
    .sec-head span { font-size: 6.9pt; color: #64748b; text-align: right; white-space: nowrap; }

    /* Lesion-wise Table */
    .tables { display: grid; gap: 3.5mm; align-items: start; }
    .tables.two { grid-template-columns: 1fr 1fr; }
    table { width: 100%; border-collapse: collapse; font-size: 7.6pt; }
    th { text-align: right; font-size: 7.2pt; font-weight: 700; color: #334155; background: #f1f5f9; border-top: 1px solid #cbd5e1; border-bottom: 1px solid #cbd5e1; padding: 0.9mm 1.4mm; }
    td { text-align: right; padding: 0.55mm 1.4mm; border-bottom: 0.5px solid #e2e8f0; font-variant-numeric: tabular-nums; color: #1e293b; }
    th:first-child, td:first-child, th:nth-child(3), td:nth-child(3) { text-align: left; }
    tr.g3-first td { border-top: 1px solid #cbd5e1; }
    tbody tr:first-child td { border-top: none; }
    td.b { font-weight: 700; color: #0f172a; }
    td.les { font-weight: 600; color: #0f172a; }
    td.les-cont { color: #64748b; font-weight: 500; }
    tr.ex td { color: #94a3b8; background: #fafafa; }
    .tag-ex { font-size: 6.2pt; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.03em; margin-left: 1mm; }
    tfoot td { border-top: 1.5px solid #0f172a; border-bottom: 1.5px solid #0f172a; background: #f8fafc; font-weight: 700; color: #0f172a; padding: 0.9mm 1.4mm; }
    .none { font-size: 7.5pt; color: #64748b; margin: 1mm 0 0; }

    /* Bottom Two-Column Split: Findings + Method */
    .bottom { display: grid; grid-template-columns: 1.15fr 1fr; gap: 3.5mm; margin-top: 2.5mm; align-items: stretch; }
    .card-box { border: 1px solid #e2e8f0; border-radius: 1.8mm; background: #f8fafc; padding: 2mm 2.8mm; }
    .card-box h3 { font-size: 7.5pt; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; color: #0f172a; margin: 0 0 1.2mm; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.6mm; }
    ul.findings { margin: 0; padding-left: 3.8mm; font-size: 7.6pt; color: #1e293b; }
    ul.findings li { margin-bottom: 0.7mm; }
    .method { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5mm 3mm; font-size: 7pt; }
    .method h4 { margin: 0 0 0.4mm; font-size: 6.6pt; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }
    .method dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 0.2mm 2mm; }
    .method dt { color: #64748b; }
    .method dd { margin: 0; color: #0f172a; font-weight: 500; word-break: break-word; }

    .warn { border: 1.5px solid #b91c1c; color: #7f1d1d; background: #fef2f2; border-radius: 1.5mm; padding: 1.5mm 2.5mm; margin-top: 2.5mm; font-size: 7.8pt; }
    .disc { margin-top: 2.2mm; border-top: 1px solid #cbd5e1; padding-top: 1.2mm; display: flex; justify-content: space-between; gap: 3mm; font-size: 6.7pt; color: #64748b; white-space: nowrap; }
    @media screen { body { background: #e2e8f0; } .page { box-sizing: content-box; background: #fff; padding: 8mm 10mm; border-radius: 2mm; box-shadow: 0 2px 12px rgba(0,0,0,.12); } }`;

  function rowHtml(r) {
    const lesCell = r.first_in_3d
      ? `<td class="les">${esc(r.lesion_3d_key)}${r.included ? '' : '<span class="tag-ex">withheld</span>'}</td>`
      : `<td class="les-cont">↳ ${esc(r.lesion_3d_key)}${r.included ? '' : '<span class="tag-ex">withheld</span>'}</td>`;
    const cls = [r.first_in_3d ? 'g3-first' : '', r.included ? '' : 'ex'].filter(Boolean).join(' ');
    return `<tr${cls ? ` class="${cls}"` : ''}><td>sl ${r.idx}</td><td>${fmt(r.z_mm)}</td>${lesCell}` +
      `<td>${fmt(r.area_mm2, 2)}</td><td>${r.peak_hu == null ? '—' : fmt(r.peak_hu, 0)}</td>` +
      `<td>${r.weight ?? '—'}</td><td class="b">${r.included ? fmt(r.agatston) : '—'}</td></tr>`;
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
    const color = TIER_HEX[m.cat4] || '#0f172a';
    const badgeBg = TIER_BG[m.cat4] || '#f1f5f9';
    const half = Math.ceil(m.rows.length / 2), two = m.rows.length > TWO_COLUMN_FROM;
    const tables = !m.rows.length ? '<p class="none">No lesion component above the delineation threshold was detected in this scan.</p>'
      : two ? `<div class="tables two">${tableHtml(m.rows.slice(0, half))}${tableHtml(m.rows.slice(half), m.totals)}</div>`
        : `<div class="tables">${tableHtml(m.rows, m.totals)}</div>`;
    const keyBox = img && img.slices.length ? `<div class="keys">${img.slices.map(s => {
      const cs = (m.calc_slices || []).find(x => x.idx === s.idx);
      const z = cs ? fmt(cs.z_mm) : '—', sc = cs ? fmt(cs.slice_score) : '—';
      return `<div class="key-card"><div class="key" style="--ar:${img.aspect}">` +
        `<img src="${esc(s.ct)}" alt="" onerror="this.parentNode.classList.add('broken')">` +
        `<img src="${esc(s.mask)}" alt="" onerror="this.parentNode.classList.add('broken')"></div>` +
        `<div class="cap"><b>Slice ${s.idx}</b> · z ${z} mm<br>Score ${sc}</div></div>`;
    }).join('')}</div>` : '<div></div>';
    const nCalcSl = (m.calc_slices || []).length;
    const brandIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/></svg>`;

    return `<!doctype html><html><head><meta charset="utf-8"><title>${esc(`PrediCT_${m.study}_${m.model}`)}</title><style>${CSS}</style></head><body><div class="page">
<header class="hdr">
  <div class="brand">
    <div class="brand-mark">${brandIcon}</div>
    <div>
      <div class="brand-name">Predi<span>CT</span> Studio</div>
      <div class="brand-sub">Coronary Artery Calcium Scoring Report</div>
    </div>
  </div>
  <div class="hdr-meta">
    <span>Study: <b>${esc(m.study)}</b></span><span>Model: <b>${esc(m.model)}</b></span>
    <span>Scan Run: <b>${esc(fmtDate(m.run_date))}</b></span><span>Report: <b>${esc(fmtDate(m.now))}</b> · <b>${esc(m.user)}</b></span>
  </div>
</header>
${m.warnings.length ? `<div class="warn"><b>Check before use — numbers disagree:</b><ul>${m.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul></div>` : ''}
<div class="hero">
  <div class="score-card" style="--tier-c:${color}">
    <div>
      <div class="sc-kicker">Total Agatston Score</div>
      <div class="sc-main">
        <div class="sc-val">${fmt(m.total)}</div>
        <div class="sc-badge" style="color:${color};background:${badgeBg}">${esc(m.cat4 || '—')}</div>
      </div>
      <div class="sc-sub">4-tier: <b>${esc(m.cat4 || '—')}</b> · 6-tier: <b>${esc(m.tier6)}</b></div>
    </div>
    ${renderSpectrumHtml(m.total, color)}
  </div>
  <div class="metrics-card">
    <div class="m-item"><span class="m-lbl">Components</span><span class="m-val">${m.totals.n_included} <small>of ${m.totals.n_all} scored</small></span></div>
    <div class="m-item"><span class="m-lbl">3D Lesions</span><span class="m-val">${m.n_lesions_3d} <small>grouped</small></span></div>
    <div class="m-item"><span class="m-lbl">Ca+ Slices</span><span class="m-val">${nCalcSl} <small>of ${m.n_slices}</small></span></div>
    <div class="m-item"><span class="m-lbl">Calcified Area</span><span class="m-val">${fmt(m.totals.area_mm2, 2)} <small>mm²</small></span></div>
    <div class="m-item"><span class="m-lbl">Peak Density</span><span class="m-val">${m.totals.peak_hu == null ? '—' : fmt(m.totals.peak_hu, 0)} <small>HU (wt ${m.totals.weight ?? '—'})</small></span></div>
    <div class="m-item"><span class="m-lbl">Withheld</span><span class="m-val">${m.withheld.n} <small>(${fmt(m.withheld.area_mm2, 2)} mm²)</small></span></div>
  </div>
  ${keyBox}
</div>
${renderAxialChartHtml(m, color)}
<div class="sec-head">
  <h2>Lesion-Wise Slice Breakdown</h2>
  <span>Grouped by 3D lesion (L001…) · Withheld (&lt;1.0 mm² or wt 0) shown as — · ${m.n_slices - nCalcSl} other slices empty</span>
</div>
${tables}
<div class="bottom">
  <div class="card-box">
    <h3>Clinical Summary &amp; Audit Findings</h3>
    <ul class="findings">${m.conclusions.map(c => `<li>${esc(c)}</li>`).join('')}</ul>
  </div>
  <div class="card-box">
    <h3>Pipeline Provenance (from run.json)</h3>
    <div class="method">${m.params.map(([g, kv]) =>
      `<div><h4>${esc(g)}</h4><dl>${kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl></div>`).join('')}</div>
  </div>
</div>
<div class="disc">
  <span><b>PrediCT Studio</b> · Automated research output, not a diagnosis · 3.0 mm slices, not vessel-attributed</span>
  <span>Clinician review required · Figures read directly from run output</span>
</div>
</div></body></html>`;
  }

  // The lesion-wise table as CSV: one self-describing row per component (study and model on every row),
  // then a TOTAL row summing scored components. A study name is typed by users, so a leading = + - @
  // is defused (spreadsheet formula injection) and separators are quoted.
  function buildCsv(m) {
    const cell = v => {
      let s = v == null ? '' : String(v);
      if (/^[=+\-@\t\r]/.test(s) && typeof v === 'string') s = `'${s}`;
      return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const r2 = v => (v == null ? '' : Number(v).toFixed(2));
    const line = r => [m.study, m.model, r.idx, r.z_mm ?? '', r.lesion_3d_key, r.included,
      r2(r.area_mm2), r.peak_hu ?? '', r.weight ?? '', r.included ? r2(r.agatston) : ''].map(cell).join(',');
    const tot = [m.study, m.model, 'TOTAL', '', `${m.totals.n_included}/${m.totals.n_all}`, true,
      r2(m.totals.area_mm2), m.totals.peak_hu ?? '', m.totals.weight ?? '', r2(m.totals.agatston)].map(cell).join(',');
    return ['study,model,slice_idx,z_mm,lesion,included,area_mm2,peak_hu,density_weight,agatston',
      ...m.rows.map(line), tot].join('\r\n') + '\r\n';
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

  // The lesion-wise slice table as a CSV the browser downloads from memory (no server round trip).
  function exportCsv() {
    const model = openResultModel();
    if (!model) return;
    const name = `PrediCT_${model.study}_${model.model}_lesions.csv`.replace(/[^\w.-]+/g, '_');
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
