// Checks ui/report.js (the one-page report) in Node — it needs no browser.
// Run by tests/test_report.py, or directly:  node tests/report_check.mjs
//
// 1. A small hand-made result: every number in the report model is checked.
// 2. Every real result under data/users/*/out and data/out (if any): the report's
//    total must equal run.json, with no "numbers disagree" warning.
// Does NOT write anything.
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';

const { buildReportModel, renderReportHtml, buildCsv } = createRequire(import.meta.url)('../ui/report.js');
const TIERS = {   // copy of app.js TIER_SCHEMES; app.js is a browser script and cannot be imported
  4: [[0, 'ZERO'], [100, 'MILD'], [400, 'MODERATE'], [Infinity, 'SEVERE']],
  6: [[0, 'ZERO'], [100, 'MILD'], [300, 'MODERATE'], [400, 'MOD-HIGH'], [1000, 'SEVERE'], [Infinity, 'EXTENSIVE']],
};

function parseCsv(text) {   // same rule as app.js parseCsv
  const lines = text.trim().split(/\r?\n/), head = lines[0].split(',');
  return lines.slice(1).filter(Boolean).map(line => {
    const cells = line.split(','), o = {};
    head.forEach((h, i) => {
      const v = (cells[i] ?? '').trim();
      o[h] = v === 'True' ? true : v === 'False' ? false : (v !== '' && !isNaN(v)) ? Number(v) : v;
    });
    return o;
  });
}
const build = (run, slices, lesions, lesions3d) =>
  buildReportModel({ run, slices, lesions, lesions3d, tierSchemes: TIERS, params: [], study: 's', user: 'u', now: '2026-01-01T00:00:00' });

// ── 1. synthetic result ──
{
  const les = (slice_idx, lesion_id, area_mm2, peak_hu, density_weight, included, id3d, key) =>
    ({ slice_idx, z_mm: slice_idx * 3, lesion_id, area_mm2, peak_hu, density_weight, included, agatston: included ? area_mm2 * density_weight : 0, lesion_3d_id: id3d, lesion_3d_key: key });
  const lesions = [les(1, 1, 4, 250, 2, true, 1, 'L001'), les(1, 2, 2, 410, 4, true, 2, 'L002'),
    les(2, 3, 5, 300, 3, true, 1, 'L001'), les(2, 4, 0.5, 200, 2, false, 3, 'L003')];
  const slices = [0, 1, 2, 3].map(i => ({ idx: i, z_mm: i * 3, has_calcium: i === 1 || i === 2,
    slice_score: i === 1 ? 16 : i === 2 ? 15 : 0 }));
  const l3 = [{ lesion_3d_key: 'L001', included: true, n_slices: 2, total_agatston: 23 },
    { lesion_3d_key: 'L002', included: true, n_slices: 1, total_agatston: 8 }];
  const run = { agatston_total: 31, risk_category: 'mild', model_id: 'm', n_lesions: 3, output: 'coverage', min_area_mm2: 1 };
  const m = build(run, slices, lesions, l3);
  assert.deepEqual(m.warnings, []);
  assert.equal(m.rows.length, 4);               // all 4 lesion components (3 scored + 1 withheld)
  assert.deepEqual(m.rows.map(r => `${r.lesion_3d_key}:${r.idx}`), ['L001:1', 'L001:2', 'L002:1', 'L003:2']);
  assert.equal(m.rows[0].first_in_3d, true);
  assert.equal(m.rows[1].first_in_3d, false);
  assert.equal(m.rows[3].included, false);
  assert.equal(m.totals.agatston, 31);
  assert.equal(m.totals.n_included, 3);
  assert.equal(m.totals.n_all, 4);
  assert.equal(m.totals.peak_hu, 410);
  assert.equal(m.withheld.n, 1);
  assert.equal(m.tier6, 'MILD');
  assert.deepEqual(m.key_slices, [1, 2]);       // both calcium slices, in slice order
  const html = renderReportHtml(m, { aspect: '1', slices: m.key_slices.map(i => ({ idx: i, ct: 'a.png', mask: 'b.png' })) });
  assert.equal(html.split('class="key"').length - 1, 2);
  assert.ok(html.includes('PrediCT Studio') && html.includes('31.0') && html.includes('L003'));

  // CSV: header + one row per lesion component + TOTAL, total equals the report total
  const csv = buildCsv(m).trim().split('\r\n');
  assert.equal(csv.length, 1 + m.rows.length + 1);
  assert.equal(csv[0].split(',').length, 10);
  assert.ok(csv[csv.length - 1].startsWith('s,m,TOTAL,,3/4,true,'));
  assert.ok(csv[csv.length - 1].endsWith(',31.00'));
  // a study name that is a spreadsheet formula, or holds a comma or quote, is defused / quoted
  const fm = buildReportModel({ run, slices, lesions, lesions3d: l3, tierSchemes: TIERS, params: [],
    study: '=cmd|x, "y"', user: 'u', now: '2026-01-01T00:00:00' });
  assert.ok(buildCsv(fm).split('\r\n')[1].startsWith('"\'=cmd|x, ""y""",m,1,'));

  // numbers that disagree must be reported, never hidden
  const bad = build({ ...run, agatston_total: 99 }, slices, lesions, l3);
  assert.ok(bad.warnings.some(w => w.includes('run.json total')));
  assert.ok(renderReportHtml(bad, null).includes('numbers disagree'));

  // zero calcium
  const zero = build({ agatston_total: 0, risk_category: 'zero', model_id: 'm', n_lesions: 0 },
    slices.map(s => ({ ...s, has_calcium: false, slice_score: 0 })), [], []);
  assert.equal(zero.rows.length, 0);
  assert.ok(zero.conclusions[0].startsWith('No coronary calcium was scored'));
  assert.doesNotThrow(() => renderReportHtml(zero, null));

  // user-typed study names are never parsed as HTML
  const evil = buildReportModel({ run, slices, lesions, lesions3d: l3, tierSchemes: TIERS, params: [],
    study: '<img src=x onerror=alert(1)>', user: 'u', now: '2026-01-01T00:00:00' });
  assert.ok(!renderReportHtml(evil, null).includes('<img src=x'));

  // no total in run.json: refuse
  assert.throws(() => build({ model_id: 'm' }, slices, lesions, l3), /agatston_total/);
}
console.log('synthetic: ok');

// ── 2. real results ──
const roots = [];
const users = path.resolve('data/users');
if (fs.existsSync(users)) for (const u of fs.readdirSync(users)) roots.push(path.join(users, u, 'out'));
roots.push(path.resolve('data/out'));
let n = 0;
for (const root of roots.filter(fs.existsSync))
  for (const study of fs.readdirSync(root))
    for (const model of fs.readdirSync(path.join(root, study))) {
      const d = path.join(root, study, model), f = x => path.join(d, x);
      if (!['run.json', 'slices.json', 'lesions.csv', 'lesions_3d.csv'].every(x => fs.existsSync(f(x)))) continue;
      const run = JSON.parse(fs.readFileSync(f('run.json'), 'utf8'));
      const m = build(run, JSON.parse(fs.readFileSync(f('slices.json'), 'utf8')),
        parseCsv(fs.readFileSync(f('lesions.csv'), 'utf8')), parseCsv(fs.readFileSync(f('lesions_3d.csv'), 'utf8')));
      assert.deepEqual(m.warnings, [], `${study}/${model}: ${m.warnings}`);
      assert.ok(Math.abs(m.totals.agatston - run.agatston_total) < 0.05, `${study}/${model} total`);
      assert.ok(renderReportHtml(m, null).length > 1000);
      console.log(`  ${study}/${model}: total ${run.agatston_total.toFixed(1)}, ${m.rows.length} slice rows`);
      n++;
    }
console.log(`real results checked: ${n}`);
