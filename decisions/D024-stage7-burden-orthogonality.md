# D024 Stage 7 burden orthogonality protocol

**Date**: 2026-06-06
**Stage**: analyse (stage 7)
**Status**: Active
**Module**: `src/predict/analyse/orthogonality.py`

## Decision

Test whether the spatial-only k=2 phenotype (focal vs diffuse) is independent of total calcium burden via a three-statistic protocol on `agatston_total`:

| Statistic | Tests for | Library |
|---|---|---|
| Mann-Whitney U p-value | location (median) difference between focal and diffuse burden distributions | `scipy.stats.mannwhitneyu`, two-sided |
| Levene's test p-value | dispersion (variance) difference between focal and diffuse burden distributions | `scipy.stats.levene` with `center='median'` |
| Cliff's delta on agatston_total | rank-based effect size for the burden difference | in-house, same as D023 |

Both the p-value and the effect size are written to `burden_orthogonality.csv` regardless of pass / fail, with a 3-level `interpretation` column:

```
orthogonal   = Mann-Whitney p > 0.05 AND |Cliff's delta| < 0.20
                 (no detectable median shift AND no clinically meaningful effect)
marginal     = exactly one arm crosses; PASS but worth noting:
               (a) p < 0.05 AND |delta| < 0.20    => significant but tiny effect
                   (the cohort is large; even trivial differences are detectable)
               (b) p >= 0.05 AND |delta| >= 0.20  => underpowered but visible effect
                   (sub-cohort is small; lack of significance is a power issue)
confounded   = p < 0.05 AND |delta| >= 0.20
                 (clear burden difference; phenotype is NOT orthogonal to burden)
```

PASS criterion (binary `passes` column): `interpretation in {orthogonal, marginal}`.
FAIL criterion: `interpretation == confounded`.

Additionally, burden-stratified spatial profiles are computed: within each Agatston tertile (low / mid / high, defined per cohort with `pd.qcut`), recompute the directional-hypothesis test (D026 six hypotheses) on the focal vs diffuse comparison restricted to that tertile.

This is the strongest possible orthogonality evidence: if the spatial phenotype distinguishes focal from diffuse even *within* a burden stratum (where burden is held approximately constant by construction), the phenotype cannot be a burden artefact.

PASS criterion for burden-stratified replication: at least 4 of 6 directional hypotheses hold in the same predicted direction in at least 2 of 3 burden tertiles.

## Rationale

### Why both Mann-Whitney AND Levene

Mann-Whitney tests whether two distributions have different medians (location). Levene tests whether they have different spreads (scale). Two distributions can have identical medians but very different variances (one tight, one spread); a phenotype that selectively expands burden variance is still a burden-confounded phenotype.

For full orthogonality we need both: (a) median agatston the same in focal vs diffuse, AND (b) variance of agatston the same. Either alone is necessary but not sufficient. We report both p-values; the `interpretation` column above only thresholds on the Mann-Whitney p, but the Levene p appears as a separate column for the paper to comment on.

### Why a 3-level interpretation instead of binary pass/fail

A binary "PASS / FAIL" treats two very different scientific situations identically:

- **p = 0.04, delta = 0.08** in a large cohort (N=420): the test detects a real-but-trivial median shift. The phenotype is empirically orthogonal at the clinical scale even though formally significant. Binary "PASS" hides this nuance.
- **p = 0.06, delta = 0.25** in a small cohort (N=200): the test fails to reach significance because of limited power, but the effect size is moderate. Binary "PASS" misleads the reader into thinking orthogonality is established.

The 3-level column lets the paper sentence read e.g. "burden orthogonality: full cohort orthogonal; Qr36d/2 stratum marginal (underpowered with visible effect); I30f/3 stratum orthogonal" rather than "all three pass".

### Why qcut tertiles instead of agatston-category breakpoints

The Agatston category breakpoints (1-99 / 100-399 / >=400) are clinically conventional but unbalanced in this cohort (the 400+ tier dominates). `pd.qcut(q=3)` produces equal-sized strata, which is what we need to test orthogonality with comparable statistical power per stratum.

## Alternatives considered

- **Skip burden-stratified replication**: rejected; it's the strongest orthogonality argument and replaces a substantially weaker single-cohort claim.
- **t-test on log(agatston + 1) instead of Mann-Whitney**: rejected; even after log transformation Agatston is heavy-tailed, and we want the test to match the rank-based effect size.
- **Use a single composite pass/fail instead of 3-level interpretation**: rejected; nuance must be visible.
- **Use the gap-statistic-selected k for the burden partition instead of qcut tertiles**: rejected; the gap statistic does not support discrete k for burden, and qcut is a more conservative choice that does not impose structure.

## Verified by

- `tests/analyse/test_orthogonality.py` covering: 3-level interpretation rule on synthetic (orthogonal, marginal-significant-but-small, marginal-large-but-underpowered, confounded) cases; Levene's test runs on heteroscedastic synthetic; burden-stratified replication produces a (tertile x hypothesis) table.
