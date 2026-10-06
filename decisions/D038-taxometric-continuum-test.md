# D038 Taxometric test of dimensional vs categorical structure

**Date**: 2026-10-03
**Stage**: discover (validation of Finding 1)
**Status**: EXPERIMENTAL, CLOSED (2026-10-03): not usable. The revision-2 stopping rule rejected the method (continua read dimensional in 9 of 12 runs; strongly skewed latent dimensions fail). The test was stopped and never run on the real data. Reported as a methods result: a skewed latent dimension confounds standard taxometrics (copula comparison data read skewed continua as categorical) and the corrected model is valid only up to moderate skew.
**Module**: `src/predict/discover/taxometric.py`, `scripts/probe_taxometric.py`, `submit_taxometric.sh`

## Why

D036 and D036.1 showed that the gap statistic (kmeans, ward and GMM) cannot detect separated groups in this geometry and returns about k = 11 on the real data whatever is planted, so it cannot support "no discrete phenotypes". The literature describes calcium burden as continuous; that may motivate a test only through a new decision written before looking. D038 is that test. It compares the two hypotheses directly and can return either answer.

## Method

Taxometric comparison curves (Ruscio and colleagues). Observed MAMBAC, MAXEIG and L-Mode curves are compared with the mean curves of simulated comparison data that reproduce the observed marginals and correlations under a dimensional model (one population) and a categorical model (two groups from k-means k=2, each regenerated with its own marginals and correlations). CCFI = RMSR_dim / (RMSR_dim + RMSR_cat) per procedure; the decision uses the mean of the three.

Implementation choices (simplifications of the published procedures, fixed here):
- comparison data: intermediate normal correlation calibrated on a 10,000-row population (marginals bootstrapped from the data, 20 iterations), then one sample of N rows rank-matched to the data's own sorted columns;
- MAMBAC: one indicator as output, the sum of the others as input, 50 cuts with 25 cases kept at each end;
- MAXEIG: one indicator as input, 50 windows with 90% overlap, largest eigenvalue of the zero-diagonal covariance of the other indicators;
- L-Mode: density of the first principal component score (z-scored) on 50 grid points from -2.5 to 2.5;
- curves averaged over all input / output configurations.

### Revision 1 (2026-10-03, after the first unit-test run, before any real-data run)

The pre-registered unit test "a skewed single-factor continuum reads dimensional" FAILED with the Ruscio-style comparison data: mean CCFI 0.72 (MAMBAC 0.81, MAXEIG 0.63, L-Mode 0.73), categorical, k-means base rate 0.17. The normal copula reproduces each indicator's skew but not the co-movement of indicators that share one skewed latent dimension; the long tail then resembles a small taxon. Calcium burden is such a dimension, so a categorical verdict from the original method would be uninterpretable here.

Change: the dimensional comparison model becomes a single latent dimension of any unimodal shape. The leading factor of the (calibrated) intermediate correlation matrix carries scores from a Johnson SU distribution fitted to the data's standardised first principal-component scores; the remaining correlation is normal; marginals are rank-matched exactly. A Johnson SU is unimodal, so two separated groups (a bimodal leading axis) are not absorbed; the positive control checks this. The copula model is kept selectable (`dimensional_model="copula"`) for the diagnostic `scripts/diagnose_taxometric.py`, which runs both models on a normal continuum, skewed continua (lognormal sigma 0.4, 0.6, 0.9) and a two-group taxon. Added unit tests: the taxon reads categorical under both models; a normal continuum reads dimensional under both; the skewed continuum reads dimensional under the skewed-factor model. Everything else in D038 (indicators, controls, thresholds, interpretation) is unchanged.

### Revision 2 (2026-10-03, after the diagnostic, before any real-data run)

`scripts/diagnose_taxometric.py` (3 seeds, 20 comparison datasets) confirmed the cause is skew, not a bug: the copula model read the normal continuum dimensional (0.25 to 0.28) and the lognormal continua categorical, increasingly with skew (sigma 0.4: 0.49 to 0.57; 0.6: 0.61 to 0.70; 0.9: 0.72 to 0.81). Revision 1 fixed sigma 0.4 and 0.6 (0.33 to 0.44, dimensional) but not 0.9 (0.55 to 0.60). Cause: the Johnson SU was fitted to first principal-component scores, which are the latent plus noise, so the fitted skew was diluted, more so at high skew; and PCA loadings overstate each indicator's share of the latent.

Change: a one-factor model (maximum likelihood factor analysis) gives loadings and uniquenesses; Bartlett scores equal the latent plus normal noise of known variance, so the latent's skewness and excess kurtosis are recovered from the scores' third and fourth cumulants with the noise-corrected variance; a Johnson SU is matched to those two moments (unimodal and leptokurtic; a platykurtic two-group latent cannot be matched and falls back toward normal). Comparison data are lambda * f + E, E normal with covariance R_int - lambda lambda', R_int calibrated as before, marginals rank-matched. New unit test: the estimate recovers the skewness 2.26 of a lognormal(0, 0.6) latent within 0.5 (n = 20,000).

**Stopping rule (fixed before the revision-2 diagnostic):** revision 2 is accepted only if, under the skewed-factor model, the continua (normal and lognormal sigma 0.4, 0.6, 0.9; 3 seeds each) read dimensional in at least 11 of 12 runs and the taxon reads categorical in 3 of 3. Otherwise the taxometric test is declared not usable for strongly skewed continua and no further method changes are made; that is reported.

## Indicators: equal representation across feature families

From the D033 stage 5 outputs (`outputs/06_reduce/`), within each of the six D022 blocks (burden, hu_statistics, density_tier, spatial, texture, shape): among the block's representatives with at least 10 distinct values in the prepared matrix (taxometric procedures need near-continuous indicators), rank by redundancy-cluster size (descending; the representative of the largest cluster carries the block's most shared signal), then ICC (descending), then name, and take the top 2. A block with fewer eligible representatives contributes what it has. At most 12 indicators. The rule uses only D022 outputs, not any clustering or taxometric result; the chosen list is written to `indicator_selection.csv` before the curves are computed.

## Controls (must pass before the real data are read)

Planted groups use the D036 helpers in the full representative matrix (random direction), then the same indicator subset. Grid: k_true 2 and 3; delta 0, 4, 8, 12, 16 background SDs; balanced and imbalanced; gaussian (N(0, real covariance)) and real backgrounds; 5 replicates; 50 comparison datasets per model.

- **Negative control**: gaussian background, delta 0: verdict dimensional in at least 80% of runs.
- **Positive control**: gaussian background, k_true 2, balanced: verdict categorical in at least 80% of runs at some delta <= 16 (that delta is the taxometric floor).
- Reported, not gating: the real-background floors (the operative detection limit on this manifold) and k_true 3.

Real data: 100 comparison datasets per model, seed 38.

## Decision thresholds (fixed)

Mean CCFI <= 0.45 dimensional; >= 0.55 categorical; otherwise ambiguous (the conventional bands).

## Interpretation (fixed before running)

| Controls | Real mean CCFI | Reading |
|---|---|---|
| both pass | <= 0.45 | Positive evidence for a dimensional structure: "a taxometric comparison that detects planted two-group taxa from delta* (real background floor) favours a continuum (CCFI x)". Finding 1 can be stated as dimensional, with that floor and the D036.1 gap result alongside. |
| both pass | >= 0.55 | Evidence for a categorical structure. Contradicts Finding 1 as worded; reported as such and investigated (base rate, which indicators drive it). |
| both pass | ambiguous | No conclusion; reported with the CCFI values. |
| either fails | any | The test is not usable on this geometry; the real value is reported but not interpreted. |

## Limitations (stated in the paper)

- Classic taxometrics compares one dimension with two groups; k_true 3 is reported from the controls only.
- Planting happens after gate, ComBat and redundancy selection (as in D036).
- The comparison-data generator and procedure settings are simplifications of the published ones (above).
- The categorical comparison model uses k-means assignment; a poor assignment biases CCFI toward dimensional, which the positive control checks.

## Verified by

- `tests/discover/test_taxometric.py`: marginals preserved exactly and correlations within RMSR 0.06; curve shapes; thresholds; a clear two-group taxon reads categorical; a skewed single-factor continuum reads dimensional.
- Smoke run (`--smoke`) before the full job.
