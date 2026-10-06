# Test fixtures

Small data fixtures used by unit/integration tests. Not committed (large, PHI-adjacent).

## Expected contents

```
fixtures/
├── synthetic/
│   ├── patient_synth_01/        # synthetic DICOM series + XML (generated)
│   └── patient_synth_02/
└── real/
    ├── patient_001/              # one real COCA patient (single-series)
    └── patient_078/              # one real COCA patient (multi-series, same-Z)
```

## Generation

- Synthetic fixtures: a small helper script (to be written) produces a 30-slice 64×64 CT volume with one known calcium lesion and a matching plist XML. Used for: rasterisation correctness, mask building, Agatston math, perturbation pipeline.
- Real fixtures: copied verbatim from `data/raw/{pid}` and `data/calcium_xml/{pid}.xml`. Used for: end-to-end integration, multi-series handling.

Mark integration tests with `@pytest.mark.integration` so they can be deselected on CI.
