# PrediCT — Coronary Artery Calcium Scoring

**Automated, auditable Agatston scoring from cardiac CT.** A deep-learning
segmentation core, and **PrediCT Studio**: a multi-user clinical workstation
that shows the evidence behind every number it reports.

| | |
|---|---|
| **Programme** | Google Summer of Code 2026 @ ML4Sci |
| **This branch** | [`predict_software`](https://github.com/ML4Sci/PrediCT/tree/predict_software) — the workstation (PrediCT Studio) |
| **Algorithmic core** | [`soham_segmentation`](https://github.com/ML4Sci/PrediCT/tree/soham_segmentation) — model training and evaluation |
| **Dataset** | [Stanford AIMI COCA](https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa) (gated cardiac CT with radiologist calcium outlines) |
| **Full system design** | [`docs/SYSTEM.md`](docs/SYSTEM.md) — every workflow, diagram, API and security rule |

---

## Contents
1. [What is PrediCT?](#1-what-is-predict)
2. [Results](#2-results)
3. [Core pipeline](#3-core-pipeline)
4. [PrediCT Studio](#4-predict-studio)
5. [Architecture](#5-architecture)
6. [Models](#6-models)
7. [Quick start](#7-quick-start)
8. [Using the app](#8-using-the-app)
9. [Command line](#9-command-line)
10. [Accounts, privacy and security](#10-accounts-privacy-and-security)
11. [Project structure](#11-project-structure)
12. [Configuration](#12-configuration)
13. [Testing](#13-testing)
14. [Design principles](#14-design-principles)
15. [Documentation](#15-documentation)
16. [Roadmap](#16-roadmap)
17. [Acknowledgments](#17-acknowledgments)

---

## 1. What is PrediCT?

**Coronary artery calcium (CAC)** seen on a non-contrast, ECG-gated cardiac CT
is one of medicine's strongest predictors of a future heart attack. Doctors
summarise it as the **Agatston score**: for every calcified lesion on every
3 mm slice, *lesion area × a density weight from its brightest pixel*, summed
over the heart. The total places a patient in a risk category, and that
category drives treatment (for example, whether to start a statin).

```text
  Agatston score = Σ over lesions ( area in mm² × density weight )

  peak HU   130-199 → 1    200-299 → 2    300-399 → 3    ≥ 400 → 4
  risk      0 none · 1-100 mild · 101-400 moderate · > 400 severe   (6-tier view also available)
```

**The problem.** Scoring by hand is slow, and automated scoring usually trains
a *binary* segmentation network. Converting the radiologist's smooth outlines
into on/off pixels distorts small lesions, and on a tiny plaque a two-pixel
error can double its area and move a patient into the wrong risk category.

**What PrediCT does about it — two pillars:**

| Pillar | Idea | Where |
|---|---|---|
| **I · Algorithmic core** | *Approach 3 (soft coverage)*: each boundary voxel is trained on the exact fraction of it the radiologist's outline covers, and scored as `Σ probability × pixel area` instead of a voxel count. | `soham_segmentation` branch |
| **II · Clinical workstation** | *PrediCT Studio*: upload a scan, run a model, and audit the result: every lesion, every withheld speck, the model that produced it, and its settings. | this branch |

---

## 2. Results

From the GSoC 2026 evaluation (training, test protocol and full analysis on the
[`soham_segmentation`](https://github.com/ML4Sci/PrediCT/tree/soham_segmentation) branch);
held-out test set *n* = 66, replication cohort *n* = 374.

| Measure | A1 binary | A3 soft coverage |
|---|---|---|
| Area error of the training labels vs the radiologist's outline | 10.19 % | **0.03 %** |
| Median absolute Agatston error (test) | 42.97 | **19.27** (2.2× lower) |
| 6-tier risk agreement (test, *n* = 66) | 77.3 % | **83.3 %** |
| Risk agreement (replication, *n* = 374) | 70.7 % | **76.7 %** (McNemar *p* = 0.038) |
| Validation Dice of the shipped checkpoint | 0.652 | **0.723** |

---

## 3. Core pipeline

What happens to one scan, from file to reviewed result (`src/backend/run.py`):

```text
  DICOM folder or NIfTI file
  (uploaded in PrediCT Studio, or a server path for the admin)
                               │
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ 1  LOAD          DICOM (GDCM) or NIfTI → 3D image      │
  └────────────────────────────┬───────────────────────────┘
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2  RESAMPLE      0.37 × 0.37 × 3.0 mm voxels           │
  └────────────────────────────┬───────────────────────────┘
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3  HEART CROP    TotalSegmentator heart + 8 mm         │  ──► cache work/<study>/crop|full
  │                  on/off per run (default: model's)     │
  └────────────────────────────┬───────────────────────────┘
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ 4  ORIENT+CHECK  reorient to RAS                       │
  │                  assert spacing and axes, or stop      │
  └────────────────────────────┬───────────────────────────┘
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ 5  SEGMENT       UNet: A1 binary or A3 coverage        │  ──► pred.nii.gz
  │                  or nnUNet → probability per voxel     │
  └────────────────────────────┬───────────────────────────┘
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ 6  SCORE         per slice: area × density weight      │  ──► lesions.csv
  │                  binary   area = voxels × pixel area   │
  │                  coverage area = Σ prob × pixel area   │
  │                  < 1.0 mm² → withheld, still listed    │
  └────────────────────────────┬───────────────────────────┘
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ 7  GROUP 3D      link slices into 3D lesions           │  ──► lesions_3d.csv
  │                  (describes; never changes a score)    │
  └────────────────────────────┬───────────────────────────┘
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ 8  VERIFY        3D total == per-slice total, or stop  │
  └────────────────────────────┬───────────────────────────┘
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ 9  OUTPUT        meshes · PNG slices · provenance      │  ──► mesh/ · slices/ · run.json
  └────────────────────────────────────────────────────────┘
                               │
                               ▼
  PrediCT Studio:  Argument · Instrument · Contact Sheet · Anatomy
```

* **The A1 / A3 difference is one line** in `scoring.py` (step 6); everything else is shared.
* **Nothing is silent.** Wrong spacing, wrong axes, or a 3D total that disagrees with the per-slice total stop the run instead of producing a plausible-looking wrong number.
* **Every result records what produced it** (`run.json`: model, checkpoint SHA-256, threshold, crop, HU window, spacing, date).
* Each step in detail, with branches: [`docs/SYSTEM.md` §9](docs/SYSTEM.md#9-inference-pipeline).

---

## 4. PrediCT Studio

```text
  ┌────────────────┐      ┌───────────────┐      ┌──────────────┐      ┌────────────────┐
  │ Sign in        │  ─►  │ Upload a scan │  ─►  │ Run          │  ─►  │ Review         │
  │ (access token) │      │ DICOM / NIfTI │      │ model + crop │      │ 4 views, audit │
  └────────────────┘      └───────────────┘      └──────────────┘      └────────────────┘
```

| View | What you see |
|---|---|
| **01 Argument** | Total score and risk tier, plain-language findings, the key slice with every calcium slice beside it, Pipeline Parameters, the 3D lesion index and the withheld (< 1 mm²) ledger |
| **02 Instrument** | A slice-by-slice calcium track and a large viewer with **zoom and pan** (Ctrl+wheel or pinch), a 10 mm scale bar and per-lesion details (area, voxels, peak HU, weight, mean coverage) |
| **03 Contact Sheet** | Every slice at once (adjustable tile size), marked scored / sub-minimum / empty, with the counted components |
| **04 Anatomy** | 3D surfaces of the calcium inside a heart shell, a slice plane that follows the 2D view, click-to-select. Clearly labelled *not vessel-attributed*. |

**Workstation features:** accounts with private data per user · drag-and-drop
upload with real progress · searchable scan picker (newest first) · per-run
heart-crop toggle · runs continue on the server while you browse · 4-tier /
6-tier risk view · resizable, collapsible panels · delete results or whole
studies · keyboard navigation · **one-page report** (icon on the Argument view → print or Save as PDF: score, key-slice images,
slice-by-slice table, conclusions, method) and **CSV of the slice table**, both built in the
browser from the open result; nothing is written to the server.

---

## 5. Architecture

```text
  ┌────────────────────┐   cookie   ┌────────────────────┐ subprocess ┌────────────────────┐
  │ Browser            │ ─────────► │ FastAPI server     │ ─────────► │ Inference run.py   │
  │ HTML · JS · 3D     │ ◄───────── │ accounts · files   │            │ pipeline · scoring │
  └────────────────────┘ JSON+files │ jobs · limits      │            │ mesh · render      │
                                    └──────────┬─────────┘            └──────────┬─────────┘
                                               ▼                                 ▼
                                    ┌────────────────────┐            ┌────────────────────┐
                                    │ data/accounts.db   │            │ data/users/<id>/   │
                                    │ users · sessions   │            │ raw·work·out·tmp   │
                                    └────────────────────┘            └────────────────────┘
```

* The **browser** only displays; it never computes a score.
* The **server** signs users in, checks every request belongs to the signed-in user, serves their files, and starts runs, one at a time.
* **Inference** runs as a separate process per run, so a crashed run cannot take the server down.
* All of a user's data lives in their own folder; nothing under `data/` is ever public.

---

## 6. Models

Every model folder has a `manifest.yaml` declaring the facts its weights cannot
hold (spacing, HU window, threshold, crop, output type). `registry.py` checks
the weights against the SHA-256 in the manifest before every run.

| Model id | Approach | Output | Threshold | Crop by default | Weights |
|---|---|---|---|---|---|
| `a1-roi` | A1 binary, heart region | binary | 0.50 | on | in the repo |
| `a3-coverage-v2` | A3 soft coverage, heart region | coverage | 0.10 (delineation only) | on | in the repo |
| `nnunet_dob_scv` | nnUNet (DOB SCV) | binary | 0.50 | off | not in git — add `fold_0/checkpoint_final.pth` |
| `nnunet_simple_scv` | nnUNet (Simple SCV) | binary | 0.50 | off | not in git — add `fold_0/checkpoint_final.pth` |

Crop can be switched on or off for any model before a run; a result run with a
non-default crop is flagged in its Pipeline Parameters.

---

## 7. Quick start

**Requirements:** Python 3.11+, an NVIDIA GPU with CUDA (strongly recommended:
TotalSegmentator and nnUNet are slow on CPU), and a modern browser.
`requirements.txt` is a full freeze of the Linux GPU environment (CUDA 13
wheels included). On another platform, install the core packages instead:
`torch monai nnunetv2 TotalSegmentator SimpleITK python-gdcm pydicom nibabel scikit-image scipy numpy PyYAML Pillow fastapi uvicorn python-multipart`.

```bash
git clone https://github.com/ML4Sci/PrediCT.git
cd PrediCT && git checkout predict_software
cd Predict-Studio
pip install -r requirements.txt

# 1. access tokens you hand out (comma-separated)      Windows cmd: set PREDICT_ACCESS_TOKENS=tok1,tok2
export PREDICT_ACCESS_TOKENS="$(python -c 'import secrets; print(secrets.token_urlsafe(16))')"

# 2. the admin account (password is prompted, stored only as a salted hash)
python -m src.backend.accounts create-admin you@example.com

# 3. start
python -m src.backend.server
```

Open **http://127.0.0.1:8001**, sign in as the admin, and share a token with
each person who should create an account.

> **Upgrading from a version before accounts?** Move the old shared data into
> one account once: `python -m src.backend.accounts adopt-legacy <username>`.

---

## 8. Using the app

1. **Sign in** — username, password and the access token you were given (or *Create account*).
2. **Upload** — the upload icon or drag a folder onto the page. Name the study (optional). The upload card shows progress, then checks the files; non-DICOM files can be removed in one click. The new scan is selected automatically.
3. **Run** — pick a model, keep or change *Crop to heart*, press **Run**. The run happens on the server: keep browsing or reload, and a message with **Open** appears when it finishes.
4. **Review** — the four views. Click a lesion anywhere to follow it across views.
5. **Manage** — filter the study list; the bin icon deletes chosen results or the whole study.

| Keys | Action |
|---|---|
| ← → / ↑ ↓ | previous / next slice (↑ ↓ move a row in the Contact Sheet) |
| Home / End | first / last slice with calcium |
| 1 · 2 · 3 | original · prediction · calcium-only view |
| Esc | clear the selection |
| Ctrl + wheel · + / − / 0 | zoom in the Instrument view · reset |

---

## 9. Command line

```bash
# run a model for an account (ids: python -m src.backend.accounts list)
python -m src.backend.run --user 1 --study <uploaded scan name> --model a3-coverage-v2
python -m src.backend.run --user 1 --input /path/to/dicom/folder --model a1-roi [--crop | --no-crop]

# accounts
python -m src.backend.accounts list
python -m src.backend.accounts reset-password <username>
python -m src.backend.accounts delete <username>
```

Results are written to `data/users/<id>/out/<study>/<model>/`
(`run.json`, `lesions.csv`, `lesions_3d.csv`, `pred.nii.gz`, `ct.nii.gz`, `slices/`, `mesh/`).

---

## 10. Accounts, privacy and security

* **Private by construction:** each account has its own folder; the server takes the user from the session cookie, never from a link, so a shared link cannot open someone else's data.
* **Invite-only:** creating an account and signing in need an access token from the administrator; removing a token (and restarting) signs out everyone who used it. The admin needs no token.
* **Protected sign-in:** salted scrypt password hashes, HttpOnly + SameSite=Strict session cookie, identical errors for every failure, and token-bucket rate limits on sign-ups, failed sign-ins and uploads.
* **Bounded resources:** 4 GB per upload, 100 GB per account, one run at a time on the GPU.
* **Not yet production auth** — see the [roadmap](#16-roadmap). Full threat table: [`docs/SYSTEM.md` §13](docs/SYSTEM.md#13-security-model).

---

## 11. Project structure

```text
Predict-Studio/
├── src/backend/
│   ├── server.py           web server: API, sign-in, per-user access, job runner
│   ├── accounts.py         accounts, passwords, sessions, access tokens (+ CLI)
│   ├── ratelimit.py        token-bucket limits
│   ├── paths.py            every folder path (per account), safe names
│   ├── run.py              runs the pipeline for one scan; writes run.json (+ CLI)
│   ├── pipeline.py         load · resample · heart crop · normalise · UNet inference
│   ├── pipeline_nnunet.py  nnUNet inference
│   ├── scoring.py          Agatston maths (the A1 / A3 difference lives here)
│   ├── grouping.py         per-slice components → 3D lesions (descriptive only)
│   ├── render.py           CT and mask PNG slices
│   ├── mesh.py             3D surfaces for the Anatomy view
│   ├── registry.py         manifest validation, weights SHA-256 lock
│   └── ingest.py           fixes DICOM files without an extension
├── ui/                     PrediCT Studio: index.html · app.js · app.css · view3d.js · report.js
├── models/<model id>/      manifest.yaml + weights
├── tests/                  pytest suites
├── docs/                   SYSTEM.md (system design) · Principles.md (code rules) · figures
├── data/                   accounts.db + users/<id>/{raw,work,out,tmp}  (git-ignored)
└── requirements.txt
```

---

## 12. Configuration

| Setting | Where | Default |
|---|---|---|
| Access tokens | environment variable `PREDICT_ACCESS_TOKENS` | none → only the admin can sign in |
| Session length | `accounts.SESSION_DAYS` | 7 days |
| Accounts per token | `accounts.MAX_ACCOUNTS_PER_TOKEN` | 50 |
| Rate limits | constants in `ratelimit.py` | sign-ups 20 / 3 min · failed sign-ins 5 / min per user · uploads 10 / 2 min |
| Upload size · storage | `server.MAX_UPLOAD_BYTES` · `server.USER_QUOTA_BYTES` | 4 GB · 100 GB |
| Port | `server.py` | 8001 |
| Model behaviour | `models/<id>/manifest.yaml` | per model |

---

## 13. Testing

```bash
python -m pytest tests -v
```

| Suite | Covers |
|---|---|
| `test_accounts.py` | sign-up / sign-in rules, admin exemption, token revocation, expiry, hashing, limits |
| `test_ratelimit.py` | token bucket: drain, refill, memory bound |
| `test_paths.py` | safe names, per-account folders |
| `test_grouping.py` | 3D linking never changes a score |
| `test_mesh.py` | mesh geometry and orientation (needs scikit-image) |
| `test_report.py` | one-page report numbers match `run.json` (needs Node; skipped without it) |

---

## 14. Design principles

From [`docs/Principles.md`](docs/Principles.md) — the rules every change follows:

* **A wrong number that looks reasonable is the worst failure** → fail loudly; no silent defaults or fallbacks.
* **Declared, not guessed:** what the weights cannot say lives in `manifest.yaml`, locked by SHA-256.
* **Provenance:** every output says what produced it (`run.json`).
* **Units in names, axis order in docstrings** (`area_mm2`, `(Z, Y, X)`).
* **One-way layering** (server → run → pipeline / scoring / render → paths) and small files.
* **Users are kept apart by construction:** the user comes from the session, every path from that user's id.

---

## 15. Documentation

| Document | For |
|---|---|
| [`docs/SYSTEM.md`](docs/SYSTEM.md) | how the whole system works: components, data layout, sign-in, upload, runs, pipeline, UI, API, security, operations |
| [`docs/Principles.md`](docs/Principles.md) | how code in this repository is written |
| [`soham_segmentation` branch](https://github.com/ML4Sci/PrediCT/tree/soham_segmentation) | model training, evaluation and the full results |
| `docs/Explanations/` | pipeline figures |

---

## 16. Roadmap

* **Research:** re-score the test split with `A3 Coverage v2` and matching `[0, 1200]` HU normalisation; benchmark coverage training against nnU-Net and hybrid architectures; coronary centreline integration for per-vessel scores (LM, LAD, LCx, RCA).
* **Workstation:** sign-in through an identity provider (OpenID Connect), per-person invites with admin approval, roles, two-factor sign-in, HTTPS; a run queue and cancel button; audit log of who opened which study.

---

## 17. Acknowledgments

Built during **Google Summer of Code 2026** with **ML4Sci**. Thanks to mentors
**Katy Butler** and **Anna** for their guidance, to co-contributor **Rajat** for
dataset analysis and cohort validation, and to Stanford AIMI for the COCA dataset.
