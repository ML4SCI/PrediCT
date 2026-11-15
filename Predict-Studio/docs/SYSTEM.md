# PrediCT Studio — System Design

One document to understand the whole application: what runs where, how data
flows, how users are kept apart, and how a CT scan becomes an Agatston score.
Diagrams are plain text (boxes and arrows), so they read the same in any editor,
viewer or terminal.

| | |
|---|---|
| **What it does** | Computes the coronary artery calcium (Agatston) score from a cardiac CT and lets a clinician audit every number. |
| **Stack** | Python 3.11+ · FastAPI · PyTorch/MONAI · nnUNet · TotalSegmentator · SimpleITK · plain HTML/CSS/JS · Three.js · SQLite |
| **Start** | `python -m src.backend.server` from `Predict-Studio/` → `http://127.0.0.1:8001` |
| **Principles** | See [`Principles.md`](Principles.md): fail loudly, no silent defaults, every output says what produced it. |

---

## Contents
1. [System overview](#1-system-overview)
2. [Components](#2-components)
3. [Data layout](#3-data-layout)
4. [Accounts and sign-in](#4-accounts-and-sign-in)
5. [Request authorisation (the direct-link rule)](#5-request-authorisation-the-direct-link-rule)
6. [Abuse limits (token buckets)](#6-abuse-limits-token-buckets)
7. [Upload workflow](#7-upload-workflow)
8. [Run workflow (jobs)](#8-run-workflow-jobs)
9. [Inference pipeline](#9-inference-pipeline)
10. [Result files](#10-result-files)
11. [User interface](#11-user-interface)
12. [API reference](#12-api-reference)
13. [Security model](#13-security-model)
14. [Configuration and operations](#14-configuration-and-operations)
15. [Testing](#15-testing)
16. [Known limits and next steps](#16-known-limits-and-next-steps)

---

## 1. System overview

```text
┌──────────────────────────────── BROWSER ──────────────────────────────────┐
│  index.html + app.js                         view3d.js                    │
│  one state · one render() · api() calls      Three.js 3D viewer           │
└──────────────────┬──────────────────────────────────┬─────────────────────┘
                   │ JSON + session cookie            │ PLY meshes
                   ▼                                  ▼
┌──────────────────────── FASTAPI SERVER  server.py ────────────────────────┐
│ guard middleware: upload pre-checks (401/411/413/429) + security headers  │
│                                                                           │
│ ┌──────────────┐  ┌────────────────────────┐  ┌─────────────────────────┐ │
│ │ /auth/*      │  │ /studies /raw_patients │  │ /files/{study}/{model}/ │ │
│ │ sign in / up │  │ /models /jobs          │  │ only YOUR result files  │ │
│ └──────┬───────┘  └───────────┬────────────┘  └────────────┬────────────┘ │
│        │                      │ POST /jobs                 │              │
│        │            ┌─────────▼─────────┐                  │              │
│        │            │ JOBS (in memory)  │                  │              │
│        │            │ one run at a time │                  │              │
│        │            └─────────┬─────────┘                  │              │
└────────┼──────────────────────┼────────────────────────────┼──────────────┘
         │                      │ spawns a subprocess        │
         ▼                      ▼                            ▼
┌──────────────────┐  ┌───────────────────────┐  ┌──────────────────────────┐
│ data/accounts.db │  │ INFERENCE   run.py    │  │ data/users/<id>/         │
│ users · sessions │  │ load → crop → predict │─►│ raw · work · out · tmp   │
└──────────────────┘  │ → score → mesh →      │  └──────────────────────────┘
                      │ render                │
                      └───────────▲───────────┘
                                  │ manifest + weights
                        ┌─────────┴─────────┐
                        │ models/<model id> │
                        └───────────────────┘
```

**Three processes / places, each with one job**

| Part | Runs | Owns |
|---|---|---|
| Browser UI | the user's browser | display state only; never computes a score |
| Server | one Python process | sign-in, authorisation, file access, starting runs |
| Inference | a child process per run | turning a scan into result files |

The server never imports PyTorch code paths for scoring; runs are isolated in
their own process, so a crashed run cannot take the server down.

---

## 2. Components

| File | Responsibility | Never does |
|---|---|---|
| `src/backend/server.py` | HTTP API, sign-in routes, per-user authorisation, upload checks, job runner | compute scores |
| `src/backend/accounts.py` | accounts, password hashing, sessions, access tokens; account CLI | speak HTTP |
| `src/backend/ratelimit.py` | token-bucket rate limiter and all limit numbers | know about users or HTTP |
| `src/backend/paths.py` | every folder path (per account), `safe_name` | read images or create folders |
| `src/backend/run.py` | orchestrates one inference run; writes `run.json` | handle HTTP |
| `src/backend/pipeline.py` | load, resample, heart crop, normalise, UNet inference | score |
| `src/backend/pipeline_nnunet.py` | nnUNet inference via its CLI | score |
| `src/backend/scoring.py` | Agatston maths per slice; patient totals | load models or files |
| `src/backend/grouping.py` | links per-slice components into 3D lesions (descriptive only) | change any score |
| `src/backend/render.py` | CT and mask PNGs, `slices.json` | score |
| `src/backend/mesh.py` | 3D surfaces (PLY) for the Anatomy view | measure anything |
| `src/backend/registry.py` | loads and validates `manifest.yaml`, locks weights by SHA-256 | run models |
| `src/backend/ingest.py` | fixes DICOM files without an extension | — |
| `ui/index.html`, `ui/app.css` | layout and styles | — |
| `ui/app.js` | state, render loop, all server calls (`api()`), sign-in screen | compute scores |
| `ui/view3d.js` | 3D viewer (meshes, slice plane, picking) | fetch CSVs or score |
| `ui/report.js` | one-page A4 report and slice-table CSV of the open result (§11.1) | write files, call the server, change `state` or recompute a score |

Dependency direction is one-way (from `Principles.md`):

```text
server.py ──┬──► accounts.py ──► paths.py
            ├──► ratelimit.py
            ├──► paths.py · registry.py · ingest.py
            │
            └──(subprocess)──► run.py ──┬──► paths.py · registry.py
                                        ├──► pipeline.py · pipeline_nnunet.py
                                        ├──► scoring.py ──► grouping.py
                                        └──► render.py · mesh.py
```

---

## 3. Data layout

```text
Predict-Studio/
├── data/                          (git-ignored)
│   ├── accounts.db                users + sessions (SQLite). No scan data.
│   └── users/
│       └── <account id>/          EVERYTHING one account owns
│           ├── raw/<scan>/        uploaded scan (DICOM files or one NIfTI)
│           ├── work/<study>/
│           │   ├── crop/          prep cache: ct.nii.gz + heart.nii.gz (heart-cropped)
│           │   └── full/          prep cache: ct.nii.gz (full field of view)
│           ├── out/<study>/<model>/   results the UI shows (see §10)
│           └── tmp/temp_<id>/     an upload being checked (removed afterwards)
├── models/<model id>/
│   ├── manifest.yaml              facts the weights cannot hold (spacing, threshold, crop…)
│   └── best_model.pth | nnUNet_results/
└── ui/                            the web app (public: code only, no data)
```

**Why one folder per account (and no table of who-owns-what).** The location
*is* the ownership. There is no second record that could disagree with the
disk, and isolation reduces to one rule: build every path from the signed-in
account's id. The folder name is the numeric id (never the username), ids are
never reused, so a new account cannot inherit a deleted one's data.

**Why two prep caches.** A heart-cropped CT and a full CT are different model
inputs. Keeping them apart (`crop/` vs `full/`) means a model can never
silently reuse a CT that was prepared the other way.

---

## 4. Accounts and sign-in

* **Storage:** `data/accounts.db` — `users(id, username, salt, pw_hash, is_admin, access_hash, created)` and `sessions(token_hash, user_id, access_hash, expires)`.
* **Passwords:** scrypt with a random 16-byte salt per user, compared in constant time. Never stored or logged in plain text.
* **Sessions:** a random token in an **HttpOnly, SameSite=Strict** cookie (`predict_session`), valid 7 days. The database keeps only its SHA-256.
* **Access tokens:** granted by the administrator in the environment variable `PREDICT_ACCESS_TOKENS` (comma-separated). Needed to **create an account and at every sign-in**. Each session remembers which token it used; removing that token (and restarting) ends those sessions.
* **Admin:** created from the command line, needs no access token, can run on server folders.

### 4.1 Create account

```text
  POST /auth/signup   { username, password, access token }
                    │
                    ▼
  ┌──────────────────────────────────┐  empty  ┌────────────────────────────────┐
  │ global attempt box has a ball?   │ ──────► │ 429 Too many attempts          │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ access token currently granted?  │ ──────► │ 400 Invalid access token       │
  └─────────────────┬────────────────┘         │ (sign-up box is not touched)   │
                    │ yes                      └────────────────────────────────┘
                    ▼
  ┌──────────────────────────────────┐  empty  ┌────────────────────────────────┐
  │ sign-up box has a ball?          │ ──────► │ 429 Too many new accounts      │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ username valid (3-64 chars)?     │ ──────► │ 400 with the reason            │
  │ password at least 8 chars?       │         └────────────────────────────────┘
  └─────────────────┬────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ this token made < 50 accounts?   │ ──────► │ 400 Token reached its limit    │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ username still free?             │ ──────► │ 400 Username taken             │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐
  │ INSERT user                      │
  │ (scrypt hash + token hash)       │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐
  │ sign in → session cookie         │
  └──────────────────────────────────┘
```

### 4.2 Sign in

```text
  POST /auth/login   { username, password, access token }
                    │
                    ▼
  ┌──────────────────────────────────┐  empty  ┌────────────────────────────────┐
  │ global attempt box has a ball?   │ ──────► │ 429 Too many attempts          │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐  empty  ┌────────────────────────────────┐
  │ this username's failure box      │ ──────► │ 429 Too many failed sign-ins   │
  │ has a ball?                      │         │ for this account               │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐
  │ look up user (unknown name:      │
  │ hash against a dummy, so the     │
  │ time taken reveals nothing)      │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ password matches?                │ ──────► │ take a failure ball            │
  └─────────────────┬────────────────┘         │ wait 0.5 s                     │
                    │ yes                      │ 401 Invalid username,          │
                    │                          │ password or access token       │
                    │                          └────────────────────────────────┘
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ admin?  or                       │ ──────► │ take a failure ball            │
  │ access token granted?            │         │ wait 0.5 s → same 401          │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐
  │ new random session token         │
  │ store only its SHA-256           │
  │ + which access token was used    │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐
  │ Set-Cookie predict_session       │
  │ HttpOnly · SameSite=Strict       │
  │ 7 days                           │
  └──────────────────────────────────┘
```

Every failure gives the *same* message after the *same* delay, so an attacker
cannot learn which usernames exist or which part was wrong. A successful
sign-in costs nothing, so signing in on several devices is never throttled.

### 4.3 How a session ends

| Session ends when | Mechanism |
|---|---|
| the user signs out | `POST /auth/logout` deletes the session row and clears the cookie |
| 7 days pass | `expires` is checked on every request |
| the admin removes the access token (and restarts) | the session's token hash is no longer granted, checked on every request |
| the admin resets the password or deletes the account (CLI) | all of that account's session rows are deleted |

The UI notices on the next request (any `401`) and shows the sign-in screen.

---

## 5. Request authorisation (the direct-link rule)

The classic multi-user bug: a URL such as `/data/out/172/a1-roi/run.json`
works for *anyone who knows it*. PrediCT closes this **by construction**:

1. `data/` is **never** mounted as public static files.
2. Every data route resolves the user from the **session cookie only** — never from the URL, query or body.
3. Paths are built as `data/users/<that user's id>/…/<what the URL names>`.
4. Names from the URL must pass `safe_name` (one folder name, no `/ \ : * ? " < > |`, not starting with `.`), and a file path must still be inside the folder after resolving (`../` cannot escape).

```text
  GET /files/172/a1-roi/run.json
                    │
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ valid session cookie? (exists,   │ ──────► │ 401 → UI shows the             │
  │ not expired, token granted)      │         │ sign-in screen                 │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐
  │ user = taken from the COOKIE     │
  │ (never from the URL)             │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ '172' and 'a1-roi' are single    │ ──────► │ 400 Invalid name               │
  │ safe folder names?               │         └────────────────────────────────┘
  └─────────────────┬────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐
  │ base = data/users/<user id>/     │
  │        out/172/a1-roi            │
  │ target = resolve(base/run.json)  │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ target still inside base         │ ──────► │ 404 Not found                  │
  │ and is a file?                   │         └────────────────────────────────┘
  └─────────────────┬────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐
  │ 200 the file                     │
  │ Cache-Control: private, no-cache │
  └──────────────────────────────────┘
```

**Consequence:** if Alice sends Bob her link `?study=172&model=a1-roi`, Bob's
browser asks for *Bob's* study 172. Bob sees his own (if he has one by that
name) or "not found" — never Alice's data. The same rule covers jobs
(`GET /jobs/{id}` is 404 unless the job is yours), deletes and runs.

---

## 6. Abuse limits (token buckets)

A **token bucket** is a box of balls: each action takes one ball; one ball
comes back every *n* seconds; an empty box means "wait". Refill is computed
from timestamps when the box is used — no background thread.

```text
      THE BOX                      each action takes one ball
   ┌───────────────┐               one ball comes back every n seconds
   │ ● ● ● ● ● ● ● │  ── take ──►  allowed
   │ ● ● ● ○ ○ ○ ○ │  ◄── refill   (never more than the box holds)
   └───────────────┘               empty box = 429 "try again in N s"

  action arrives
                    │
                    ▼
  ┌──────────────────────────────────┐
  │ refill: balls += elapsed/period  │
  │ (never above capacity)           │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ at least one ball in the box?    │ ──────► │ 429 "try again in N s"         │
  └─────────────────┬────────────────┘         │ N = time to the next ball      │
                    │ yes                      └────────────────────────────────┘
                    ▼
  ┌──────────────────────────────────┐
  │ take one ball → allowed          │
  └──────────────────────────────────┘
```

| Limit | Scope | Capacity | One ball back every | Stops |
|---|---|---|---|---|
| Sign-ups | whole server | 20 | 3 min | a leaked token creating hundreds of accounts |
| Accounts per access token | lifetime | 50 | — | unbounded damage from one token |
| Failed sign-ins | per username | 5 | 60 s | password guessing on one account |
| All sign-in attempts | whole server | 60 | 1 s | guessing across many usernames |
| Uploads | per account | 10 | 2 min | repeated huge uploads |
| Upload size | per upload | 4 GB | — | filling the disk in one request |
| Storage | per account | 100 GB | — | filling the disk slowly |
| Runs | whole server | 1 at a time | — | GPU memory exhaustion |

Numbers live in `ratelimit.py` (buckets), `accounts.py` (per-token cap) and
`server.py` (size, quota). Per-key tables drop idle entries when they grow,
so made-up usernames cannot exhaust memory. A token without access cannot
drain the sign-up bucket (the token is checked first).

---

## 7. Upload workflow

```text
  Browser: pick or drop a folder → name it → XHR POST /studies (progress bar)
                    │
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ guard middleware, BEFORE the     │ ──────► │ 401 Not signed in              │
  │ body is read: signed in?         │         └────────────────────────────────┘
  └─────────────────┬────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ Content-Length present?          │ ──────► │ 411 Size unknown               │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ upload ≤ 4 GB?                   │ ──────► │ 413 Upload too large           │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐  empty  ┌────────────────────────────────┐
  │ upload box has a ball?           │ ──────► │ 429 Too many uploads           │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐
  │ save files to tmp/temp_<id>/     │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ account ≤ 100 GB?                │ ──────► │ remove tmp                     │
  └─────────────────┬────────────────┘         │ 413 Storage limit reached      │
                    │ yes                      └────────────────────────────────┘
                    ▼
  ┌──────────────────────────────────┐
  │ fix extensionless DICOM files    │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ DICOM in only one folder?        │ ──────► │ 400 Upload the specific        │
  └─────────────────┬────────────────┘         │ folder                         │
                    │ yes                      └────────────────────────────────┘
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ only DICOM / NIfTI files?        │ ──────► │ reply requires_cleaning        │
  └─────────────────┬────────────────┘         │ UI asks the user:              │
                    │ yes                      │  remove → POST /studies/clean  │
                    │                          │          (continues below)     │
                    │                          │  cancel → DELETE, nothing kept │
                    │                          └────────────────────────────────┘
                    ▼
  ┌──────────────────────────────────┐
  │ name = the given name, or        │
  │ SHA-1 of SeriesInstanceUID       │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐
  │ move to raw/<name>/              │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐
  │ UI refreshes the scan list       │
  │ and auto-selects the new scan    │
  └──────────────────────────────────┘
```

---

## 8. Run workflow (jobs)

```text
 Browser                      server.py                    run.py (subprocess)
    │                             │                                │
    │ POST /jobs                  │                                │
    │ {scan, model, crop, name}   │                                │
    │────────────────────────────►│ user from cookie               │
    │                             │ scan must be YOURS             │
    │                             │ (server path: admin only)      │
    │                             │ names pass safe_name           │
    │                             │ GPU free?                      │
    │◄──── 409 if busy ───────────│ (never shows another           │
    │      (stop)                 │  user's study name)            │
    │                             │                                │
    │                             │── spawn: --user --input ──────►│
    │◄──── { job_id } ────────────│   --study --model --crop       │
    │                             │                                │── writes
    │ GET /jobs  (every second)   │                                │   work/<study>/crop|full
    │────────────────────────────►│◄──── "[NN%] stage" lines ──────│   out/<study>/<model>/
    │◄──── status · % · stage ────│                                │
    │                             │◄──── exit code ────────────────│
    │                             │      (+ last output lines)     │
    │ toast "Finished … Open"     │                                │
    │ (never navigates by itself) │                                │
```

```text
  POST /jobs accepted
         │
         ▼
  ┌─────────────┐   exit code 0    ┌──────┐
  │   running   │ ───────────────► │ done │   result appears in the study list
  └──────┬──────┘                  └──────┘
         │ exit code ≠ 0
         ▼
  ┌───────────────────────────┐
  │ failed                    │
  │ error = last output lines │
  └───────────────────────────┘

  · one job at a time on the whole server
  · progress is parsed from "[NN%] stage" lines printed by run.py
  · survives page reloads (the state lives on the server)
  · forgotten if the server restarts
```

---

## 9. Inference pipeline

`run.py` — only *where* it reads and writes depends on the account; *what* it
computes never does.

```text
                ┌──────────────────────────────────────┐
                │ load_manifest(model)                 │
                │ validate fields · check weights hash │
                └──────────────────┬───────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ crop = request, else manifest default│
                │ cache = work/<study>/crop  or  /full │
                └──────────────────┬───────────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │ cached ct.nii.gz exists?  │
                     └─────────────┬─────────────┘
                  yes ┌────────────┴────────────┐ no
                      ▼                         ▼
        ┌───────────────────────┐   ┌──────────────────────────────┐
        │ read cached CT        │   │ load: DICOM (GDCM) or NIfTI  │
        │ (+ heart mask)        │   │ resample 0.37 × 0.37 × 3 mm  │
        └───────────┬───────────┘   │ crop? TotalSegmentator heart │
                    │               │       + 8 mm margin          │
                    │               │ reorient to RAS              │
                    │               │ assert spacing + RAS axes    │
                    │               │ write cache                  │
                    │               └──────────────┬───────────────┘
                    └──────────────┬───────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │ model architecture?       │
                     └─────────────┬─────────────┘
               nnunet ┌────────────┴────────────┐ UNet
                      ▼                         ▼
        ┌───────────────────────┐   ┌──────────────────────────────┐
        │ nnUNetv2_predict CLI  │   │ normalise HU 0-1200          │
        │ (pipeline_nnunet.py)  │   │ sliding-window UNet (MONAI)  │
        └───────────┬───────────┘   └──────────────┬───────────────┘
                    └──────────────┬───────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ probability volume (Z, Y, X)         │
                │ save pred.nii.gz + ct.nii.gz         │
                └──────────────────┬───────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ score: per-slice components          │
                │ area × density weight (scoring.py)   │
                └──────────────────┬───────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ group into 3D lesions (grouping.py)  │
                │ descriptive only, never changes score│
                └──────────────────┬───────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐   no   ┌───────────────────┐
                │ 3D roll-up total == per-slice total? │ ─────► │ STOP: grouping    │
                └──────────────────┬───────────────────┘        │ leaked into score │
                                   │ yes                        └───────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ lesions.csv · lesions_3d.csv         │
                │ mesh/*.ply (mesh.py)                 │
                │ run.json (provenance)                │
                │ slices/ct, slices/mask, slices.json  │
                └──────────────────────────────────────┘
```

**Scoring rules (scoring.py)**

| Item | Rule |
|---|---|
| Area, binary model (A1) | voxel count above threshold × pixel area |
| Area, coverage model (A3) | Σ probabilities × pixel area (never thresholded) |
| Density weight (peak HU) | < 130 → 0 · 130–199 → 1 · 200–299 → 2 · 300–399 → 3 · ≥ 400 → 4 |
| Minimum lesion | 1.0 mm² (smaller components are listed as *withheld*) |
| Lesion score | area × density weight; patient total = sum of included lesions |
| Risk tier (display) | 4-tier 0 / 1–100 / 101–400 / >400 (written to run.json) · 6-tier adds 101–300, 301–400, 401–1000, >1000 (UI only) |

---

## 10. Result files

`data/users/<id>/out/<study>/<model>/`

| File | Content | Used by |
|---|---|---|
| `run.json` | total, risk category, model, threshold, crop + crop default + margin, HU window, spacing, checkpoint SHA, date, mesh manifest | all tabs, Pipeline Parameters |
| `slices.json` | per slice: index, z (mm), slice score, has calcium | strip, track, contact sheet |
| `lesions.csv` | one row per per-slice component (area, peak HU, weight, included, bbox, 3D key) | tables, rings |
| `lesions_3d.csv` | one row per 3D lesion (slices, span, total) | 3D lesion index |
| `slices/ct/slice_NNN.png` | CT for display, window −100…400 HU | every image |
| `slices/mask/slice_NNN.png` | prediction overlay (alpha = probability) | every image |
| `mesh/lesions_pNNN.ply`, `mesh/heart.ply` | 3D surfaces (display only) | Anatomy |
| `ct.nii.gz`, `pred.nii.gz` | the volumes in RAS | analysis, verification |

---

## 11. User interface

```text
  page opens → boot()
                    │
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ GET /auth/me → signed in?        │ ──────► │ sign-in screen                 │
  └─────────────────┬────────────────┘         │ (Sign in / Create account)     │
                    │ yes                      │ success → reload same URL      │
                    │                          │          → boot() again        │
                    │                          └────────────────────────────────┘
                    ▼
  ┌──────────────────────────────────┐
  │ sidebar: studies, upload,        │
  │ scan picker, models, runs        │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐   no    ┌────────────────────────────────┐
  │ ?study & ?model in the URL?      │ ──────► │ start screen                   │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ yes
                    ▼
  ┌──────────────────────────────────┐  error  ┌────────────────────────────────┐
  │ load YOUR files via /files/…     │ ──────► │ start screen + the error       │
  │ run.json, slices.json, CSVs      │         │ (e.g. not your study)          │
  └─────────────────┬────────────────┘         └────────────────────────────────┘
                    │ ok
                    ▼
  ┌──────────────────────────────────┐
  │ render()                         │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐
  │ 01 Argument · 02 Instrument      │
  │ 03 Contact Sheet · 04 Anatomy    │
  └──────────────────────────────────┘
```

**One state object, one `render()`** — every handler changes `state` and calls
`render()`; there is no other update path. Display-only preferences (tier
scheme, panel width, folded sections, tile size) are kept in `localStorage`.

| Area | What it shows | Server calls |
|---|---|---|
| Header | tabs, view mode (original / prediction / calcium), 4/6-tier, study, user, sign out (see below) | `POST /auth/logout` |
| Sidebar · upload | progress card (bytes, then "checking") | `POST /studies`, `/studies/clean/…` |
| Sidebar · Run Pipeline | scan picker (search, newest first), model, crop toggle, name, runs panel; server path field for the admin | `GET /raw_patients`, `GET /models`, `POST /jobs`, `GET /jobs` |
| Sidebar · studies | one row per study, result chips, filter, delete dialog | `GET /studies`, `DELETE /studies/{id}` |
| 01 Argument | score, tier bar, findings, exhibit + scrolling strip, Pipeline Parameters, 3D lesion index, withheld; **report + CSV export icons** (top-right of the title bar) | `/files/…` |
| 02 Instrument | slice track, zoomable viewer (Ctrl+wheel, drag, scale bar), lesion details | `/files/…` |
| 03 Contact Sheet | every slice, tile size slider, counted components | `/files/…` |
| 04 Anatomy | 3D meshes, slice plane, camera presets, "not vessel-attributed" banner | `/files/…/mesh/*.ply` |

**Header width.** The header is a CSS size container (`#tabs { container: tabs / inline-size }`), so what
fits is decided by the header's own width, which shrinks when the sidebar is open, not by the window's.
As it narrows, items drop out in this order: app name, study badge, signed-in name, then tab and
button padding is tightened. **Sign out never shrinks or leaves the screen** (checked down to a ~900 px header).
The thresholds are the `@container tabs (…)` rules in `ui/app.css`.

### 11.1 One-page report and CSV export

Two icons at the top-right of the **01 Argument** title bar (not in the header: the header is
already full at a 100 % browser zoom and would push Sign out off screen) work on the
result that is already open:

* **Report** (`summarize` icon) builds a one-page A4 report: score and tier, the
  two highest-scoring slices as CT + prediction thumbnails, a slice-by-slice table,
  short conclusions and the method. The user prints it or chooses *Save as PDF* in
  the browser's print dialog. There is no extra dialog: the title is "PrediCT
  Studio", the study line is the name the user gave the scan at upload, and the
  reader is the signed-in username.
* **CSV** (`table_view` icon) downloads the same slice table as
  `PrediCT_<study>_<model>_slices.csv`: one row per calcium slice plus a `TOTAL` row,
  with `study` and `model` on every row. A study name starting with `= + - @` is
  defused against spreadsheet formulas, and commas or quotes are quoted.

```text
  state (run.json · slices.json · lesions.csv · lesions_3d.csv,
         already loaded by app.js)
                     │ read only
                     ▼
  ┌──────────────────────────────────┐   no total in run.json
  │ buildReportModel()   pure, no DOM│ ───────────────────────► toast, no report
  │ rows · totals · conclusions      │
  │ cross-checks → warnings          │
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐
  │ renderReportHtml()  A4 HTML+CSS  │   user-typed text is escaped, never parsed as HTML
  └─────────────────┬────────────────┘
                    ▼
  ┌──────────────────────────────────┐   the only requests: the key slice's two PNGs,
  │ hidden iframe (same origin) →    │   through the same /files route as the viewer
  │ wait for images → print()        │
  └──────────────────────────────────┘
```

| Design point | What it means |
|---|---|
| **Read-only, no new route** | `ui/report.js` is loaded after `app.js` and only reads its globals. Nothing is written to `data/`, the server, or `out/`; `server.py`, `run.py` and `app.js` are unchanged. The PDF is made by the browser. |
| **Same authorisation as the viewer** | The two key-slice images come through `/files/{study}/{model}/…` with the session cookie (§5). |
| **Numbers are copied, not recomputed** | Per-slice values come from `slices.json` / `lesions.csv`; tier from `run.json` (`risk_category`) plus the 6-tier reading from the UI's own scheme. |
| **Disagreements are printed** | The table is summed only to *check* it. If a slice, the total, the component count or the tier disagrees between files, a red box on the report lists it (principle: no silent wrong number). |
| **One page** | Table rows are the slices with calcium (the rest are summarised as a count). More than 18 rows are split into two side-by-side tables. If the page still overflows, a toast says it will print on two pages. |
| **Missing data shows `—`** | A field an older `run.json` lacks is never guessed. |
| **Conclusions** | Generated from the numbers only: score and tier, lesions and slice span, largest slice and lesion with share of the total, peak density, withheld ledger, and what the area means for this model (coverage or binary). No treatment advice; the footer states it is automated output for clinician review. |
| **Method block** | The same `paramGroups()` the Pipeline Parameters panel uses, so the report describes the run that produced the numbers, not today's settings. |
| **File names** | The print dialog proposes `PrediCT_<study>_<model>` as the PDF name; the CSV is built in memory and saved by the browser (`Blob`), so there is no server round trip either. |
| **Patient identity** | The app never reads DICOM patient fields, so the only identifier on the report and CSV is the study name typed at upload (the report also names the signed-in user). |

Verified by `tests/test_report.py` (§15): the report's and CSV's totals equal `run.json`
for a hand-made result and for every result found under `data/`; disagreements,
the zero-calcium case, HTML escaping and CSV formula defusing are covered too.

Every server call goes through `api()`: a `401` anywhere shows the sign-in
screen (expired, signed out elsewhere, or access revoked).

---

## 12. API reference

All routes except `/auth/login`, `/auth/signup`, `/auth/logout`, `/` and
`/ui/*` require a valid session cookie (`401` otherwise).

| Method | Path | Body / query | Returns | Errors |
|---|---|---|---|---|
| POST | `/auth/signup` | `{username, password, access_token}` | `{username, is_admin}` + cookie | 400, 429 |
| POST | `/auth/login` | `{username, password, access_token}` | `{username, is_admin}` + cookie | 401, 429 |
| POST | `/auth/logout` | — | `{ok}`, cookie cleared | — |
| GET | `/auth/me` | — | `{username, is_admin}` | 401 |
| GET | `/files/{study}/{model}/{path}` | — | the file | 400, 404 |
| POST | `/studies` | multipart `files[]`, `custom_name?` | `{study_id}` or `{requires_cleaning, temp_id, invalid_files}` | 400, 411, 413, 429 |
| POST | `/studies/clean/{temp_id}` | `?custom_name=` | `{study_id}` | 400, 404 |
| DELETE | `/studies/clean/{temp_id}` | — | `{status: aborted}` | 400 |
| GET | `/studies` | — | `[{id, model}]` (your results) | — |
| DELETE | `/studies/{study}` | `?model=a&model=b` or `?all=true` | `{deleted: [...]}` | 400, 404, 409 |
| GET | `/raw_patients` | — | `[{id, uploaded, path?}]` newest first (`path` admin only) | — |
| GET | `/models` | — | `[{id, name, crop}]` | 500 if a manifest is invalid |
| POST | `/jobs` | `{scan \| input_path(admin), model_id, crop?, study_id?}` | `{job_id}` | 400, 403, 404, 409 |
| GET | `/jobs` | — | your jobs, newest first | — |
| GET | `/jobs/{id}` | — | one of your jobs | 404 |

Every response carries `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
`Referrer-Policy: no-referrer`.

---

## 13. Security model

| Threat | Defence |
|---|---|
| Opening another user's data by URL | no public `data/`; user from cookie only; paths built per account (§5) |
| Path traversal (`../`, encoded) | `safe_name` on every name; resolved file must stay inside the base folder |
| Running on another user's scan | `scan` must exist in your own `raw/`; server paths admin-only (403) |
| Session theft by page scripts | HttpOnly cookie |
| Cross-site requests (CSRF) | SameSite=Strict cookie; deletes are real `DELETE` (pre-flighted) |
| Clickjacking, MIME sniffing | `X-Frame-Options: DENY`, `nosniff` |
| Password guessing | per-username failure bucket + global bucket + 0.5 s delay |
| Username discovery | identical message and timing for every failure (dummy hash for unknown names) |
| Account flooding | sign-up bucket + 50 accounts per token + revocable tokens |
| Stolen database file | scrypt password hashes; session tokens stored as SHA-256 only |
| Disk filling | 4 GB per upload (checked before reading), 100 GB per account, upload bucket |
| GPU exhaustion | one run at a time |
| Leaking other users' activity | 409 for a busy GPU never names the other user's study; job lists are per user |
| Deleted account's folder reused | ids never reused (AUTOINCREMENT) |
| Admin lockout | admin needs no token; `reset-password` CLI |

---

## 14. Configuration and operations

| Setting | Where | Default |
|---|---|---|
| Access tokens | env `PREDICT_ACCESS_TOKENS` (comma-separated) | none → only the admin can sign in |
| Session length | `accounts.SESSION_DAYS` | 7 days |
| Accounts per token | `accounts.MAX_ACCOUNTS_PER_TOKEN` | 50 |
| Rate limits | `ratelimit.py` constants | see §6 |
| Upload size / quota | `server.MAX_UPLOAD_BYTES`, `server.USER_QUOTA_BYTES` | 4 GB / 100 GB |
| Port | `server.py` `__main__` | 8001 |
| Model behaviour | `models/<id>/manifest.yaml` | per model |

**First start**

```text
1. set PREDICT_ACCESS_TOKENS=tokenA,tokenB          (bash: export PREDICT_ACCESS_TOKENS=...)
   strong tokens: python -c "import secrets; print(secrets.token_urlsafe(16))"
2. python -m src.backend.accounts create-admin <admin username>     (password is prompted)
3. python -m src.backend.accounts adopt-legacy <admin username>     (one-time: old shared data → admin)
4. python -m src.backend.server
```

**Account commands** — `python -m src.backend.accounts …`

| Command | Effect |
|---|---|
| `create-admin <username>` | create the admin (password prompted twice, never echoed) |
| `list` | id, role, username, created, folder |
| `reset-password <username>` | new password; ends all that account's sessions |
| `delete <username>` | asks for confirmation; ends sessions, removes account and its folder |
| `adopt-legacy <username>` | moves pre-accounts `data/raw`, `data/work`, `data/out`, `data/uploads/<name>` into that account; never overwrites |

**Rotating access** — give each person or group their own token. To revoke:
remove it from `PREDICT_ACCESS_TOKENS` and restart; everyone who signed in
with it is signed out and cannot sign in again until given a new token.

**Command-line runs** (admin / batch): `python -m src.backend.run --user <id> --input /path/to/dicom --model a1-roi` — the result appears in that account.

Security events are logged one per line in the server output: `[auth] <time> signup|login|login-failed|logout|delete …`.

---

## 15. Testing

| Suite | Covers | Run |
|---|---|---|
| `tests/test_accounts.py` | sign-up/sign-in rules, admin exemption, revocation, expiry, hashing, per-token cap, id reuse, password reset | `python -m pytest tests -v` |
| `tests/test_ratelimit.py` | bucket drain/refill, look-only, keyed buckets, memory bound | 〃 |
| `tests/test_paths.py` | `safe_name`, per-account folders, numeric ids | 〃 |
| `tests/test_grouping.py` | 3D linking never changes a score | 〃 |
| `tests/test_mesh.py` | mesh geometry, orientation, PLY round-trip (needs scikit-image) | 〃 |
| `tests/test_report.py` | one-page report and CSV: totals equal `run.json`, disagreements reported, zero-calcium case, HTML escaping, CSV formula defusing (runs `tests/report_check.mjs` in Node; skipped without Node) | 〃 |

---

## 16. Known limits and next steps

* **Temporary sign-in.** Before real deployment: sign-in through an identity provider (OpenID Connect — Google/Microsoft or the hospital's, e.g. via Keycloak/Auth0), per-person invites with admin approval, roles (admin / clinician / viewer), two-factor sign-in, HTTPS everywhere, and an audit log of who opened which study. The folder-per-account design stays; only `accounts.py` and the sign-in screen change.
* **Token changes need a restart** (environment variable); a restart also forgets the in-memory job list.
* **One run at a time**, no queue and no cancel button yet.
* **Single server.** Accounts in SQLite and jobs in memory assume one server process; scaling out would move both to a shared database and a job queue.
