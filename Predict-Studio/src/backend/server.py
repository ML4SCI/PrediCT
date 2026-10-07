"""
server.py — the PrediCT Studio web server (FastAPI).

Every API route needs a signed-in user (session cookie, see accounts.py), and
works only inside that user's own folder data/users/<id>/ (see paths.py).
The user ALWAYS comes from the cookie, never from the URL or the request
body, so a link copied from someone else resolves inside your own folder and
finds nothing. Result files are served by /files/... for the same reason;
the data/ folder is never mounted publicly.

Abuse limits (token buckets, ratelimit.py) and upload size / quota checks are
applied here. Inference runs in a subprocess (run.py), one at a time.

Start:  python -m src.backend.server      (from Predict-Studio/)
"""
import math
import threading
import time
import uuid
import shutil
import traceback
from pathlib import Path
from typing import List

from datetime import datetime

from fastapi import Depends, FastAPI, UploadFile, File, HTTPException, Form, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

from src.backend import accounts, ratelimit
from src.backend.paths import (study_id_from_series, user_root, raw_dir, out_dir, tmp_dir,
                               study_dirs, scan_input_dir, safe_name)
from src.backend.registry import list_models
from src.backend.run import run
from src.backend.ingest import fix_extensions

app = FastAPI(title="PrediCT Server")

COOKIE = "predict_session"
MAX_UPLOAD_BYTES = 4 * 1024**3      # one upload; refused before its body is read
USER_QUOTA_BYTES = 100 * 1024**3    # everything one account stores

SIGNUPS = ratelimit.TokenBucket(*ratelimit.SIGNUPS)
LOGINS_GLOBAL = ratelimit.TokenBucket(*ratelimit.LOGINS_GLOBAL)
LOGINS_PER_USER = ratelimit.KeyedBuckets(*ratelimit.LOGINS_PER_USER)
UPLOADS_PER_USER = ratelimit.KeyedBuckets(*ratelimit.UPLOADS_PER_USER)

# In-memory job state
JOBS = {}


def log(event: str, **details):
    """One line per security-relevant event, in the server output."""
    extra = " ".join(f"{k}={v}" for k, v in details.items())
    print(f"[auth] {datetime.now().isoformat(timespec='seconds')} {event} {extra}", flush=True)


def checked(name: str) -> str:
    """paths.safe_name, reported to the client as a 400 rather than a crash."""
    try:
        return safe_name(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def too_many(wait: float, what: str):
    """429 when a token bucket is empty."""
    if wait:
        raise HTTPException(status_code=429, detail=f"Too many {what}. Try again in {math.ceil(wait)} s.")


# ── who is asking ─────────────────────────────────────────────────────────
def user_from_request(request: Request) -> dict | None:
    return accounts.user_for_session(request.cookies.get(COOKIE))


def current_user(request: Request) -> dict:
    """Dependency for every API route: the signed-in user, or 401."""
    user = user_from_request(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return user


@app.middleware("http")
async def guard(request: Request, call_next):
    """Runs before any route. Uploads are refused BEFORE their body is read
    (FastAPI would otherwise receive the whole file first): no session, unknown
    or too-large size, or too many uploads. Every response gets the security
    headers."""
    if request.method == "POST" and request.url.path == "/studies":
        user = user_from_request(request)
        size = request.headers.get("content-length", "")
        if user is None:
            return JSONResponse({"detail": "Not signed in."}, status_code=401)
        if not size.isdigit():
            return JSONResponse({"detail": "Upload size unknown (Content-Length required)."}, status_code=411)
        if int(size) > MAX_UPLOAD_BYTES:
            return JSONResponse({"detail": f"Upload too large: the limit is {MAX_UPLOAD_BYTES // 1024**3} GB."},
                                status_code=413)
        wait = UPLOADS_PER_USER.take(user["id"])
        if wait:
            return JSONResponse({"detail": f"Too many uploads. Try again in {math.ceil(wait)} s."},
                                status_code=429)
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


# ── accounts ──────────────────────────────────────────────────────────────
class Credentials(BaseModel):
    username: str
    password: str
    access_token: str = ""


def open_session(c: Credentials, request: Request, response: Response) -> dict:
    """Log in and set the session cookie. HttpOnly: page scripts cannot read
    it. SameSite=Strict: other websites cannot send requests that carry it."""
    name = (c.username or "").strip().lower()
    too_many(LOGINS_GLOBAL.take(), "sign-in attempts")
    # Only failures use up the per-account allowance: signing in on several
    # devices is never throttled, guessing a password is.
    too_many(LOGINS_PER_USER.wait(name), "failed sign-ins for this account")
    try:
        token, user = accounts.login(c.username, c.password, c.access_token)
    except accounts.AuthError as e:
        LOGINS_PER_USER.take(name)
        log("login-failed", username=name)
        raise HTTPException(status_code=401, detail=str(e))
    response.set_cookie(COOKIE, token, max_age=accounts.SESSION_DAYS * 86400, path="/",
                        httponly=True, samesite="strict", secure=request.url.scheme == "https")
    log("login", username=user["username"])
    return {"username": user["username"], "is_admin": user["is_admin"]}


@app.post("/auth/signup")
def signup(c: Credentials, request: Request, response: Response):
    # The token is checked before a sign-up ball is taken, so someone without a
    # token cannot empty the sign-up bucket for everyone else.
    too_many(LOGINS_GLOBAL.take(), "attempts")
    if not accounts.token_ok(c.access_token):
        time.sleep(accounts.FAILED_LOGIN_DELAY)
        raise HTTPException(status_code=400, detail="Invalid access token.")
    too_many(SIGNUPS.take(), "new accounts right now")
    try:
        uid = accounts.create_user(c.username, c.password, c.access_token)
    except accounts.AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    log("signup", username=accounts.clean_username(c.username), id=uid)
    return open_session(c, request, response)


@app.post("/auth/login")
def login(c: Credentials, request: Request, response: Response):
    return open_session(c, request, response)


@app.post("/auth/logout")
def logout(request: Request, response: Response):
    user = user_from_request(request)
    accounts.logout(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE, path="/")
    if user:
        log("logout", username=user["username"])
    return {"ok": True}


@app.get("/auth/me")
def me(user: dict = Depends(current_user)):
    return {"username": user["username"], "is_admin": user["is_admin"]}


# ── result files ──────────────────────────────────────────────────────────
@app.get("/files/{study}/{model}/{path:path}")
def get_file(study: str, model: str, path: str, user: dict = Depends(current_user)):
    """A file of one of YOUR results. The folder is built from the session's
    user, so another user's link cannot reach their data; the resolved path
    must stay inside that folder, so '../' cannot leave it."""
    base = out_dir(user["id"], checked(study), checked(model)).resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(target, headers={"Cache-Control": "private, no-cache"})


# ── uploads ───────────────────────────────────────────────────────────────
def temp_upload_dir(user: dict, temp_id: str) -> Path:
    """The staging folder of an upload; only ids this server issued are accepted."""
    if not checked(temp_id).startswith("temp_"):
        raise HTTPException(status_code=400, detail="Not an upload id.")
    return tmp_dir(user["id"], temp_id)


def folder_size(root: Path) -> int:
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) if root.exists() else 0


@app.post("/studies")
async def upload_study(files: List[UploadFile] = File(...), custom_name: str = Form(None),
                       user: dict = Depends(current_user)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    # Validate the name before anything is written: it becomes raw/<name>.
    custom_name = checked(custom_name) if custom_name and custom_name.strip() else None

    temp_id = f"temp_{uuid.uuid4().hex}"
    temp_dir = tmp_dir(user["id"], temp_id)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        original_parents = {}
        for f in files:
            file_name = Path(f.filename).name
            file_path = temp_dir / file_name
            original_parents[file_name] = str(Path(f.filename).parent)
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(f.file, buffer)

        if folder_size(user_root(user["id"])) > USER_QUOTA_BYTES:
            raise HTTPException(status_code=413, detail=(
                f"Storage limit reached ({USER_QUOTA_BYTES // 1024**3} GB). Delete old studies first."))

        fix_extensions(temp_dir)

        invalid_files = []
        dicom_dirs = set()
        for f_path in temp_dir.iterdir():
            if f_path.is_file():
                lower_name = f_path.name.lower()
                is_valid = lower_name.endswith(".dcm") or lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")

                original_name = f_path.name
                if original_name not in original_parents and original_name.endswith(".dcm"):
                    original_name = original_name[:-4]

                parent_dir = original_parents.get(original_name, "")

                if not is_valid:
                    invalid_files.append(f_path.name)
                elif lower_name.endswith(".dcm"):
                    dicom_dirs.add(parent_dir)

        if len(dicom_dirs) > 1:
            raise HTTPException(status_code=400, detail="Multiple folders contain DICOM files. Please upload the specific folder.")

        if invalid_files:
            return {"requires_cleaning": True, "temp_id": temp_id, "invalid_files": invalid_files}

        # Get study ID
        if custom_name:
            study_id = custom_name
        else:
            try:
                study_id = study_id_from_series(temp_dir)
            except StopIteration:
                raise HTTPException(status_code=400, detail="No DICOM or NIfTI files found in the upload.")

        final_dir = raw_dir(user["id"], study_id)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.rename(final_dir)

        return {"study_id": study_id, "requires_cleaning": False}
    except HTTPException:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/studies/clean/{temp_id}")
def clean_and_commit_study(temp_id: str, custom_name: str = None, user: dict = Depends(current_user)):
    temp_dir = temp_upload_dir(user, temp_id)
    custom_name = checked(custom_name) if custom_name and custom_name.strip() else None
    if not temp_dir.exists():
        raise HTTPException(status_code=404, detail="Temp directory not found.")

    try:
        for f in temp_dir.iterdir():
            lower_name = f.name.lower()
            is_valid = lower_name.endswith(".dcm") or lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")
            if not is_valid:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)

        if custom_name:
            study_id = custom_name
        else:
            try:
                study_id = study_id_from_series(temp_dir)
            except StopIteration:
                raise HTTPException(status_code=400, detail="No DICOM or NIfTI files remained after cleaning.")

        final_dir = raw_dir(user["id"], study_id)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.rename(final_dir)

        return {"study_id": study_id}
    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/studies/clean/{temp_id}")
def abort_upload(temp_id: str, user: dict = Depends(current_user)):
    temp_dir = temp_upload_dir(user, temp_id)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    return {"status": "aborted"}


# ── listings ──────────────────────────────────────────────────────────────
@app.get("/studies")
def get_studies(user: dict = Depends(current_user)):
    out_root = user_root(user["id"]) / "out"
    if not out_root.exists():
        return []

    results = []
    for d in out_root.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            found = False
            for md in d.iterdir():
                if md.is_dir():
                    results.append({"id": d.name, "model": md.name})
                    found = True
            if not found:
                results.append({"id": d.name, "model": "a1-roi"})
    # Sort results by id as a fallback
    results.sort(key=lambda x: (x["id"], x["model"]))
    return results


@app.get("/models")
def get_models(user: dict = Depends(current_user)):
    # Validated manifests only: a broken one fails loudly instead of being
    # listed. crop is the model's own default; a run may override it.
    models = [{"id": m["id"], "name": m["name"], "crop": m["crop"]} for m in list_models()]
    return sorted(models, key=lambda m: m["id"])


@app.get("/raw_patients")
def get_raw_patients(user: dict = Depends(current_user)):
    raw_root = user_root(user["id"]) / "raw"
    if not raw_root.exists():
        return []

    patients = []
    for d in raw_root.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            uploaded = datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds")
            row = {"id": d.name, "uploaded": uploaded}
            if user["is_admin"]:      # server paths are shown to the admin only
                row["path"] = str(scan_input_dir(user["id"], d.name).absolute())
            patients.append(row)

    patients.sort(key=lambda x: x["uploaded"], reverse=True)   # newest upload first
    return patients


# ── runs ──────────────────────────────────────────────────────────────────
class JobRequest(BaseModel):
    scan: str | None = None         # one of YOUR uploads: the server finds its folder
    input_path: str | None = None   # any folder on the server: admin only
    study_id: str | None = None     # result name; defaults to the scan's name
    model_id: str
    crop: bool | None = None        # None = the model's manifest default


# One run at a time: two TotalSegmentator / nnUNet runs would compete for the
# same GPU memory. The lock makes check-and-start a single step.
JOBS_LOCK = threading.Lock()


@app.post("/jobs")
def start_job(req: JobRequest, user: dict = Depends(current_user)):
    uid = user["id"]
    if req.input_path:
        if not user["is_admin"]:
            raise HTTPException(status_code=403, detail="Only the administrator can run on a server path.")
        input_dir = Path(req.input_path)
        if not input_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"Folder not found on the server: {req.input_path}")
        # An uploaded scan is raw/<id>[/<series>]; anywhere else the parent
        # folder is usually the patient ID.
        own_raw = (user_root(uid) / "raw").resolve()
        default_name = input_dir.name if input_dir.parent.resolve() == own_raw else input_dir.parent.name
    elif req.scan:
        scan = checked(req.scan)
        if not raw_dir(uid, scan).is_dir():
            raise HTTPException(status_code=404, detail=f"No uploaded scan named {scan!r}.")
        input_dir = scan_input_dir(uid, scan)
        default_name = scan
    else:
        raise HTTPException(status_code=400, detail="Choose an uploaded scan to run.")
    study_id = checked(req.study_id or default_name)   # becomes out/<study_id>
    model_id = checked(req.model_id)

    job_id = uuid.uuid4().hex[:12]

    with JOBS_LOCK:
        busy = [j for j in JOBS.values() if j["status"] == "running"]
        if busy:
            mine = busy[0]["user_id"] == uid
            raise HTTPException(status_code=409, detail=(
                f"A run is already in progress ({busy[0]['study_id']} · {busy[0]['model_id']}). "
                "Start the next one when it finishes." if mine else
                "Another user's run is in progress. Try again in a few minutes."))
        JOBS[job_id] = {
            "job_id": job_id,
            "user_id": uid,
            "study_id": study_id,
            "model_id": model_id,
            "crop": req.crop,
            "started": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
            "stage": "started",
            "pct": 0.0,
            "error": None
        }

    def work():
        job = JOBS[job_id]
        import subprocess, sys
        from collections import deque
        tail = deque(maxlen=20)   # last output lines, reported if the run fails
        try:
            cmd = [sys.executable, "-m", "src.backend.run", "--user", str(uid), "--model", model_id,
                   "--input", str(input_dir), "--study", study_id]
            if req.crop is not None:
                cmd.append("--crop" if req.crop else "--no-crop")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("[") and "%]" in line:
                    try:
                        pct_str, rest = line.split("%]", 1)
                        pct_val = float(pct_str.strip().strip("[")) / 100.0
                        stage_name = rest.strip()
                        job["pct"] = pct_val
                        job["stage"] = stage_name
                    except:
                        pass
                else:
                    tail.append(line)

            process.wait()
            if process.returncode == 0:
                job["status"] = "done"
                job["pct"] = 1.0
                job["stage"] = "done"
            else:
                job["status"] = "failed"
                # The traceback's last line names the actual problem.
                job["error"] = "\n".join(tail) or f"Process exited with code {process.returncode}"

        except Exception as e:
            job["status"] = "failed"
            job["error"] = traceback.format_exc()

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.get("/jobs")
def list_jobs(user: dict = Depends(current_user)):
    """YOUR runs since this server started, newest first. The UI reads this on
    every page load, so a run survives reloads and navigation."""
    mine = [j for j in JOBS.values() if j["user_id"] == user["id"]]
    return sorted(mine, key=lambda j: j["started"], reverse=True)


@app.get("/jobs/{job_id}")
def get_job(job_id: str, user: dict = Depends(current_user)):
    job = JOBS.get(job_id)
    if not job or job["user_id"] != user["id"]:      # someone else's job does not exist for you
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ── delete ────────────────────────────────────────────────────────────────
@app.delete("/studies/{study_id}")
def delete_study(study_id: str, model: List[str] = Query(default=[]),
                 everything: bool = Query(False, alias="all"), user: dict = Depends(current_user)):
    """Delete some results of one of YOUR studies (?model=a&model=b), or
    everything it owns (?all=true): uploaded scan, prep cache and all results."""
    uid = user["id"]
    study_id = checked(study_id)
    if any(j["status"] == "running" and j["user_id"] == uid and j["study_id"] == study_id
           for j in JOBS.values()):
        raise HTTPException(status_code=409, detail=f"A run for {study_id} is in progress.")

    if everything:
        targets = study_dirs(uid, study_id)
    elif model:
        targets = [out_dir(uid, study_id, checked(m)) for m in model]
    else:
        raise HTTPException(status_code=400, detail="Say what to delete: ?model=<id> or ?all=true")

    existing = [p for p in targets if p.exists()]
    if not existing:
        raise HTTPException(status_code=404, detail=f"Nothing to delete for {study_id}.")
    try:
        for p in existing:
            shutil.rmtree(p)
        # a study whose last result was deleted should not remain as an empty folder
        results = user_root(uid) / "out" / study_id
        if results.exists() and not any(results.iterdir()):
            results.rmdir()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not delete: {e}")
    log("delete", username=user["username"], study=study_id, scope="all" if everything else ",".join(model))
    return {"deleted": [str(p.relative_to(user_root(uid))) for p in existing]}


# Mount static files. Only the UI code is public; data/ is never mounted —
# result files go through /files/..., which checks the session.
ui_dir = Path("ui")

if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")


@app.get("/")
def read_root():
    return RedirectResponse(url="/ui/index.html")


if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    args, _ = parser.parse_known_args()
    
    if not accounts.access_hashes():
        print("\n  NOTE: PREDICT_ACCESS_TOKENS is not set, so only the admin can sign in.")
    print(f"\n  PrediCT Studio -> http://127.0.0.1:{args.port}\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
