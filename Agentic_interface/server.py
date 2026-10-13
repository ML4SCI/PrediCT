import importlib.util
import os
import sys
import shutil
import tempfile
import zipfile
import gc
from pathlib import Path

import SimpleITK as sitk
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

# 1. Fix ModuleNotFoundError: Add project root and subfolders to sys.path
PROJECT_ROOT = Path(r"E:\MyProjects\Gsoc_2026_Official")
for p in [
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "Segmentation_Rajat"),
    str(PROJECT_ROOT / "nnunet_dob_scv"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)


AGATSTON_SCRIPT_PATH = Path(r"E:\MyProjects\Gsoc_2026_Official\Segmentation_Rajat\agatston_script.py")


def _load_agatston_function(script_path: Path):
    """Dynamically imports calculate_agatston() from agatston_script.py
    without requiring it to be installed as a package."""
    if not script_path.exists():
        print(f"  [warn] Agatston script not found at {script_path}; "
              f"GT/Pred Agatston scores will show N/A")
        return None
    try:
        spec = importlib.util.spec_from_file_location("agatston_script", script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["agatston_script"] = module
        spec.loader.exec_module(module)
        return module.calculate_agatston
    except Exception as e:
        print(f"  [warn] Failed to import calculate_agatston from {script_path}: {e}")
        return None


_calculate_agatston = _load_agatston_function(AGATSTON_SCRIPT_PATH)


app = FastAPI(title="Medical Segmentation API")

# Allow requests from Katy's browser / frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
RERONET_CKPT = str(PROJECT_ROOT / "Segmentation_Rajat" / "runs" / "run_no_coord" / "ckpts" / "best_dice.pth")
# Removed OUTPUT_BASE_DIR to ensure no permanent storage of medical data on server



def cleanup_vram():
    """Aggressively clear VRAM and system memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass

def cleanup_temp_dir(temp_dir: str):
    """Removes the temporary directory after the response is sent."""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error cleaning up temp dir {temp_dir}: {e}")


def extract_dicom_zip(upload_file: UploadFile, input_dir: Path):
    """Saves an uploaded .zip (or single .dcm) and extracts all DICOM slices into input_dir."""
    safe_name = Path(upload_file.filename).name
    temp_file_path = input_dir / safe_name

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    # If it's a .zip file of a DICOM folder, extract all files flatly into input_dir
    if safe_name.lower().endswith(".zip"):
        with zipfile.ZipFile(temp_file_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                if not member.endswith("/"):  # Skip directory entries
                    member_name = Path(member).name
                    if member_name and not member_name.startswith("."):
                        with zip_ref.open(member) as source, open(input_dir / member_name, "wb") as target:
                            shutil.copyfileobj(source, target)
        temp_file_path.unlink()  # Remove the .zip file so only .dcm files remain in input_dir


def read_dicom_folder(dicom_dir: Path, output_nifti_path: Path):
    """
    Reads a DICOM series from a directory and saves it as a NIfTI file.
    Same logic as nnunet_dob_scv/predict.py, so the viewer scan matches the mask's space.
    """
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in directory: {dicom_dir}")

    best_series_id = max(
        series_ids,
        key=lambda sid: len(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), sid)),
    )
    dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), best_series_id)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_nifti_path))


def create_download_response(result_path: str | Path) -> FileResponse:
    """Forces Swagger UI and browsers to show a 'Download file' button instead of printing raw binary."""
    res_path = Path(result_path)
    return FileResponse(
        path=str(res_path),
        filename=res_path.name,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{res_path.name}"'}
    )


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Medical Segmentation API is running",
        "swagger_ui": "http://localhost:8000/docs",
        "app": "http://localhost:8000/app",
    }


@app.get("/app", response_class=HTMLResponse)
async def viewer():
    return HTMLResponse(VIEWER_HTML)


@app.post("/convert/dicom")
async def convert_dicom(background: BackgroundTasks, dicom_zip: UploadFile = File(..., description="Upload a .zip file containing your DICOM (.dcm) slices")):
    """Display-only helper: DICOM zip -> NIfTI (same SimpleITK logic as predict.py). Nothing is kept on the server."""
    temp_dir = tempfile.mkdtemp()
    try:
        input_dir = Path(temp_dir) / "input"
        input_dir.mkdir()
        extract_dicom_zip(dicom_zip, input_dir)

        out_path = Path(temp_dir) / "scan.nii.gz"
        read_dicom_folder(input_dir, out_path)

        background.add_task(cleanup_temp_dir, temp_dir)
        return create_download_response(out_path)
    except Exception as e:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agatston")
async def agatston(ct: UploadFile = File(..., description="CT scan as .nii / .nii.gz"),
                   mask: UploadFile = File(..., description="Predicted calcium mask as .nii / .nii.gz")):
    """Runs calculate_agatston() from agatston_script.py on a CT + mask pair. Nothing is kept on the server."""
    if _calculate_agatston is None:
        raise HTTPException(status_code=503, detail=f"Agatston script could not be loaded from {AGATSTON_SCRIPT_PATH} (see server console).")

    def suffix(name: str) -> str:
        return ".nii.gz" if (name or "").lower().endswith(".gz") else ".nii"

    temp_dir = tempfile.mkdtemp()
    try:
        ct_path = Path(temp_dir) / ("ct" + suffix(ct.filename))
        mask_path = Path(temp_dir) / ("mask" + suffix(mask.filename))
        for src, dst in ((ct, ct_path), (mask, mask_path)):
            with open(dst, "wb") as buffer:
                shutil.copyfileobj(src.file, buffer)
        return _calculate_agatston(str(ct_path), str(mask_path))
    except ValueError as e:  # e.g. CT / mask shape mismatch
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_dir(temp_dir)


@app.post("/reronet/nifti")
async def reronet_nifti(background: BackgroundTasks, file: UploadFile = File(..., description="Upload a .nii or .nii.gz file")):
    temp_dir = tempfile.mkdtemp()
    try:
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        file_path = input_dir / Path(file.filename).name
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from Segmentation_Rajat.wrapper import predict as reronet_predict
        result_path = reronet_predict(
            mode="nii",
            input_path=str(file_path),
            output_dir=str(output_dir),
            riunet_ckpt_path=RERONET_CKPT
        )

        background.add_task(cleanup_temp_dir, temp_dir)
        return create_download_response(result_path)
    except Exception as e:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_vram()


@app.post("/reronet/dicom")
async def reronet_dicom(background: BackgroundTasks, dicom_zip: UploadFile = File(..., description="Upload a .zip file containing your DICOM (.dcm) slices")):
    temp_dir = tempfile.mkdtemp()
    try:
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Extract the uploaded .zip of DICOM slices into input_dir
        extract_dicom_zip(dicom_zip, input_dir)

        from Segmentation_Rajat.wrapper import predict as reronet_predict
        result_path = reronet_predict(
            mode="dicom",
            input_path=str(input_dir),
            output_dir=str(output_dir),
            riunet_ckpt_path=RERONET_CKPT
        )

        background.add_task(cleanup_temp_dir, temp_dir)
        return create_download_response(result_path)
    except Exception as e:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_vram()


@app.post("/nnunet/nifti")
async def nnunet_nifti(background: BackgroundTasks, file: UploadFile = File(..., description="Upload a .nii or .nii.gz file")):
    temp_dir = tempfile.mkdtemp()
    try:
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        file_path = input_dir / Path(file.filename).name
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from nnunet_dob_scv.predict import predict as nnunet_predict
        nnunet_predict("nii", str(file_path), str(output_dir))

        predictions = list(output_dir.glob("*.nii.gz"))
        if not predictions:
            raise RuntimeError("nnU-Net did not produce a prediction file.")
        result_path = predictions[0]

        background.add_task(cleanup_temp_dir, temp_dir)
        return create_download_response(result_path)
    except Exception as e:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_vram()


@app.post("/nnunet/dicom")
async def nnunet_dicom(background: BackgroundTasks, dicom_zip: UploadFile = File(..., description="Upload a .zip file containing your DICOM (.dcm) slices")):
    temp_dir = tempfile.mkdtemp()
    try:
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Extract the uploaded .zip of DICOM slices into input_dir
        extract_dicom_zip(dicom_zip, input_dir)

        from nnunet_dob_scv.predict import predict as nnunet_predict
        nnunet_predict("dicom", str(input_dir), str(output_dir))

        predictions = list(output_dir.glob("*.nii.gz"))
        if not predictions:
            raise RuntimeError("nnU-Net did not produce a prediction file.")
        result_path = predictions[0]

        background.add_task(cleanup_temp_dir, temp_dir)
        return create_download_response(result_path)
    except Exception as e:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_vram()


# ---------------------------------------------------------------------------
# Embedded viewer (served at /app). NiiVue + JSZip are loaded from a CDN, so the
# *browser* needs internet access; no scan data is sent to any third party.
# ---------------------------------------------------------------------------
VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Segmentation Viewer</title>
<style>
  :root {
    --bg: #0c0e11; --panel: #14181d; --panel2: #1a1f26; --line: #262d36;
    --text: #e6e9ee; --muted: #8a94a3; --accent: #3ddc97; --accent-ink: #06231a;
    --danger: #ff6b6b;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg); color: var(--text);
    font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    display: grid; grid-template-columns: 320px 1fr; height: 100vh;
  }
  aside {
    background: var(--panel); border-right: 1px solid var(--line);
    overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 18px;
  }
  h1 { font-size: 15px; margin: 0; letter-spacing: .02em; display: flex; align-items: center; gap: 8px; }
  h1 i { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); display: inline-block; }
  section { display: flex; flex-direction: column; gap: 8px; }
  .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
  .seg { display: grid; grid-auto-flow: column; grid-auto-columns: 1fr; background: var(--panel2);
         border: 1px solid var(--line); border-radius: 8px; padding: 3px; gap: 3px; }
  .seg button { background: none; border: 0; color: var(--muted); padding: 7px 4px; border-radius: 6px;
                cursor: pointer; font: inherit; }
  .seg button.on { background: var(--accent); color: var(--accent-ink); font-weight: 600; }
  .btn { background: var(--panel2); color: var(--text); border: 1px solid var(--line); border-radius: 8px;
         padding: 9px 10px; cursor: pointer; font: inherit; text-align: center; }
  .btn:hover { border-color: #3a4552; }
  .btn.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); font-weight: 600; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .slider { display: grid; grid-template-columns: 62px 1fr 52px; align-items: center; gap: 8px; color: var(--muted); }
  .slider output { text-align: right; color: var(--text); font-variant-numeric: tabular-nums; font-size: 12px; }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  select { background: var(--panel2); color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 7px; font: inherit; }
  label.chk { display: flex; align-items: center; gap: 8px; color: var(--text); cursor: pointer; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chips button { background: var(--panel2); color: var(--text); border: 1px solid var(--line); border-radius: 999px;
                  padding: 4px 10px; cursor: pointer; font: inherit; font-size: 12px; }
  .chips button:hover { border-color: var(--accent); }
  #status { font-size: 12px; color: var(--muted); min-height: 32px; }
  #status.err { color: var(--danger); }
  #bar { height: 3px; background: var(--line); border-radius: 2px; overflow: hidden; display: none; }
  #bar.on { display: block; }
  #bar b { display: block; height: 100%; width: 35%; background: var(--accent); animation: slide 1.1s infinite ease-in-out; }
  @keyframes slide { 0% { margin-left: -35%; } 100% { margin-left: 100%; } }
  main { position: relative; min-width: 0; min-height: 0; display: flex; flex-direction: column; }
  #stage { position: relative; flex: 1; min-height: 0; }
  #gl { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
  #ov { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
  .pane { position: absolute; box-sizing: border-box; }
  .pane.b { border: 1.5px solid rgba(61, 220, 151, .55); }
  .pane span { position: absolute; left: 6px; top: 6px; padding: 2px 8px; border-radius: 999px;
               background: rgba(12, 14, 17, .72); color: var(--text); font-size: 11px; letter-spacing: .06em;
               text-transform: uppercase; border: 1px solid rgba(255, 255, 255, .14); }
  .card { background: var(--panel2); border: 1px solid var(--line); border-radius: 10px; padding: 12px; display: flex; flex-direction: column; gap: 8px; }
  #agTotal { font-size: 30px; font-weight: 650; line-height: 1; font-variant-numeric: tabular-nums; }
  #agCat { align-self: flex-start; font-size: 11px; letter-spacing: .06em; text-transform: uppercase; padding: 2px 9px;
           border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
  #agStats { display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: 12.5px; }
  #agStats b { color: var(--text); font-weight: 500; }
  #agat details summary { cursor: pointer; color: var(--muted); font-size: 12.5px; }
  #agList { max-height: 160px; overflow-y: auto; margin-top: 6px; font-size: 12px; font-variant-numeric: tabular-nums; color: var(--muted); }
  #agList div { display: flex; justify-content: space-between; padding: 1px 2px; }
  #agList b { color: var(--text); font-weight: 500; }
  #agErr { color: var(--danger); font-size: 12.5px; }
  #agNote { color: var(--muted); font-size: 11px; line-height: 1.4; }
  #hint { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
          text-align: center; color: var(--muted); pointer-events: none; padding: 24px; }
  #hint div { border: 1.5px dashed #34404d; border-radius: 14px; padding: 34px 40px; line-height: 1.7; }
  #hint b { color: var(--text); }
  #readout { background: var(--panel); border-top: 1px solid var(--line); padding: 8px 14px; color: var(--muted);
             font-variant-numeric: tabular-nums; font-size: 12.5px; display: flex; gap: 22px; flex-wrap: wrap; min-height: 34px; }
  #readout span b { color: var(--text); font-weight: 500; }
  body.drag #stage { outline: 2px solid var(--accent); outline-offset: -2px; }
  .hide { display: none !important; }
  @media (max-width: 800px) {
    body { grid-template-columns: 1fr; grid-template-rows: auto 1fr; height: auto; min-height: 100vh; }
    aside { border-right: 0; border-bottom: 1px solid var(--line); }
    main { height: 70vh; }
  }
</style>
</head>
<body>
<aside>
  <h1><i></i> Segmentation Viewer</h1>

  <section>
    <div class="lbl">1 · Scan</div>
    <div class="row">
      <button class="btn" id="pickFolder">DICOM folder…</button>
      <button class="btn" id="pickFiles">NIfTI / ZIP…</button>
    </div>
    <input type="file" id="inFolder" webkitdirectory directory multiple hidden>
    <input type="file" id="inFiles" accept=".nii,.nii.gz,.gz,.zip,.dcm" multiple hidden>
    <div class="lbl" style="text-transform:none;letter-spacing:0">…or drag &amp; drop a folder, .zip, .nii or .nii.gz anywhere.</div>
  </section>

  <section>
    <div class="lbl">2 · Model</div>
    <div class="seg" id="modelSeg">
      <button data-v="reronet" class="on">ReRoNet</button>
      <button data-v="nnunet">nnU-Net</button>
    </div>
    <button class="btn primary" id="run" disabled>Run segmentation</button>
    <div id="bar"><b></b></div>
    <div id="status">Load a scan to begin.</div>
  </section>

  <section id="agat" class="hide">
    <div class="lbl">Agatston score</div>
    <div class="card">
      <div id="agErr" class="hide"></div>
      <div id="agBody">
        <div id="agTotal">–</div>
        <div id="agCat">–</div>
        <div id="agStats"><span>Lesions <b id="agLes">–</b></span><span>Slices with calcium <b id="agSl">–</b></span></div>
        <details><summary>Per-slice scores</summary><div id="agList"></div></details>
        <div id="agNote">Computed from the predicted mask with your agatston_script.py. Slice numbers follow its RAS-oriented order. Risk bands (1–10 minimal, 11–100 mild, 101–400 moderate, &gt;400 severe) are reference only, not a diagnosis.</div>
      </div>
    </div>
  </section>

  <section>
    <div class="lbl">View</div>
    <div class="seg" id="viewSeg">
      <button data-v="multi" class="on">2×2</button>
      <button data-v="ax">Axial</button>
      <button data-v="cor">Coronal</button>
      <button data-v="sag">Sagittal</button>
      <button data-v="3d">3D</button>
    </div>
    <label class="chk"><input type="checkbox" id="bord" checked> Pane borders</label>
    <label class="chk"><input type="checkbox" id="lab" checked> Pane headings</label>
    <label class="chk"><input type="checkbox" id="chOn" checked> Slice crosshair lines</label>
    <div class="slider"><span>Thickness</span><input type="range" id="chW" min="1" max="10" step="0.5" value="1"><output id="chWV">1</output></div>
  </section>

  <section>
    <div class="lbl">Window / level</div>
    <div class="slider"><span>Window</span><input type="range" id="win"><output id="winV">–</output></div>
    <div class="slider"><span>Level</span><input type="range" id="lvl"><output id="lvlV">–</output></div>
    <div class="chips" id="presets">
      <button data-p="auto">Auto</button>
      <button data-p="-160,240">Cardiac (W400 L40)</button>
      <button data-p="0,500">Calcium (0–500 HU)</button>
      <button data-p="-200,800">Wide (W1000 L300)</button>
    </div>
  </section>

  <section>
    <div class="lbl">Mask overlay</div>
    <label class="chk"><input type="checkbox" id="maskOn" checked> Show mask</label>
    <div class="slider"><span>Opacity</span><input type="range" id="op" min="0" max="1" step="0.05" value="0.55"><output id="opV">0.55</output></div>
    <div class="slider"><span>Colour</span>
      <select id="cmap" style="grid-column: span 2">
        <option value="red">Red</option><option value="green">Green</option><option value="blue">Blue</option>
        <option value="warm">Warm</option><option value="winter">Winter</option><option value="hot">Hot</option>
      </select>
    </div>
    <button class="btn" id="dl" disabled>Download mask (.nii.gz)</button>
  </section>
</aside>

<main>
  <div id="stage">
    <canvas id="gl"></canvas>
    <div id="ov"></div>
    <div id="hint"><div><b>Drop a DICOM folder, .zip or NIfTI here</b><br>The scan appears instantly; press <b>Run segmentation</b> to overlay the predicted mask.</div></div>
  </div>
  <div id="readout"><span>Voxel <b id="rVox">–</b></span><span>mm <b id="rMm">–</b></span><span>Scan <b id="rScan">–</b></span><span>Mask <b id="rMask">–</b></span></div>
</main>

<script src="https://cdn.jsdelivr.net/npm/jszip@3.10.2/dist/jszip.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@niivue/niivue@0.69.0/dist/niivue.umd.js"></script>
<script>
const $ = id => document.getElementById(id);

const nv = new niivue.Niivue({
  backColor: [0, 0, 0, 1],
  isColorbar: false,
  show3Dcrosshair: true,
  crosshairColor: [0.24, 0.86, 0.59, 1],
  multiplanarShowRender: niivue.SHOW_RENDER.ALWAYS,
  multiplanarLayout: niivue.MULTIPLANAR_TYPE.GRID,
  dragAndDropEnabled: false,   // we handle drops ourselves (folders, auto-zip)
});
nv.attachToCanvas($('gl'));

const S = {
  kind: null,          // 'nifti' | 'dicom'
  niftiFile: null,     // original File when kind === 'nifti'
  zipBlob: null,       // zipped DICOM when kind === 'dicom'
  scanUrl: null, maskUrl: null,
  scanBlob: null, scanName: 'scan.nii.gz',    // the NIfTI shown in the viewer (original or DICOM-converted)
  maskBlob: null, maskName: 'mask.nii.gz',
  model: 'reronet', busy: false,
  autoWin: null, range: [0, 1],
};

/* ---------- status / progress ---------- */
let timer = null;
function setStatus(t, err) { $('status').textContent = t; $('status').classList.toggle('err', !!err); }
function startTimer(label) {
  stopTimer(); const t0 = Date.now(); $('bar').classList.add('on');
  const tick = () => setStatus(`${label}… ${Math.round((Date.now() - t0) / 1000)}s`);
  tick(); timer = setInterval(tick, 500);
}
function stopTimer() { clearInterval(timer); timer = null; $('bar').classList.remove('on'); }
function setBusy(b) { S.busy = b; $('run').disabled = b || !S.kind; }

/* ---------- networking ---------- */
function filenameFrom(cd, fallback) {
  const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd || '');
  return m ? decodeURIComponent(m[1]) : fallback;
}
function post(url, fd, label) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open('POST', url); x.responseType = 'blob';
    $('bar').classList.add('on');
    x.upload.onprogress = e => { if (e.lengthComputable) setStatus(`${label}: uploading ${Math.round(100 * e.loaded / e.total)}%`); };
    x.upload.onload = () => startTimer(label);
    x.onload = async () => {
      stopTimer();
      if (x.status >= 200 && x.status < 300) {
        resolve({ blob: x.response, name: filenameFrom(x.getResponseHeader('Content-Disposition'), 'result.nii.gz') });
      } else {
        let msg = 'HTTP ' + x.status;
        try { const t = await x.response.text(); msg = (JSON.parse(t).detail) || t; } catch (e) {}
        reject(new Error(msg));
      }
    };
    x.onerror = () => { stopTimer(); reject(new Error('Network error – is the server reachable?')); };
    x.send(fd);
  });
}

/* ---------- ingest files / folders ---------- */
const JUNK = /^(\.|thumbs\.db$|dicomdir$)/i;
async function filesFromDrop(dt) {
  const entries = [...dt.items].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);  // must be sync
  if (!entries.length) return [...dt.files];
  const out = [];
  async function walk(e) {
    if (e.isFile) out.push(await new Promise((res, rej) => e.file(res, rej)));
    else if (e.isDirectory) {
      const r = e.createReader(); let batch;
      do { batch = await new Promise((res, rej) => r.readEntries(res, rej)); for (const c of batch) await walk(c); } while (batch.length);
    }
  }
  for (const e of entries) await walk(e);
  return out;
}

async function ingest(files) {
  if (S.busy) return;
  files = files.filter(f => !JUNK.test(f.name));
  if (!files.length) return;
  setBusy(true);
  try {
    const isNii = f => /\.nii(\.gz)?$/i.test(f.name);
    let niftiUrl, niftiName;
    S.niftiFile = null; S.zipBlob = null;

    if (files.length === 1 && isNii(files[0])) {
      S.kind = 'nifti'; S.niftiFile = files[0];
      S.scanBlob = files[0]; S.scanName = files[0].name;
      niftiUrl = URL.createObjectURL(files[0]); niftiName = files[0].name;
      setStatus('Loading scan…');
    } else {
      S.kind = 'dicom';
      if (files.length === 1 && /\.zip$/i.test(files[0].name)) {
        S.zipBlob = files[0];
      } else {
        setStatus(`Zipping ${files.length} files in your browser…`); $('bar').classList.add('on');
        const zip = new JSZip();
        files.forEach((f, i) => zip.file(`${i}_${f.name}`, f));
        S.zipBlob = await zip.generateAsync({ type: 'blob', compression: 'STORE' });
        $('bar').classList.remove('on');
      }
      const fd = new FormData(); fd.append('dicom_zip', S.zipBlob, 'dicom.zip');
      const r = await post('/convert/dicom', fd, 'Converting DICOM for display');
      S.scanBlob = r.blob; S.scanName = 'scan.nii.gz';
      niftiUrl = URL.createObjectURL(r.blob); niftiName = 'scan.nii.gz';
    }
    await showScan(niftiUrl, niftiName);
    setStatus(`Scan loaded (${S.kind === 'nifti' ? 'NIfTI' : 'DICOM'}). Pick a model and run segmentation.`);
  } catch (e) {
    S.kind = null; setStatus('Could not load scan: ' + e.message, true);
  } finally { setBusy(false); }
}

/* ---------- viewer ---------- */
async function showScan(url, name) {
  if (nv.volumes.length > 1) nv.removeVolumeByIndex(1);
  clearMask();
  if (S.scanUrl) URL.revokeObjectURL(S.scanUrl);
  S.scanUrl = url;
  await nv.loadVolumes([{ url, name }]);
  $('hint').classList.add('hide');
  const v = nv.volumes[0];
  S.autoWin = [v.cal_min, v.cal_max];
  S.range = [v.global_min ?? v.cal_min, v.global_max ?? v.cal_max];
  const span = Math.max(1e-6, S.range[1] - S.range[0]);
  $('win').min = span / 1000; $('win').max = span * 1.5; $('win').step = span / 1000;
  $('lvl').min = S.range[0]; $('lvl').max = S.range[1]; $('lvl').step = span / 1000;
  syncSliders();
  setView($('viewSeg').querySelector('.on').dataset.v);
}

function clearMask() {
  S.maskBlob = null; $('dl').disabled = true;
  if (S.maskUrl) { URL.revokeObjectURL(S.maskUrl); S.maskUrl = null; }
  $('rMask').textContent = '–';
  $('agat').classList.add('hide');
}

/* ---------- Agatston score (server runs your agatston_script.py on CT + mask) ---------- */
function agCategory(s) { return s <= 0 ? 'No calcification' : s <= 10 ? 'Minimal' : s <= 100 ? 'Mild' : s <= 400 ? 'Moderate' : 'Severe'; }
function showAgatstonError(msg) {
  $('agat').classList.remove('hide'); $('agBody').classList.add('hide');
  $('agErr').classList.remove('hide'); $('agErr').textContent = 'Agatston score unavailable: ' + msg;
}
async function scoreAgatston() {
  if (!S.scanBlob || !S.maskBlob) return;
  $('agat').classList.remove('hide'); $('agErr').classList.add('hide'); $('agBody').classList.remove('hide');
  $('agTotal').textContent = '…'; $('agCat').textContent = 'computing'; $('agLes').textContent = $('agSl').textContent = '–';
  $('agList').textContent = '';
  try {
    const fd = new FormData();
    fd.append('ct', S.scanBlob, S.scanName);
    fd.append('mask', S.maskBlob, S.maskName);
    const r = await fetch('/agatston', { method: 'POST', body: fd });
    if (!r.ok) {
      let msg = 'HTTP ' + r.status;
      try { msg = (await r.json()).detail || msg; } catch (e) {}
      throw new Error(msg);
    }
    const a = await r.json();
    $('agTotal').textContent = Number(a.agatston_total).toLocaleString(undefined, { maximumFractionDigits: 2 });
    $('agCat').textContent = agCategory(a.agatston_total);
    $('agLes').textContent = a.n_lesions; $('agSl').textContent = a.n_slices_with_calcium;
    for (const s of a.agatston_per_slice || []) {
      const row = document.createElement('div');
      const l = document.createElement('span'); l.textContent = 'slice ' + s.slice;
      const v = document.createElement('b'); v.textContent = Number(s.score).toFixed(2);
      row.append(l, v); $('agList').appendChild(row);
    }
  } catch (e) { showAgatstonError(e.message); }
}

async function showMask(blob, name) {
  if (nv.volumes.length > 1) nv.removeVolumeByIndex(1);
  if (S.maskUrl) URL.revokeObjectURL(S.maskUrl);
  S.maskBlob = blob; S.maskName = name; S.maskUrl = URL.createObjectURL(blob);
  const img = await niivue.NVImage.loadFromUrl({
    url: S.maskUrl, name: 'mask.nii.gz', colormap: $('cmap').value,
    opacity: $('maskOn').checked ? parseFloat($('op').value) : 0,
  });
  const mx = img.global_max ?? img.cal_max ?? 1;
  img.cal_min = 0.5; img.cal_max = mx > 0.5 ? mx : 1;     // 0 = background stays transparent
  nv.addVolume(img);
  nv.updateGLVolume();
  $('dl').disabled = false;
}

function setView(v) {
  const T = niivue.SLICE_TYPE;
  nv.setSliceType({ multi: T.MULTIPLANAR, ax: T.AXIAL, cor: T.CORONAL, sag: T.SAGITTAL, '3d': T.RENDER }[v]);
}

function applyWindow(lo, hi) {
  const v = nv.volumes[0]; if (!v) return;
  v.cal_min = lo; v.cal_max = hi; nv.updateGLVolume(); syncSliders();
}
function syncSliders() {
  const v = nv.volumes[0]; if (!v) return;
  const w = v.cal_max - v.cal_min, l = (v.cal_max + v.cal_min) / 2;
  $('win').value = w; $('lvl').value = l;
  $('winV').textContent = Math.round(w); $('lvlV').textContent = Math.round(l);
}
nv.onIntensityChange = () => syncSliders();   // right-drag in the viewer also updates the sliders

nv.onLocationChange = d => {
  if (!d) return;
  $('rVox').textContent = d.vox ? [...d.vox].slice(0, 3).map(Math.round).join(', ') : '–';
  $('rMm').textContent = d.mm ? [...d.mm].slice(0, 3).map(n => n.toFixed(1)).join(', ') : '–';
  const f = n => (n === undefined || n === null || Number.isNaN(n)) ? '–' : (Number.isInteger(n) ? n : n.toFixed(2));
  if (d.values && d.values[0]) $('rScan').textContent = f(d.values[0].value);
  $('rMask').textContent = (d.values && d.values[1]) ? f(d.values[1].value) : '–';
};

/* ---------- controls ---------- */
function bindSeg(id, cb) {
  $(id).addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    $(id).querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); cb(b.dataset.v);
  });
}
bindSeg('modelSeg', v => S.model = v);
bindSeg('viewSeg', v => { if (nv.volumes.length) setView(v); });

$('win').addEventListener('input', () => { const w = +$('win').value, l = +$('lvl').value; applyWindow(l - w / 2, l + w / 2); });
$('lvl').addEventListener('input', () => { const w = +$('win').value, l = +$('lvl').value; applyWindow(l - w / 2, l + w / 2); });
$('presets').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b || !nv.volumes.length) return;
  if (b.dataset.p === 'auto') applyWindow(...S.autoWin);
  else applyWindow(...b.dataset.p.split(',').map(Number));
});

function maskOpacity() { return $('maskOn').checked ? parseFloat($('op').value) : 0; }
$('maskOn').addEventListener('change', () => { if (nv.volumes.length > 1) nv.setOpacity(1, maskOpacity()); });
$('op').addEventListener('input', () => { $('opV').textContent = parseFloat($('op').value).toFixed(2); if (nv.volumes.length > 1) nv.setOpacity(1, maskOpacity()); });
$('cmap').addEventListener('change', () => { if (nv.volumes.length > 1) { nv.volumes[1].colormap = $('cmap').value; nv.updateGLVolume(); } });

$('dl').addEventListener('click', () => {
  if (!S.maskBlob) return;
  const a = document.createElement('a'); a.href = URL.createObjectURL(S.maskBlob); a.download = S.maskName;
  document.body.appendChild(a); a.click(); a.remove();
});

$('run').addEventListener('click', async () => {
  if (!S.kind || S.busy) return;
  setBusy(true);
  try {
    const fd = new FormData();
    if (S.kind === 'nifti') fd.append('file', S.niftiFile, S.niftiFile.name);
    else fd.append('dicom_zip', S.zipBlob, 'dicom.zip');
    const label = (S.model === 'reronet' ? 'ReRoNet' : 'nnU-Net') + ' running';
    const r = await post(`/${S.model}/${S.kind}`, fd, label);
    await showMask(r.blob, r.name);
    setStatus(`Done – mask overlaid (${r.name}). Computing Agatston score…`);
    await scoreAgatston();
    setStatus(`Done – mask overlaid (${r.name}).`);
  } catch (e) { setStatus('Segmentation failed: ' + e.message, true); }
  finally { setBusy(false); }
});

/* ---------- pane borders + headings (HTML overlay on NiiVue's tile rects) ---------- */
const PANE_NAMES = { 0: 'Axial', 1: 'Coronal', 2: 'Sagittal', 4: '3D' };
let lastSig = '';
function syncOverlay() {
  const cv = $('gl'), ov = $('ov');
  const tiles = (nv.volumes.length ? nv.screenSlices || [] : []).filter(t => t.axCorSag in PANE_NAMES);
  const showB = $('bord').checked, showL = $('lab').checked;
  const k = cv.clientWidth ? cv.width / cv.clientWidth : 1;           // device px per CSS px
  const sig = [showB, showL, k.toFixed(3), ...tiles.map(t => t.axCorSag + ':' + t.leftTopWidthHeight.map(Math.round).join(','))].join('|');
  if (sig === lastSig) return;
  lastSig = sig; ov.textContent = '';
  for (const t of tiles) {
    const [x, y, w, h] = t.leftTopWidthHeight.map(n => n / k);
    const d = document.createElement('div');
    d.className = 'pane' + (showB ? ' b' : '');
    Object.assign(d.style, { left: x + 'px', top: y + 'px', width: w + 'px', height: h + 'px' });
    if (showL) { const s = document.createElement('span'); s.textContent = PANE_NAMES[t.axCorSag]; d.appendChild(s); }
    ov.appendChild(d);
  }
}
(function overlayLoop() { try { syncOverlay(); } catch (e) {} requestAnimationFrame(overlayLoop); })();

/* ---------- slice crosshair (the lines showing the current slice) ---------- */
function applyCrosshair() {
  const w = parseFloat($('chW').value); $('chWV').textContent = w;
  nv.setCrosshairWidth(w);
  nv.setCrosshairColor([0.24, 0.86, 0.59, $('chOn').checked ? 1 : 0]);   // alpha 0 = hidden
}
$('chW').addEventListener('input', applyCrosshair);
$('chOn').addEventListener('change', applyCrosshair);

/* ---------- pickers + drag & drop ---------- */
$('pickFolder').onclick = () => $('inFolder').click();
$('pickFiles').onclick = () => $('inFiles').click();
$('inFolder').onchange = e => { ingest([...e.target.files]); e.target.value = ''; };
$('inFiles').onchange = e => { ingest([...e.target.files]); e.target.value = ''; };
['dragenter', 'dragover'].forEach(t => window.addEventListener(t, e => { e.preventDefault(); document.body.classList.add('drag'); }));
window.addEventListener('dragleave', e => { if (!e.relatedTarget) document.body.classList.remove('drag'); });
window.addEventListener('drop', async e => {
  e.preventDefault(); document.body.classList.remove('drag');
  ingest(await filesFromDrop(e.dataTransfer));
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Server starting!")
    print("👉 Your Local Docs:  http://localhost:8000/docs")
    print("👉 Viewer:           http://localhost:8000/app\n")
    # host="0.0.0.0" allows Katy to connect over LAN/Tailscale/Ngrok while you use localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)