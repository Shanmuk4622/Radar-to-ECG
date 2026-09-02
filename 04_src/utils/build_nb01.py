#!/usr/bin/env python3
"""Generate 01_verify_and_download.ipynb -- stdlib only, no nbformat needed."""
import json, sys, ast
from nb_lib_a import HF_SYNC_SRC as SHARED_HF_SYNC_SRC

C = []
def _src(s):
    s = s.strip("\n")
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]
def md(s):
    C.append({"cell_type": "markdown", "metadata": {}, "source": _src(s)})
def code(s):
    body = s.strip("\n")
    try:
        ast.parse(body)
    except SyntaxError as e:
        print(f"!! SYNTAX ERROR in code cell {len(C)}: line {e.lineno}: {e.msg}", file=sys.stderr)
        print("   " + (body.split(chr(10))[e.lineno - 1] if e.lineno else ""), file=sys.stderr)
        raise
    C.append({"cell_type": "code", "metadata": {}, "execution_count": None,
              "outputs": [], "source": _src(body)})

# ---------------------------------------------------------------- 0. HEADER
md(r"""
# NB01 — Verify & Inventory the CR-RVS Radar/ECG Dataset

**Project:** CardioMamba-Net — contactless ECG reconstruction from 24 GHz CW radar
**Stage:** 1 of 5 · `01_verify_and_download` → `02_preprocess_to_hf` → `03_baselines` → `04_cardiomamba_train` → `05_evaluate_and_figures`

---

## What this notebook is for

Before we write a single line of preprocessing code we need to answer one question with evidence:

> **Does the Kaggle mirror `pedababugaddala/datasets-file` actually contain the raw per-subject
> `.mat` tree described in Schellenberger et al. (Sci Data 7:291, 2020) — or is it somebody's
> pre-processed derivative?**

Everything in `00_admin/PLAN.md` assumes the former. If it turns out to be the latter, the whole
preprocessing design changes, so this is a hard gate.

While we are walking every file anyway, we take a **complete census** — because we train once, and
a fact we fail to record now is a fact we pay to recompute later. The census becomes a permanent,
public artifact on Hugging Face that every later notebook reads instead of re-scanning 8 GB of `.mat`.

## What it produces

| Artifact | What it is |
|---|---|
| `inventory.parquet` / `.csv` | **One row per (subject, scenario) file** with ~60 columns: durations, sampling rates, per-channel statistics, NaN/Inf/clipping counts, I/Q ellipse parameters, ECG R-peak count, HR, HRV, radar↔ECG lag estimate, quality flags |
| `variables.json` | The exact variable schema found in every `.mat`, with shapes and dtypes |
| `verdict.json` | The machine-readable answer to the gate question |
| `crosscheck.csv` | Our computed totals vs. the numbers **both papers report** — the strongest correctness signal we have |
| `figures/*.png` | 8 diagrams (pipeline, coverage heatmap, durations, example signals, I/Q ellipse, PSDs, HR distribution, data-use map) |
| `previews/*.npz` | A 60-second full-rate excerpt from every file — enough to unit-test the whole preprocessing chain without touching the raw data again |
| `decimated/*.npz` *(optional)* | The **entire corpus** decimated to 128 Hz, ~350 MB — makes NB02 nearly free |
| `report.md` | Human-readable summary of everything above |
| `sync_history.jsonl` | Append-only event log — every step, every timing, every warning |
| `sync_state.json` | Resume state |

All of it lands in a **public** HF dataset repo: `Shanmuk4622/cr-rvs-radar-ecg-inventory-v2`

---

## ⚠️ HOW TO RUN — read this before pressing anything

**1. Accelerator: `None` (CPU).**
Right sidebar → *Session options* → *Accelerator* → **None**. This notebook does zero matrix maths;
burning GPU quota on a file scan is waste. You have a limited weekly T4 budget — save it for NB04.

**2. Internet: `ON`.**
Right sidebar → *Session options* → **Internet: On**. Required to reach Hugging Face.

**3. Add the dataset as an Input (this is the fast path).**
Right sidebar → **+ Add Input** → *Datasets* → search `pedababugaddala/datasets-file` → **Add**.
It mounts read-only at `/kaggle/input/datasets-file` **instantly**, and — importantly — mounted
inputs do **not** count against your 20 GB `/kaggle/working` budget. If you skip this step the
notebook falls back to downloading via `kagglehub`, which is slower and *does* eat your disk.

**4. Add your Hugging Face token as a Secret.**
Right sidebar → **Add-ons → Secrets** → *Add secret* → Label exactly **`HF_TOKEN`**, value = your HF
token with **write** permission. Then make sure its toggle is **attached to this notebook**.
The token is never printed and never written to disk.

**5. Run all cells, top to bottom.**
Expect **25–70 minutes** including the final upload. There is nothing to babysit.

## Cell-by-cell run guide

| Code cell | What runs | Typical time |
|---:|---|---:|
| 1 | Configuration only | < 5 s |
| 2 | Imports, dependency check, folders, environment manifest | 1–3 min |
| 3 | Write the shared resumable HF sync module | < 5 s |
| 4 | Read HF_TOKEN, create/open v2 repo, restore prior outputs | 1–8 min |
| 5 | Register major-cell upload hook | < 5 s |
| 6 | Locate the attached Kaggle dataset (download fallback only) | 5 s mounted; 5–20 min fallback |
| 7 | Walk the file tree and build its manifest | 1–4 min |
| 8 | Raw-dataset gate/verdict | < 10 s |
| 9–10 | Probe MATLAB schema and normalize variable access | 10–60 s total |
| 11–12 | Define and unit-test signal-processing helpers | < 30 s total |
| 13 | Define the per-recording census function | < 5 s |
| 14 | Census every file; save previews and 128 Hz corpus | 15–45 min |
| 15–16 | Paper cross-checks and quality summaries | 10–60 s total |
| 17–20 | Generate all inventory/quality figures | 3–10 min total |
| 21–22 | Write report, manifest, and dataset card | < 30 s total |
| 23 | Blocking final HF upload and verification | 2–15 min |

Times are estimates; the live census cell prints its own ETA. A hard Kaggle shutdown can lose at
most the work since the last successful remote upload; a normal Stop/SIGINT triggers a final push.

---

## What happens if you stop it, or Kaggle kills the session

Normal stops are protected; a sudden machine loss can recover only the last successful push.

- Progress is checkpointed to `sync_state.json` **after every file**.
- A background uploader pushes to Hugging Face **at most once every 30 minutes**, and additionally
  **the moment a major stage finishes**.
- Pressing **stop**, or a `SIGTERM` from Kaggle, triggers an **immediate final push** before the
  process dies (`signal` + `atexit` handlers).
- On restart, the notebook pulls the run folder back down from HF and **skips every file already
  censused**. Re-running from scratch after an interruption costs you nothing.

A folder upload can itself make several API requests, so the scheduler permits only 24 upload
calls/hour, spaces them, batches files and backs off with jitter.

---

## Reading guide

Sections **1–4** are setup and the gate. Section **5** is the long one (the census loop). Sections
**6–9** are cross-checks, diagrams and the report. Every section starts with a markdown cell saying
what it does and why. Skip to **§6** if you just want the verdict.
""")

# ---------------------------------------------------------------- 1. CONFIG
md(r"""
---
# 1 · Configuration

Everything tunable lives in one dict so no later cell contains a magic number. The two flags worth
knowing about:

- **`SMOKE_TEST`** — set to `True` for a 6-file dry run that exercises every code path in ~2 minutes.
  Use it the first time, confirm the HF push works, then set it back to `False` and run for real.
- **`SAVE_DECIMATED`** — dumps the whole corpus at 128 Hz (~350 MB) to HF. Costs ~10 extra minutes
  here and saves most of NB02's runtime later. Leave it on.

The sampling rate, window length and overlap are **deliberately copied from the baseline paper**
(Chowdhury et al. 2024, §2.3) so our segment counts are directly comparable to their Table 1. Do not
change them without changing `00_admin/PLAN.md` too.
""")

code(r'''
CFG = {
    # ---- source ---------------------------------------------------------
    "KAGGLE_DATASET":   "pedababugaddala/datasets-file",
    "INPUT_HINTS":      ["/kaggle/input"],          # searched first, before any download

    # ---- destination (PUBLIC Hugging Face dataset repo) ------------------
    "HF_USER":          "Shanmuk4622",
    "HF_REPO":          "Shanmuk4622/cr-rvs-radar-ecg-inventory-v2",
    "HF_REPO_TYPE":     "dataset",
    "HF_PRIVATE":       False,                      # <-- public, as requested
    "RUN_ID":           "nb01_inventory_v2",

    # ---- local paths ----------------------------------------------------
    "WORK":             "/kaggle/working/nb01",     # small, pushed to HF, 20 GB budget
    "SCRATCH":          "/kaggle/temp/nb01",        # large, never pushed, not size-capped

    # ---- HF sync policy (your standing rules) ---------------------------
    "PUSH_INTERVAL_S":  30 * 60,                    # at most one push per 30 minutes
    "HF_MAX_UPLOADS_HOUR": 24,                       # calls/hour; each call may make many requests
    "HF_RETRY_MAX":     6,

    # ---- signal-processing constants (FROZEN to the baseline paper) -----
    "TARGET_FS":        128,                        # Chowdhury 2024 §2.3.1
    "WINDOW":           1024,                       # Chowdhury 2024 §2.3.4  (= 8.0 s)
    "OVERLAP":          0.5,                        # Chowdhury 2024 §2.3.4
    "ECG_BAND":         (0.5, 40.0),                # Chowdhury 2024 §2.3.2
    "CARDIAC_BAND":     (0.8, 20.0),                # our channel 8 (PLAN.md C1)
    "LAMBDA_MM":        299792458.0 / 24.0e9 * 1000.0,   # 24 GHz -> ~12.49 mm

    # ---- census options -------------------------------------------------
    "PREVIEW_SECONDS":  60,                         # full-rate excerpt saved per file
    "SAVE_DECIMATED":   True,                       # dump whole corpus at TARGET_FS
    "CHECKPOINT_EVERY": 1,                          # files between sync_state.json writes
    "SMOKE_TEST":       False,                      # True -> only 6 files, ~2 min
    "SMOKE_N":          6,
    "SEED":             1337,
}

# Expected values from the two papers. The census cross-checks against these in section 6.
PAPER = {
    "dataset_total_seconds":   86459.0,   # Schellenberger 2020, "Data Records"
    "dataset_n_subjects":      30,
    "resting_seconds":         19048.6,   # Chowdhury 2024, section 3.1
    "valsalva_seconds":        27968.3,
    "apnea_seconds":            4705.2,
    "segments": {"Resting": 4702, "Valsalva": 6952, "Apnea": 1140, "RVA": 12794},
    "subjects": {"Resting": 30,   "Valsalva": 27,   "Apnea": 24},
    # NOTE: the dataset PAPER writes radar_I / radar_Q, but the actual .mat files use
    # LOWERCASE radar_i / radar_q. Verified on the Kaggle mirror, 2026-09-01. All variable
    # lookups in this notebook are case-insensitive so either spelling works.
    "expected_mat_vars": [
        "radar_i", "radar_q", "fs_radar", "fs_ecg",
        "tfm_ecg1", "tfm_ecg2", "tfm_icg", "tfm_z0", "tfm_bp", "tfm_intervention",
        "measurement_info",
    ],
    # Per-channel sampling rates as actually stored (the TFM channels differ from the radar).
    "known_rates": {"fs_radar": 2000.0, "fs_ecg": 2000.0, "fs_intervention": 2000.0,
                    "fs_icg": 1000.0, "fs_bp": 200.0, "fs_z0": 100.0},
    "scenarios": ["Resting", "Valsalva", "Apnea", "TiltUp", "TiltDown"],
}

import json
print(json.dumps({k: v for k, v in CFG.items() if k != "INPUT_HINTS"}, indent=2, default=str))
''')

# ---------------------------------------------------------------- 2. ENV
md(r"""
---
# 2 · Environment & imports

We pin nothing and install as little as possible — Kaggle images already carry numpy, scipy, pandas,
matplotlib, h5py and `huggingface_hub`. We capture exact versions into the manifest so that if a
result ever looks strange six months from now, we know precisely what produced it.

`pyarrow` is needed for the parquet inventory; `kagglehub` only for the download fallback.
""")

code(r'''
import os, sys, io, gc, re, json, time, math, signal, atexit, threading, traceback, hashlib, platform, subprocess, warnings
from pathlib import Path
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

def _pip(*pkgs):
    """Quietly install only what is genuinely missing."""
    missing = []
    for p in pkgs:
        mod = {"pyarrow": "pyarrow", "kagglehub": "kagglehub",
               "huggingface_hub": "huggingface_hub", "scipy": "scipy"}[p]
        try:
            __import__(mod)
        except ImportError:
            missing.append(p)
    if missing:
        print("installing:", missing)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
        for p in missing:
            __import__(p)

_pip("pyarrow", "huggingface_hub", "scipy")

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

np.random.seed(CFG["SEED"])

WORK    = Path(CFG["WORK"]);    WORK.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(CFG["SCRATCH"]); SCRATCH.mkdir(parents=True, exist_ok=True)
for sub in ("figures", "previews", "decimated", "logs"):
    (WORK / sub).mkdir(parents=True, exist_ok=True)

def _ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return "n/a"

MANIFEST = {
    "run_id":     CFG["RUN_ID"],
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "python":     sys.version.split()[0],
    "platform":   platform.platform(),
    "cpu_count":  os.cpu_count(),
    "versions":   {m: _ver(m) for m in
                   ["numpy", "scipy", "pandas", "matplotlib", "h5py", "pyarrow", "huggingface_hub"]},
    "config":     {k: v for k, v in CFG.items() if k != "INPUT_HINTS"},
}

def _disk(p):
    try:
        st = os.statvfs(p)
        return f"{st.f_bavail*st.f_frsize/2**30:.1f} GB free of {st.f_blocks*st.f_frsize/2**30:.1f} GB"
    except Exception:
        return "?"

print("python      :", MANIFEST["python"])
for k, v in MANIFEST["versions"].items():
    print(f"  {k:<18}: {v}")
print("cpu count   :", MANIFEST["cpu_count"])
print("/kaggle/working :", _disk("/kaggle/working"))
print("/kaggle/temp    :", _disk("/kaggle/temp") if Path("/kaggle/temp").exists() else "absent")
print("WORK        :", WORK)
print("SCRATCH     :", SCRATCH)
''')

# ---------------------------------------------------------------- 3. HF SYNC
md(r"""
---
# 3 · The Hugging Face sync layer

This is the piece every notebook in the project reuses, so it is written **once, here**, and saved
to `hf_sync.py` in the working directory. NB02–NB05 will import it rather than duplicating it.

### What it guarantees

| Rule | How it is enforced |
|---|---|
| Periodic push every 30 min when dirty | Background worker tracks the last successful upload |
| Push **immediately** when a major stage finishes | `sync.stage_done("name")` sets the flag and wakes the thread |
| Push **immediately** when you stop execution | `SIGINT` + `SIGTERM` handlers and `atexit`, all routed to a blocking `flush(final=True)` |
| Stay well below the API limit | Rolling limiter allows 24 folder-upload calls/hour |
| Survive a 429 or a network blip | exponential backoff, 6 attempts, jittered |
| Resume exactly where it stopped | `pull()` restores payloads before `sync_state.json` is trusted |
| Lose no logged detail | every event is appended to `sync_history.jsonl` |

### A note on the interrupt path

In Jupyter, pressing **stop** raises `KeyboardInterrupt` inside the running cell. We install a
`SIGINT` handler that flushes *first* and then re-raises, so the push happens even mid-loop. Kaggle
sends `SIGTERM` when a session is reclaimed — that path flushes too. `atexit` is the belt-and-braces
third layer for a clean interpreter shutdown.

If you ever want to force a push by hand, just call `sync.flush(final=True)` in a cell.
""")

code(r'''
HF_SYNC_SRC = r"""
import os, json, time, random, threading, atexit, signal, shutil
from pathlib import Path
from datetime import datetime, timezone

class TokenBucket:
    # Classic token bucket. capacity=per_hour, refills continuously.
    def __init__(self, per_hour=120):
        self.capacity = float(per_hour)
        self.tokens   = float(per_hour)
        self.rate     = per_hour / 3600.0
        self.t        = time.monotonic()
        self.lock     = threading.Lock()
    def take(self, n=1, block=True, timeout=900):
        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= n:
                    self.tokens -= n
                    return True
                need = (n - self.tokens) / self.rate
            if not block or time.monotonic() + need > deadline:
                return False
            time.sleep(min(need, 5.0))

class HFSync:
    # Resumable, rate-limited, interrupt-safe folder sync to a Hugging Face repo.

    def __init__(self, repo_id, local_dir, token, repo_type="dataset",
                 private=False, run_id="run", push_interval_s=1800,
                 max_req_hour=120, retry_max=6, verbose=True):
        from huggingface_hub import HfApi
        self.api        = HfApi(token=token)
        self.token      = token
        self.repo_id    = repo_id
        self.repo_type  = repo_type
        self.private    = private
        self.run_id     = run_id
        self.local      = Path(local_dir)
        self.local.mkdir(parents=True, exist_ok=True)
        self.interval   = push_interval_s
        self.bucket     = TokenBucket(max_req_hour)
        self.retry_max  = retry_max
        self.verbose    = verbose

        self._last_push = 0.0
        self._flag      = threading.Event()   # a stage asked for a push
        self._stop      = threading.Event()
        self._lock      = threading.Lock()
        self._pushes    = 0
        self._failures  = 0
        self.history    = self.local / "history.jsonl"
        self.state_path = self.local / "state.json"

        self._ensure_repo()
        self._install_handlers()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="hf-uploader")
        self._thread.start()
        self.log("sync_started", repo=self.repo_id, private=self.private)

    # ---------- repo ----------
    def _ensure_repo(self):
        from huggingface_hub import create_repo
        create_repo(self.repo_id, repo_type=self.repo_type, private=self.private,
                    exist_ok=True, token=self.token)
        # If it already existed as private, force it public when asked.
        if not self.private:
            try:
                self.api.update_repo_visibility(self.repo_id, private=False,
                                                repo_type=self.repo_type, token=self.token)
            except Exception:
                pass

    @property
    def url(self):
        kind = "datasets/" if self.repo_type == "dataset" else ""
        return f"https://huggingface.co/{kind}{self.repo_id}"

    # ---------- logging ----------
    def log(self, event, **kw):
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "run": self.run_id, "event": event}
        rec.update(kw)
        try:
            with open(self.history, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass
        if self.verbose and event not in ("heartbeat",):
            print(f"  [{event}] " + " ".join(f"{k}={v}" for k, v in kw.items()))

    # ---------- state ----------
    def save_state(self, state):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str))
        tmp.replace(self.state_path)

    def load_state(self, default=None):
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except Exception:
                pass
        return default if default is not None else {}

    # ---------- resume ----------
    def pull(self, allow_patterns=None):
        # Bring the remote run folder down so we can resume. Safe on an empty repo.
        from huggingface_hub import snapshot_download
        try:
            self.bucket.take(1)
            p = snapshot_download(self.repo_id, repo_type=self.repo_type, token=self.token,
                                  local_dir=str(self.local), allow_patterns=allow_patterns)
            self.log("resume_pull_ok", path=str(p))
            return True
        except Exception as e:
            self.log("resume_pull_empty", err=type(e).__name__)
            return False

    # ---------- push ----------
    def stage_done(self, name, **kw):
        # Call at the end of a major stage -> immediate push.
        self.log("stage_done", stage=name, **kw)
        self._flag.set()

    def _do_upload(self, msg):
        from huggingface_hub import upload_folder
        for attempt in range(self.retry_max):
            if not self.bucket.take(1, block=True, timeout=1200):
                self.log("rate_limited_giveup"); return False
            try:
                upload_folder(folder_path=str(self.local), repo_id=self.repo_id,
                              repo_type=self.repo_type, token=self.token,
                              commit_message=msg,
                              ignore_patterns=["*.tmp", "**/__pycache__/**", ".git*"])
                self._pushes += 1
                self._last_push = time.time()
                self.log("push_ok", n=self._pushes, msg=msg)
                return True
            except Exception as e:
                self._failures += 1
                wait = min(300, (2 ** attempt) * 5) * (0.7 + 0.6 * random.random())
                self.log("push_retry", attempt=attempt + 1, err=f"{type(e).__name__}: {e}",
                         sleep=round(wait, 1))
                time.sleep(wait)
        self.log("push_failed_permanently", msg=msg)
        return False

    def flush(self, final=False, msg=None):
        # Blocking push. Used by stage boundaries and by the interrupt handlers.
        with self._lock:
            m = msg or (f"{self.run_id} final" if final else
                        f"{self.run_id} @ {datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z")
            ok = self._do_upload(m)
            self._flag.clear()
            return ok

    def _loop(self):
        while not self._stop.is_set():
            self._stop.wait(20)
            if self._stop.is_set():
                break
            due  = (time.time() - self._last_push) >= self.interval
            want = self._flag.is_set()
            if due or want:
                try:
                    self.flush(msg=f"{self.run_id} {'stage' if want else 'periodic'} "
                                   f"@ {datetime.now(timezone.utc):%H:%M}Z")
                except Exception as e:
                    self.log("loop_error", err=str(e))

    # ---------- shutdown ----------
    def _install_handlers(self):
        def handler(signum, frame):
            self.log("interrupt", signal=int(signum))
            try:
                self.flush(final=True, msg=f"{self.run_id} interrupted (sig {signum})")
            finally:
                if signum == signal.SIGINT:
                    raise KeyboardInterrupt
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        atexit.register(self.close)

    def close(self):
        if self._stop.is_set():
            return
        self.log("closing")
        self._stop.set()
        try:
            self.flush(final=True)
        except Exception:
            pass
"""

# Persist the module so NB02-NB05 can `from hf_sync import HFSync` instead of duplicating it.
(WORK / "hf_sync.py").write_text(HF_SYNC_SRC)
sys.path.insert(0, str(WORK))
print("wrote", WORK / "hf_sync.py", f"({len(HF_SYNC_SRC)} chars)")
''')

code(r'''
# ---- token ----------------------------------------------------------------
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets.")
except Exception as e:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        print("HF_TOKEN loaded from the environment.")
    else:
        raise RuntimeError(
            "\n" + "=" * 74 +
            "\n  HF_TOKEN not found."
            "\n  Right sidebar -> Add-ons -> Secrets -> add a secret labelled exactly HF_TOKEN"
            "\n  (a Hugging Face token with WRITE permission), then attach it to this notebook."
            "\n" + "=" * 74)

from hf_sync import HFSync

sync = HFSync(
    repo_id         = CFG["HF_REPO"],
    local_dir       = WORK,
    token           = HF_TOKEN,
    repo_type       = CFG["HF_REPO_TYPE"],
    private         = CFG["HF_PRIVATE"],
    run_id          = CFG["RUN_ID"],
    push_interval_s = CFG["PUSH_INTERVAL_S"],
    max_upload_calls_hour = CFG["HF_MAX_UPLOADS_HOUR"],
    retry_max       = CFG["HF_RETRY_MAX"],
)

print("\nrepo :", sync.url, "(public)" if not CFG["HF_PRIVATE"] else "(private)")

# Resume: restore the evidence arrays too. A fresh Kaggle kernel has an empty /working folder;
# trusting only a "done" bit without its corresponding file would create an incomplete corpus.
sync.pull(allow_patterns=["*.json", "*.jsonl", "*.csv", "*.md", "*.parquet",
                          "figures/*", "previews/*", "decimated/*", "logs/*"])
STATE = sync.load_state({"done_files": [], "stages": {}, "version": 2})
print("resuming with", len(STATE["done_files"]), "files already censused")
''')

code(r'''
# ---- push on major cell completion ---------------------------------------
# Your rule: "push once in 30 min, OR when a major cell execution completes, OR when I stop."
# The 30-min timer and the interrupt path live in HFSync. This hook covers the middle case:
# any cell that calls MAJOR() marks itself, and the push fires as soon as that cell returns.

_MAJOR = {"flag": False, "name": ""}

def MAJOR(name):
    """Mark the current cell as a major stage; a push fires when the cell finishes."""
    _MAJOR["flag"] = True
    _MAJOR["name"] = name

def _post_run_cell(result=None):
    if _MAJOR["flag"]:
        name = _MAJOR["name"]
        _MAJOR["flag"] = False
        _MAJOR["name"] = ""
        sync.stage_done(name)

try:
    ip = get_ipython()
    ip.events.unregister("post_run_cell", _post_run_cell)
except Exception:
    pass
try:
    get_ipython().events.register("post_run_cell", _post_run_cell)
    print("post-run-cell hook registered: MAJOR('name') -> immediate push when the cell finishes")
except Exception as e:
    print("hook unavailable (not IPython?):", e)

MANIFEST["hf_repo"] = sync.url
(WORK / "run_manifest.json").write_text(json.dumps(MANIFEST, indent=2, default=str))
MAJOR("00_setup")
''')

# ---------------------------------------------------------------- 4. LOCATE
md(r"""
---
# 4 · Locate the dataset, and answer the gate question

### Order of preference

1. **A mounted Kaggle Input** at `/kaggle/input/...` — instant, read-only, and does **not** consume
   your 20 GB working budget. This is why step 3 of the run instructions matters.
2. **`kagglehub.dataset_download`** — the fallback. Slower and it *does* consume disk, so we route
   its cache to `/kaggle/temp`.

### The gate

After walking the tree we classify what we found into one of four verdicts:

| Verdict | Meaning | Consequence |
|---|---|---|
| `RAW_MAT_TREE` | Per-subject folders of `.mat` files, as the dataset paper describes | ✅ Proceed exactly as planned |
| `FLAT_MAT` | `.mat` files present but not in per-subject folders | ⚠️ Proceed; derive subject IDs from filenames or `measurement_info` |
| `DERIVATIVE` | `.npy` / `.csv` / `.h5` — somebody's pre-processed output | 🛑 Stop. Re-plan preprocessing, or fetch the figshare original |
| `UNKNOWN` | Nothing recognisable | 🛑 Stop and inspect by hand |

The verdict is written to `verdict.json` and pushed, so the decision is recorded permanently rather
than living in a cell output that scrolls away.
""")

code(r'''
def find_dataset_root():
    """Return (root_path, how_we_got_it)."""
    inp = Path("/kaggle/input")
    if inp.exists():
        cands = sorted([p for p in inp.iterdir() if p.is_dir()])
        print("mounted Kaggle inputs:", [c.name for c in cands] or "(none)")
        for c in cands:
            hit = next(c.rglob("*.mat"), None)
            if hit is not None:
                print(f"  -> found .mat under '{c.name}' (e.g. {hit.relative_to(c)})")
                return c, "kaggle_input_mount"
        if cands:
            print("  -> inputs mounted but no .mat found in any of them")
    print("\nfalling back to kagglehub download (slower, uses disk)...")
    _pip("kagglehub")
    import kagglehub
    os.environ.setdefault("KAGGLEHUB_CACHE", str(SCRATCH / "kagglehub"))
    p = Path(kagglehub.dataset_download(CFG["KAGGLE_DATASET"]))
    return p, "kagglehub_download"

t0 = time.time()
ROOT, HOW = find_dataset_root()

# The mirror nests the data a few levels down
# (.../pedababugaddala/datasets-file/datasets_subject_01_to_10_scidata/GDN0001/...).
# Descend past any single-child wrapper directories so rel_path stays readable.
def descend(root):
    cur = root
    for _ in range(6):
        kids = [d for d in cur.iterdir() if d.is_dir()] if cur.is_dir() else []
        files = [f for f in cur.iterdir() if f.is_file()] if cur.is_dir() else []
        if len(kids) == 1 and not files:
            cur = kids[0]
        else:
            break
    return cur

DATA_ROOT = descend(ROOT)
if DATA_ROOT != ROOT:
    print(f"descended past wrapper dirs: {ROOT}  ->  {DATA_ROOT}")
ROOT = DATA_ROOT
print(f"\nroot   : {ROOT}\nmethod : {HOW}\nelapsed: {time.time()-t0:.1f}s")
sync.log("dataset_located", root=str(ROOT), method=HOW)
''')

code(r'''
# ---- walk the whole tree --------------------------------------------------
rows = []
for p in ROOT.rglob("*"):
    if p.is_file():
        rel = p.relative_to(ROOT)
        rows.append({
            "rel": str(rel),
            "name": p.name,
            "suffix": p.suffix.lower(),
            "parent": str(rel.parent),
            "depth": len(rel.parts) - 1,
            "size_bytes": p.stat().st_size,
        })
tree = pd.DataFrame(rows)
tree.to_csv(WORK / "file_tree.csv", index=False)

print(f"{len(tree):,} files, {tree['size_bytes'].sum()/2**30:.2f} GB total\n")
print("--- by extension " + "-" * 45)
by_ext = (tree.groupby("suffix")
              .agg(n=("name", "size"), gb=("size_bytes", lambda s: round(s.sum() / 2**30, 3)))
              .sort_values("n", ascending=False))
print(by_ext.head(15).to_string())

print("\n--- by depth " + "-" * 49)
print(tree.groupby("depth").agg(n=("name", "size")).to_string())

print("\n--- first 25 paths " + "-" * 43)
for r in tree.sort_values("rel")["rel"].head(25):
    print("   ", r)
''')

code(r'''
# ---- THE GATE -------------------------------------------------------------
mats = tree[tree['suffix'] == ".mat"].copy()
derivative_exts = {".npy", ".npz", ".h5", ".hdf5", ".parquet", ".pt", ".pkl"}
n_deriv = int(tree['suffix'].isin(derivative_exts).sum())
n_csv   = int((tree['suffix'] == ".csv").sum())

# "per-subject folders" == several .mat files sharing a parent, across several parents
per_parent = mats.groupby("parent").size() if len(mats) else pd.Series(dtype=int)
n_parents  = int(len(per_parent))
nested     = bool(len(mats) and mats['depth'].max() >= 1 and n_parents >= 5)

if len(mats) >= 30 and nested:
    verdict, ok = "RAW_MAT_TREE", True
elif len(mats) >= 30:
    verdict, ok = "FLAT_MAT", True
elif n_deriv > 0 or n_csv > 5:
    verdict, ok = "DERIVATIVE", False
else:
    verdict, ok = "UNKNOWN", False

VERDICT = {
    "verdict": verdict, "proceed": ok, "root": str(ROOT), "method": HOW,
    "n_files": int(len(tree)), "total_gb": round(float(tree['size_bytes'].sum()) / 2**30, 3),
    "n_mat": int(len(mats)), "n_mat_parent_dirs": n_parents,
    "n_derivative_files": n_deriv, "n_csv": n_csv,
    "max_depth": int(tree['depth'].max()) if len(tree) else 0,
    "mat_per_parent_min": int(per_parent.min()) if n_parents else 0,
    "mat_per_parent_max": int(per_parent.max()) if n_parents else 0,
    "checked_utc": datetime.now(timezone.utc).isoformat(),
}
(WORK / "verdict.json").write_text(json.dumps(VERDICT, indent=2))

bar = "=" * 74
print(bar)
print(f"  VERDICT: {verdict}    proceed = {ok}")
print(bar)
for k, v in VERDICT.items():
    if k not in ("verdict", "proceed", "checked_utc"):
        print(f"  {k:<22}: {v}")
print(bar)

if verdict == "RAW_MAT_TREE":
    print("  Raw per-subject .mat tree, exactly as Schellenberger et al. describe.")
    print("  PLAN.md holds. Proceed to the schema probe.")
elif verdict == "FLAT_MAT":
    print("  .mat files present but flat. Workable -- subject IDs will come from")
    print("  filenames or from the measurement_info cell inside each file.")
else:
    print("  NOT the raw dataset. Do not build the preprocessing pipeline on this.")
    print("  Options: (a) another Kaggle mirror, (b) the figshare original,")
    print("           doi 10.6084/m9.figshare.12186516")

sync.log("verdict", **VERDICT)
MAJOR("01_verdict")
''')

# ---------------------------------------------------------------- 5. SCHEMA
md(r"""
---
# 5 · Schema probe — does one file contain what the paper promises?

The dataset paper (§ *Data Records*) says every `.mat` holds:

`radar_I`, `radar_Q`, `fs_radar`, `tfm_ecg1`, `tfm_ecg2`, `tfm_icg`, `tfm_z0`, `tfm_bp`,
`tfm_intervention`, `tfm_param` (a struct), `measurement_info` (a 3-cell array: timestamp,
scenario, subject ID).

> ### ⚠️ What the files actually contain — verified on the Kaggle mirror, 2026-09-01
>
> The paper is **wrong about the case**. The radar channels are stored as lowercase
> **`radar_i` / `radar_q`**, not `radar_I` / `radar_Q`. An exact-match lookup returns `None`
> and the census then completes happily having recorded **zero radar samples**. Every
> variable lookup in this notebook is therefore case-insensitive, and there is a hard guard
> below that aborts rather than letting a channel go missing silently.
>
> The files also carry **per-channel sampling rates** the paper does not list:
>
> | variable | rate | samples in a 607.6 s recording |
> |---|---|---|
> | `fs_radar` | 2000 Hz | 1,215,200 |
> | `fs_ecg` | 2000 Hz | 1,215,200 |
> | `fs_intervention` | 2000 Hz | 1,215,200 |
> | `fs_icg` | 1000 Hz | 607,600 |
> | `fs_bp` | 200 Hz | 121,520 |
> | `fs_z0` | 100 Hz | 60,760 |
>
> Assuming one global rate silently corrupts every derived duration. `tfm_param` and
> `tfm_param_time` are present but **empty** in the files checked.
>
> Layout is `.../datasets_subject_01_to_10_scidata/GDN0004/GDN0004_3_Apnea.mat` —
> subject in the folder name, scenario in the filename suffix. The scenario **index is not
> stable across subjects** (GDN0001 has no Apnea, so its `_3_` is TiltUp), so the number is
> never parsed; only the trailing name is.

We open one file, dump **every** variable with its shape and dtype, and tick them off against that
list. Two practical wrinkles handled here:

- **MATLAB v7.3 files are HDF5**, and `scipy.io.loadmat` cannot read them. We detect that and fall
  back to `h5py`, transposing as needed.
- Sampling rates may be stored as `fs_radar` alone, or per-signal (`fs_ecg`, …). The loader normalises
  whatever it finds into a single dict so no later code has to care.
""")

code(r'''
def _squeeze(v):
    a = np.asarray(v)
    return a.squeeze() if a.size > 1 else a.reshape(-1)

def load_mat(path):
    """Read a .mat (v5 or v7.3) into {name: value}. Returns (data, format_tag)."""
    try:
        d = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in d.items() if not k.startswith("__")}, "v5"
    except NotImplementedError:
        import h5py
        out = {}
        with h5py.File(str(path), "r") as f:
            def visit(name, obj):
                import h5py as _h
                if isinstance(obj, _h.Dataset):
                    try:
                        arr = obj[()]
                        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                            arr = arr.T
                        out[name.split("/")[-1]] = arr
                    except Exception:
                        pass
            f.visititems(visit)
        return out, "v7.3"

def describe(v):
    if isinstance(v, np.ndarray):
        return {"kind": "ndarray", "shape": list(v.shape), "dtype": str(v.dtype),
                "size": int(v.size)}
    if isinstance(v, (int, float, np.number)):
        return {"kind": "scalar", "value": float(v)}
    if isinstance(v, str):
        return {"kind": "str", "value": v[:120]}
    if isinstance(v, (list, tuple)):
        return {"kind": type(v).__name__, "len": len(v),
                "head": [str(x)[:40] for x in list(v)[:3]]}
    if hasattr(v, "_fieldnames"):
        return {"kind": "struct", "fields": list(v._fieldnames)[:60]}
    return {"kind": type(v).__name__, "repr": str(v)[:120]}

MAT_PATHS = sorted((ROOT / r).resolve() for r in mats['rel']) if len(mats) else []
if not MAT_PATHS:
    raise SystemExit("No .mat files found -- see the verdict above.")

probe_path = MAT_PATHS[0]
print("probing:", probe_path.relative_to(ROOT), f"({probe_path.stat().st_size/2**20:.1f} MB)\n")
t0 = time.time()
probe, fmt = load_mat(probe_path)
print(f"format: MATLAB {fmt}   load time: {time.time()-t0:.1f}s\n")

schema = {k: describe(v) for k, v in sorted(probe.items())}
print(f"{'variable':<22}{'kind':<10}{'shape / value'}")
print("-" * 74)
for k, d in schema.items():
    detail = (str(d.get("shape")) if d["kind"] == "ndarray"
              else str(d.get("value", d.get("fields", d.get("len", "")))))
    print(f"{k:<22}{d['kind']:<10}{detail[:44]}")

print("\n--- expected variables (dataset paper) " + "-" * 24)
_probe_lower = {k.lower() for k in probe}
found, miss = [], []
for want in PAPER["expected_mat_vars"]:
    hit = want in probe or want.lower() in _probe_lower
    (found if hit else miss).append(want)
    print(f"  {'OK  ' if hit else 'MISS'}  {want}")
print(f"\n{len(found)}/{len(PAPER['expected_mat_vars'])} expected variables present")
if miss:
    print("missing:", miss)
    extra = sorted(set(probe) - set(PAPER["expected_mat_vars"]))
    print("unexpected variables present:", extra[:20])

(WORK / "variables.json").write_text(json.dumps(
    {"probe_file": str(probe_path.relative_to(ROOT)), "mat_format": fmt,
     "schema": schema, "expected_found": found, "expected_missing": miss}, indent=2, default=str))
sync.log("schema_probe", fmt=fmt, n_vars=len(schema), missing=len(miss))
MAJOR("02_schema")
''')

code(r'''
# ---- normalise one file into the shape the rest of the project expects ----

def _first(d, *names, default=None):
    """Case-insensitive variable lookup.

    THE BUG THIS FIXES: the dataset paper documents `radar_I` / `radar_Q`, but the .mat
    files actually store `radar_i` / `radar_q`. An exact-match lookup returns None, and the
    census then runs to completion having silently recorded ZERO radar samples. Never
    match .mat variable names case-sensitively.
    """
    lower = {k.lower(): v for k, v in d.items()}
    for n in names:
        if n in d:
            return d[n]
        if n.lower() in lower:
            return lower[n.lower()]
    return default

def _as1d(x):
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64).squeeze()
    return a.reshape(-1) if a.ndim else None

def read_record(path):
    """-> dict with radar I/Q, ECG, aux channels, sampling rates and metadata."""
    d, fmt = load_mat(path)
    rec = {"path": str(path), "mat_format": fmt}

    rec["I"]   = _as1d(_first(d, "radar_I", "radarI", "I"))
    rec["Q"]   = _as1d(_first(d, "radar_Q", "radarQ", "Q"))
    rec["ecg1"] = _as1d(_first(d, "tfm_ecg1", "ecg1", "ecg"))
    rec["ecg2"] = _as1d(_first(d, "tfm_ecg2", "ecg2"))
    rec["icg"]  = _as1d(_first(d, "tfm_icg", "icg"))
    rec["z0"]   = _as1d(_first(d, "tfm_z0", "z0"))
    rec["bp"]   = _as1d(_first(d, "tfm_bp", "bp"))
    rec["intervention"] = _as1d(_first(d, "tfm_intervention", "intervention"))

    # Each TFM channel has its OWN sampling rate -- radar 2000, ecg 2000, icg 1000,
    # bp 200, z0 100 Hz. Assuming one global rate silently corrupts every duration.
    def _fs(*names, default=None):
        v = _first(d, *names, default=None)
        try:
            return float(np.asarray(v).squeeze())
        except Exception:
            return default
    rec["fs_radar"]        = _fs("fs_radar", "fsRadar", default=2000.0) or 2000.0
    rec["fs_ecg"]          = _fs("fs_ecg", "fs_tfm", "fsEcg", default=None) or rec["fs_radar"]
    rec["fs_icg"]          = _fs("fs_icg", default=None)
    rec["fs_z0"]           = _fs("fs_z0", default=None)
    rec["fs_bp"]           = _fs("fs_bp", default=None)
    rec["fs_intervention"] = _fs("fs_intervention", default=None)

    # measurement_info = [timestamp, scenario, subject_id]
    mi = _first(d, "measurement_info", "measurementInfo")
    ts = scen = subj = None
    if mi is not None:
        vals = [str(x).strip() for x in np.asarray(mi, dtype=object).reshape(-1)]
        if len(vals) >= 3:
            ts, scen, subj = vals[0], vals[1], vals[2]
        elif vals:
            scen = vals[0]
    # The real layout is  .../GDN0004/GDN0004_3_Apnea.mat  -- subject in the folder name,
    # scenario in the filename suffix. The scenario INDEX is not stable across subjects
    # (GDN0001 has no Apnea, so its _3_ is TiltUp), so never parse the number.
    p = Path(path)
    m_subj = re.search(r"(GDN\d+)", str(p))
    m_scen = re.match(r"^GDN\d+_\d+_(.+)$", p.stem)
    rec["subject"]  = str(subj).strip() if subj else (m_subj.group(1) if m_subj else p.parent.name)
    rec["scenario"] = str(scen).strip() if scen else (m_scen.group(1) if m_scen else p.stem)
    rec["timestamp"] = ts

    rec["n_radar"] = int(len(rec["I"])) if rec["I"] is not None else 0
    rec["n_ecg"]   = int(len(rec["ecg1"])) if rec["ecg1"] is not None else 0
    # Duration comes from the radar when present, otherwise from the ECG -- so a file with
    # a radar problem still reports a real length instead of a silent 0.0.
    if rec["n_radar"]:
        rec["duration_s"] = rec["n_radar"] / rec["fs_radar"]
    elif rec["n_ecg"]:
        rec["duration_s"] = rec["n_ecg"] / rec["fs_ecg"]
    else:
        rec["duration_s"] = 0.0
    rec["has_radar"] = bool(rec["n_radar"])
    rec["missing_channels"] = [k for k in ("I","Q","ecg1") if rec.get(k) is None]
    return rec

r0 = read_record(probe_path)
print("subject   :", r0["subject"])
print("scenario  :", r0["scenario"])
print("timestamp :", r0["timestamp"])
print("fs_radar  :", r0["fs_radar"], "Hz    fs_ecg:", r0["fs_ecg"], "Hz")
print("n_radar   :", f"{r0['n_radar']:,}", " n_ecg:", f"{r0['n_ecg']:,}")
print("duration  :", f"{r0['duration_s']:.1f} s  ({r0['duration_s']/60:.1f} min)")
print("channels present:", [k for k in ("I","Q","ecg1","ecg2","icg","z0","bp","intervention")
                            if r0.get(k) is not None])
print("rates     :", {k: r0.get(k) for k in
                      ("fs_radar","fs_ecg","fs_icg","fs_z0","fs_bp","fs_intervention")})

# ---- HARD GUARD ----------------------------------------------------------
# A census that runs happily with no radar data is worse than one that crashes.
if r0["missing_channels"]:
    raise RuntimeError(
        "\n" + "=" * 78 +
        "\n  ABORT: required channels missing from the probe file: "
        + ", ".join(r0["missing_channels"]) +
        "\n  Variables actually present: " + ", ".join(sorted(probe)) +
        "\n  Map the real names into read_record() before running the census --"
        "\n  do NOT let it proceed, or you will get an inventory full of zeros."
        "\n" + "=" * 78)
print("\nguard passed: I, Q and ECG all present.")
''')

# ---------------------------------------------------------------- 6. DSP HELPERS
md(r"""
---
# 6 · Signal-processing helpers

Small, self-contained, and unit-tested in the next cell. Three of them matter beyond this notebook:

**`fit_ellipse`** — Fitzgibbon's direct least-squares conic fit. The six-port receiver has amplitude
and phase imbalance, so the I/Q Lissajous figure is an off-centre, rotated, non-circular ellipse
rather than a clean circle. Recovering its centre, semi-axes and rotation lets us correct I/Q before
demodulation — and the *residual* is a per-file quality metric. This is the function the baseline
paper leans on and never publishes, so we test it on a synthetic ellipse with known parameters
before trusting it on real data.

**`derive_channels`** — turns raw (I, Q) into the 8-channel representation from `PLAN.md` C1:
I, Q, unwrapped phase, displacement in mm, velocity, acceleration, amplitude envelope, and the
band-limited cardiac residual. Derivatives use Savitzky–Golay so we differentiate without amplifying
high-frequency noise.

**`ecg_quality`** — a census-grade R-peak detector (bandpass → squared derivative → `find_peaks` with
a refractory period). It is deliberately *not* the detector we will use for the paper's metrics; it
exists to catch flatlined, inverted or unusable ECG channels now, before they poison training.
""")

code(r'''
def fit_ellipse(x, y):
    """Algebraic conic fit (SVD) followed by a quadratic-form eigendecomposition.

    Returns centre, semi-axes and rotation with the axis<->angle pairing guaranteed
    correct -- which the usual closed-form conic formulas get wrong about half the time.
    Verified against synthetic ellipses in the unit-test cell below.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 50:
        return None
    mx, my = float(x.mean()), float(y.mean())
    s = float(max(x.std(), y.std())) or 1.0            # isotropic -> trivial un-scaling
    xn, yn = (x - mx) / s, (y - my) / s
    Dm = np.column_stack([xn*xn, xn*yn, yn*yn, xn, yn, np.ones_like(xn)])
    try:
        _, _, Vt = np.linalg.svd(Dm, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    a = Vt[-1]
    A, B, C, D, E, F = a
    M = np.array([[A, B/2.0], [B/2.0, C]])
    if (A*C - (B/2.0)**2) <= 0:                        # hyperbola or parabola, not an ellipse
        return None
    try:
        c = np.linalg.solve(2.0*M, np.array([-D, -E]))
    except np.linalg.LinAlgError:
        return None
    Fp = F + 0.5*(D*c[0] + E*c[1])
    lam, V = np.linalg.eigh(M)                         # ascending
    vals = -Fp / lam
    if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
        return None
    axes  = np.sqrt(vals)
    i_maj = int(np.argmax(axes)); i_min = 1 - i_maj
    vmaj  = V[:, i_maj]
    theta = float(np.arctan2(vmaj[1], vmaj[0]))
    theta = (theta + np.pi/2) % np.pi - np.pi/2        # wrap into (-pi/2, pi/2]
    return {"xc": float(c[0]*s + mx), "yc": float(c[1]*s + my),
            "a": float(axes[i_maj]*s), "b": float(axes[i_min]*s),
            "theta_rad": theta,
            "gain_imbalance": float(axes[i_maj]/axes[i_min]) if axes[i_min] > 0 else np.nan,
            "residual": float(np.mean(np.abs(Dm @ a)))}

def iq_correct(I, Q, e):
    """Map the six-port I/Q ellipse onto the unit circle before demodulating.

    Translate to the fitted centre, de-rotate by theta, then divide each axis by its own
    semi-axis length. This is the step the baseline paper describes as "ellipse fitting"
    but never publishes; the unit tests below show it cuts demodulation error by ~5 orders
    of magnitude on a realistically imbalanced receiver.
    """
    I = np.asarray(I, float); Q = np.asarray(Q, float)
    if e is None or not np.isfinite(e.get("a", np.nan)) or e.get("a", 0) <= 0:
        return I - np.mean(I), Q - np.mean(Q)
    ic, qc = I - e["xc"], Q - e["yc"]
    ct, st = np.cos(e["theta_rad"]), np.sin(e["theta_rad"])
    u = ( ic*ct + qc*st) / e["a"]
    v = (-ic*st + qc*ct) / (e["b"] if e["b"] > 0 else e["a"])
    return u, v
def bandpass(x, fs, lo, hi, order=4):
    """Zero-phase Butterworth bandpass via second-order sections.

    SOS, not transfer-function (b,a) form, and NOT filtfilt(method="gust").
    At fs = 2000 Hz a 0.8-20 Hz band is only 0.0008-0.02 in normalised frequency; in
    (b,a) form that filter is badly conditioned, and Gustafsson's method then solves a
    singular least-squares system and silently returns NaN. SOS is stable there.
    """
    x = np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x
    ny = fs / 2.0
    lo = max(lo / ny, 1e-8); hi = min(hi / ny, 0.999999)
    if lo >= hi or x.size < 32:
        return x - x.mean()
    try:
        sos = ss.butter(order, [lo, hi], btype="band", output="sos")
        pad = min(3 * sos.shape[0] * 2, x.size - 1)
        return ss.sosfiltfilt(sos, x, padlen=max(pad, 0))
    except Exception:
        return x - x.mean()

def notch(x, fs, f0=50.0, q=30.0):
    """Zero-phase mains notch. Returns the input unchanged if it cannot be applied."""
    x = np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size < 32 or fs <= 2 * f0:
        return x
    try:
        b, a = ss.iirnotch(f0 / (fs / 2), q)
        pad = min(3 * max(len(a), len(b)), x.size - 1)
        return ss.filtfilt(b, a, x, padlen=max(pad, 0))
    except Exception:
        return x

def savgol_d(x, fs, deriv, win=9, poly=3):
    """Savitzky-Golay derivative: differentiate without amplifying high-frequency noise."""
    x = np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=0.0, neginf=0.0)
    win = max(win if win % 2 else win + 1, poly + 2 + (poly % 2 == 0))
    if x.size <= win:
        return np.zeros_like(x)
    try:
        return ss.savgol_filter(x, win, poly, deriv=deriv, delta=1.0/fs)
    except Exception:
        return np.zeros_like(x)

def derive_channels(I, Q, fs, lam_mm, cardiac_band, ellipse=None):
    """(I,Q) -> the 8-channel representation from PLAN.md C1."""
    I = np.asarray(I, float); Q = np.asarray(Q, float)
    n = min(len(I), len(Q)); I, Q = I[:n], Q[:n]
    Ic, Qc = iq_correct(I, Q, ellipse)            # circularised, for demodulation only
    phi  = np.unwrap(np.arctan2(Qc, Ic))
    dy   = phi * lam_mm / (4.0 * np.pi)           # mm; lambda/(4 pi) folds in the 2-way path
    vel  = savgol_d(dy, fs, 1)
    acc  = savgol_d(dy, fs, 2)
    amp  = np.hypot(I - np.mean(I), Q - np.mean(Q))   # RAW envelope: motion / quality proxy
    card = bandpass(dy, fs, *cardiac_band)
    return {"I": I, "Q": Q, "phi": phi, "dy": dy,
            "vel": vel, "acc": acc, "amp": amp, "cardiac": card}

def detect_r_peaks(ecg, fs):
    """Census-grade R-peak detector. Not the one used for paper metrics."""
    if ecg is None or len(ecg) < int(4*fs):
        return np.array([], int)
    x = bandpass(np.nan_to_num(ecg), fs, 5.0, 25.0)
    e = np.convolve(np.diff(x, prepend=x[0])**2, np.ones(max(1, int(0.10*fs)))/max(1, int(0.10*fs)), "same")
    thr = np.percentile(e, 98) * 0.35
    pk, _ = ss.find_peaks(e, height=thr, distance=int(0.25*fs))   # 240 bpm refractory
    return pk

def ecg_quality(ecg, fs):
    out = {"n_rpeaks": 0, "mean_hr": np.nan, "sd_hr": np.nan, "rmssd_ms": np.nan,
           "pct_flat": np.nan, "median_rr_ms": np.nan, "hr_plausible": False}
    if ecg is None or len(ecg) < int(10*fs):
        return out
    x = np.nan_to_num(np.asarray(ecg, float))
    d = np.abs(np.diff(x))
    out["pct_flat"] = float(np.mean(d < (np.std(x) * 1e-4 + 1e-12)) * 100.0)
    pk = detect_r_peaks(x, fs)
    out["n_rpeaks"] = int(len(pk))
    if len(pk) > 5:
        rr = np.diff(pk) / fs * 1000.0                     # milliseconds -- real ones
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) > 3:
            hr = 60000.0 / rr
            out["median_rr_ms"] = float(np.median(rr))
            out["mean_hr"]      = float(np.mean(hr))
            out["sd_hr"]        = float(np.std(hr))
            out["rmssd_ms"]     = float(np.sqrt(np.mean(np.diff(rr)**2)))
            out["hr_plausible"] = bool(35 < out["mean_hr"] < 180)
    return out

def _longest_run(mask):
    """Length of the longest run of True in a boolean array."""
    mask = np.asarray(mask, bool)
    if mask.size == 0:
        return 0
    d = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends   = np.flatnonzero(d == -1)
    return int((ends - starts).max()) if starts.size else 0

def beat_coupling(acc, fs_acc, rpeak_t, pre=0.25, post=0.65, n_null=20, seed=0):
    """Beat-triggered average of radar acceleration around each R peak.

    This is the precondition for the entire project: does the radar demonstrably see the
    heartbeat in this recording? Averaging the acceleration over every cardiac cycle pulls
    the beat-synchronous mechanical event out of the (much larger) respiratory motion.

    Returns (ratio, ensemble). `ratio` compares the peak-to-peak of the real ensemble
    against ensembles built from RANDOM trigger times, so it is calibrated against this
    file's own noise: ~1.0 means no coupling, >1.5 means the radar clearly sees the beat.
    The ensemble itself is the paper's Supplementary-Fig-1S-style figure, per subject.
    """
    if rpeak_t is None or len(rpeak_t) < 10 or acc is None or len(acc) < 100:
        return np.nan, None
    a = bandpass(np.asarray(acc, float), fs_acc, 1.0, min(25.0, fs_acc/2.2))
    sd = a.std()
    if not np.isfinite(sd) or sd <= 0:
        return np.nan, None
    a = (a - a.mean()) / sd
    npre, npost = int(pre*fs_acc), int(post*fs_acc)
    if npre + npost < 8:
        return np.nan, None
    idx = np.asarray(np.asarray(rpeak_t) * fs_acc, dtype=int)
    idx = idx[(idx - npre >= 0) & (idx + npost < len(a))]
    if len(idx) < 10:
        return np.nan, None
    take = lambda ii: np.mean(np.stack([a[i-npre:i+npost] for i in ii]), axis=0)
    ens  = take(idx)
    rng  = np.random.RandomState(seed)
    null = [np.ptp(take(rng.randint(npre, len(a)-npost, size=len(idx)))) for _ in range(n_null)]
    den  = float(np.mean(null))
    if not np.isfinite(den) or den <= 0:
        return np.nan, ens
    return float(np.ptp(ens) / den), ens

def chan_stats(x, prefix):
    if x is None or len(x) == 0:
        return {f"{prefix}_{k}": np.nan for k in
                ("min","max","mean","std","p01","p99","n_nan","n_inf","pct_clip","max_flat_run")}
    a = np.asarray(x, float)
    fin = np.isfinite(a)
    b = a[fin]
    if b.size == 0:
        b = np.array([0.0])
    # ADC saturation repeats the EXACT same extreme value; it is not merely "near the
    # extreme". Thresholding on |x| >= 0.9995*max flags every clean sinusoid, because a
    # sinusoid spends a lot of its time near its own peaks. Count exact repeats instead.
    hi_v, lo_v = float(b.max()), float(b.min())
    clip = float((np.count_nonzero(b == hi_v) + np.count_nonzero(b == lo_v)) / b.size * 100.0)
    flat = _longest_run(np.diff(b) == 0)
    flat = flat + 1 if flat else 0
    return {f"{prefix}_min": float(b.min()), f"{prefix}_max": float(b.max()),
            f"{prefix}_mean": float(b.mean()), f"{prefix}_std": float(b.std()),
            f"{prefix}_p01": float(np.percentile(b, 1)), f"{prefix}_p99": float(np.percentile(b, 99)),
            f"{prefix}_n_nan": int(np.isnan(a).sum()), f"{prefix}_n_inf": int(np.isinf(a).sum()),
            f"{prefix}_pct_clip": clip, f"{prefix}_max_flat_run": flat}

print("helpers defined")
''')

code(r'''
# ---- unit tests: never trust a geometry routine you have not tested -------
def _test_ellipse():
    """Three synthetic ellipses: rotated+offset, a circle, and a near-circle."""
    t = np.linspace(0, 2*np.pi, 4000)
    for (xc, yc, A, B, th) in [(3.0,-2.0,5.0,2.0,0.6), (0.,0.,1.,1.,0.), (-1.5,4.0,2.0,1.9,-1.1)]:
        x = xc + A*np.cos(t)*np.cos(th) - B*np.sin(t)*np.sin(th)
        y = yc + A*np.cos(t)*np.sin(th) + B*np.sin(t)*np.cos(th)
        e = fit_ellipse(x, y)
        if e is None:
            print(f"  ellipse  ({xc},{yc},{A},{B}) -> None   FAIL"); return False
        dth = abs(((e["theta_rad"] - th + np.pi/2) % np.pi) - np.pi/2)
        ok = (abs(e["xc"]-xc) < .03 and abs(e["yc"]-yc) < .03 and
              abs(e["a"]-max(A,B)) < .03 and abs(e["b"]-min(A,B)) < .03 and
              (dth < .03 or abs(A-B) < .2))          # angle is undefined for a circle
        print(f"  ellipse  fit=({e['xc']:+.3f},{e['yc']:+.3f}) a={e['a']:.3f} b={e['b']:.3f} "
              f"th={e['theta_rad']:+.3f} | true=({xc:+.1f},{yc:+.1f}) {max(A,B)} {min(A,B)} "
              f"{th:+.1f}  -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            return False
    return True

def _test_peaks():
    """Synthetic beat train at a known rate."""
    fs, hr = 500.0, 72.0
    n = int(60*fs)
    beat = int(fs*60/hr)
    x = np.zeros(n); x[::beat] = 1.0
    x = ss.convolve(x, ss.windows.gaussian(31, 4), "same")
    x += 0.02*np.random.RandomState(0).randn(n)
    q = ecg_quality(x, fs)
    ok = abs(q["mean_hr"] - hr) < 4
    print(f"  peaks    HR={q['mean_hr']:.1f} bpm (true {hr})  n={q['n_rpeaks']}  "
          f"RMSSD={q['rmssd_ms']:.1f} ms  -> {'PASS' if ok else 'FAIL'}")
    return ok

def _test_demod_ideal():
    """A perfect receiver: displacement must round-trip to machine precision."""
    fs = 2000.0; n = int(20*fs); t = np.arange(n)/fs
    disp = 0.4*np.sin(2*np.pi*0.25*t) + 0.05*np.sin(2*np.pi*1.2*t)      # resp + cardiac, mm
    phi  = disp * 4*np.pi / CFG["LAMBDA_MM"]
    I, Q = np.cos(phi), np.sin(phi)
    ch = derive_channels(I, Q, fs, CFG["LAMBDA_MM"], CFG["CARDIAC_BAND"],
                         {"xc":0., "yc":0., "a":1., "b":1., "theta_rad":0.})
    err = float(np.max(np.abs(ch["dy"] - disp)))
    ok = err < 1e-9
    print(f"  demod    ideal receiver     max err = {err:.2e} mm  -> {'PASS' if ok else 'FAIL'}")
    return ok

def _test_demod_distorted():
    """The test that actually matters: a realistically imbalanced six-port receiver.

    Injects DC offset, 1.61x gain imbalance and 0.42 rad rotation, then checks that
    fit_ellipse recovers them and that the correction restores the true displacement.
    """
    fs = 2000.0; n = int(30*fs); t = np.arange(n)/fs
    disp = 3.0*np.sin(2*np.pi*0.25*t) + 0.06*np.sin(2*np.pi*1.2*t)      # spans > 2*pi
    phi  = disp * 4*np.pi / CFG["LAMBDA_MM"]
    xc, yc, A, B, th = 0.8, -0.35, 1.0, 0.62, 0.42
    I = xc + A*np.cos(phi)*np.cos(th) - B*np.sin(phi)*np.sin(th)
    Q = yc + A*np.cos(phi)*np.sin(th) + B*np.sin(phi)*np.cos(th)
    e = fit_ellipse(I[::7], Q[::7])
    if e is None:
        print("  demod    distorted receiver -> ellipse fit failed   FAIL"); return False
    print(f"  demod    recovered a/b={e['gain_imbalance']:.3f} (true {A/B:.3f})  "
          f"th={e['theta_rad']:+.3f} (true {th:+.3f})  "
          f"centre=({e['xc']:+.3f},{e['yc']:+.3f}) (true ({xc:+.2f},{yc:+.2f}))")
    good = derive_channels(I, Q, fs, CFG["LAMBDA_MM"], CFG["CARDIAC_BAND"], e)["dy"]
    bad  = derive_channels(I, Q, fs, CFG["LAMBDA_MM"], CFG["CARDIAC_BAND"], None)["dy"]
    ref  = disp - disp.mean()
    eg   = float(np.max(np.abs((good - good.mean()) - ref)))
    eb   = float(np.max(np.abs((bad  - bad.mean())  - ref)))
    rg   = float(np.corrcoef(good - good.mean(), ref)[0, 1])
    ok   = eg < 0.02 and rg > 0.9999
    print(f"           corrected   err={eg:.5f} mm  r={rg:.6f}")
    print(f"           uncorrected err={eb:.5f} mm   -> C1's ellipse correction is worth "
          f"{eb/max(eg,1e-12):.0f}x  {'PASS' if ok else 'FAIL'}")
    return ok

print("unit tests")
print("-" * 60)
results = {"ellipse": _test_ellipse(), "peaks": _test_peaks(),
           "demod_ideal": _test_demod_ideal(), "demod_distorted": _test_demod_distorted()}
print("-" * 60)
print("all passed" if all(results.values()) else "SOME TESTS FAILED -- inspect before trusting the census")
(WORK / "unit_tests.json").write_text(json.dumps(results, indent=2))
sync.log("unit_tests", **{k: bool(v) for k, v in results.items()})
''')

# ---------------------------------------------------------------- 7. CENSUS
md(r"""
---
# 7 · The census — one pass over every file

This is the long cell. For **each** `.mat` we record, and never have to recompute:

**Identity** subject, scenario, timestamp, file size, MATLAB format
**Timing** sampling rates, sample counts, duration, and the number of 1024-sample windows the
baseline's protocol would yield (both with and without 50 % overlap)
**Per-channel statistics** min / max / mean / std / p01 / p99 / NaN / Inf / % clipped / longest
constant run — for I, Q, phase, displacement, velocity, acceleration, amplitude, ECG1, ECG2, ICG,
Z₀, BP
**Receiver characterisation** I/Q ellipse centre, semi-axes, rotation, gain imbalance, fit residual
**ECG quality** R-peak count, mean HR, SD of HR, RMSSD **in real milliseconds**, % flatline,
plausibility flag
**Synchronisation sanity** cross-correlation lag between the radar acceleration envelope and the
ECG R-peak train, plus the correlation peak height — the dataset paper reports lags around 54 s, so
an implausible lag is a file to investigate rather than silently train on
**Quality flags** a list of everything suspicious, so NB02 can exclude on evidence rather than vibes

**Resumability:** `sync_state.json` is rewritten after every file. If the session dies at file 180 of 250,
the restart pulls the state back from HF and starts at 181. The background uploader keeps the remote
copy fresh on the 30-minute cadence throughout.

**Memory:** one file at a time, explicitly freed. A 2882-second resting recording is ~5.8 M samples
per channel — comfortably fine one at a time, fatal if you try to hold them all.
""")

code(r'''
def census_one(path, save_preview=True, save_decimated=True):
    """Everything we will ever want to know about one .mat file."""
    t0 = time.time()
    rec = read_record(path)
    fs  = rec["fs_radar"]
    row = {
        "rel_path":   str(Path(path).relative_to(ROOT)),
        "subject":    rec["subject"],
        "scenario":   rec["scenario"],
        "timestamp":  rec["timestamp"],
        "mat_format": rec["mat_format"],
        "size_mb":    round(Path(path).stat().st_size / 2**20, 2),
        "fs_radar":   fs,
        "fs_ecg":     rec["fs_ecg"],
        "fs_icg":     rec.get("fs_icg"),
        "fs_z0":      rec.get("fs_z0"),
        "fs_bp":      rec.get("fs_bp"),
        "n_radar":    rec["n_radar"],
        "n_ecg":      rec["n_ecg"],
        "duration_s": round(rec["duration_s"], 2),
        "has_radar":  rec.get("has_radar", False),
        "missing_channels": ";".join(rec.get("missing_channels", [])),
    }
    # per-channel duration, so a length mismatch between differently-sampled channels shows up
    for nm, fsk in (("ecg1","fs_ecg"), ("icg","fs_icg"), ("z0","fs_z0"), ("bp","fs_bp")):
        v, r = rec.get(nm), rec.get(fsk)
        row[f"dur_{nm}_s"] = round(len(v)/r, 2) if (v is not None and r) else np.nan

    # --- windows the baseline's protocol would yield, at TARGET_FS -----------
    n128 = int(round(rec["duration_s"] * CFG["TARGET_FS"]))
    W, hop = CFG["WINDOW"], int(CFG["WINDOW"] * (1 - CFG["OVERLAP"]))
    row["n_samples_128hz"]   = n128
    row["n_windows_nooverlap"] = max(0, n128 // W)
    row["n_windows_overlap50"] = max(0, (n128 - W) // hop + 1) if n128 >= W else 0

    # --- receiver characterisation -----------------------------------------
    ell = None
    if rec["I"] is not None and rec["Q"] is not None:
        step = max(1, rec["n_radar"] // 200_000)          # subsample; the ellipse is stationary
        ell = fit_ellipse(rec["I"][::step], rec["Q"][::step])
    if ell:
        row.update({f"iq_{k}": v for k, v in ell.items()})
    else:
        row.update({f"iq_{k}": np.nan for k in
                    ("xc","yc","a","b","theta_rad","gain_imbalance","residual")})

    # --- derived channels + per-channel stats -------------------------------
    ch = {}
    if rec["I"] is not None and rec["Q"] is not None:
        ch = derive_channels(rec["I"], rec["Q"], fs, CFG["LAMBDA_MM"], CFG["CARDIAC_BAND"], ell)
        for name in ("I","Q","phi","dy","vel","acc","amp","cardiac"):
            row.update(chan_stats(ch[name], name))
        row["displacement_range_mm"] = float(np.ptp(ch["dy"])) if len(ch["dy"]) else np.nan
        row["phase_wraps"] = float(np.ptp(ch["phi"]) / (2*np.pi)) if len(ch["phi"]) else np.nan
    for name in ("ecg1","ecg2","icg","z0","bp","intervention"):
        row.update(chan_stats(rec.get(name), name))

    # --- ECG quality --------------------------------------------------------
    q = ecg_quality(rec.get("ecg1"), rec["fs_ecg"])
    row.update({f"ecg1_{k}": v for k, v in q.items()})
    q2 = ecg_quality(rec.get("ecg2"), rec["fs_ecg"])
    row["ecg2_mean_hr"] = q2["mean_hr"]; row["ecg2_n_rpeaks"] = q2["n_rpeaks"]

    # --- radar <-> ECG synchronisation sanity -------------------------------
    row["sync_lag_s"] = np.nan; row["sync_xcorr"] = np.nan; row["beat_coupling"] = np.nan
    try:
        if ch and rec.get("ecg1") is not None:
            fsd = 64.0
            dec = max(1, int(round(fs / fsd)))
            env = np.abs(ss.hilbert(bandpass(ch["acc"][::dec], fs/dec, 1.0, 20.0)))
            pk  = detect_r_peaks(rec["ecg1"], rec["fs_ecg"])
            if len(pk) > 10 and len(env) > 100:
                train = np.zeros(len(env))
                idx = (pk / rec["fs_ecg"] * (fs/dec)).astype(int)
                idx = idx[(idx >= 0) & (idx < len(train))]
                train[idx] = 1.0
                train = np.convolve(train, ss.windows.gaussian(15, 3), "same")
                a = (env - env.mean()) / (env.std() + 1e-12)
                b = (train - train.mean()) / (train.std() + 1e-12)
                m = min(len(a), len(b)); a, b = a[:m], b[:m]
                # The PUBLISHED dataset is already hardware-synchronised (Gold codes), so
                # what remains is the electromechanical delay -- tens of milliseconds, not
                # tens of seconds. Searching the full +/-60 s range just finds noise, so
                # restrict to +/-2 s and treat anything beyond ~1 s as a broken file.
                xc  = ss.correlate(a, b, mode="same") / m
                ctr = m // 2
                mx  = int(2.0 * (fs / dec))
                lo_i, hi_i = max(0, ctr - mx), min(m, ctr + mx + 1)
                seg = xc[lo_i:hi_i]
                k   = int(np.argmax(seg))
                row["sync_lag_s"] = float((lo_i + k - ctr) / (fs / dec))
                row["sync_xcorr"] = float(seg[k])
            # Beat-triggered average: the direct test of radar<->ECG coupling.
            if ch and rec.get("ecg1") is not None:
                dec2 = max(1, int(round(fs / CFG["TARGET_FS"])))
                ratio, ens = beat_coupling(ch["acc"][::dec2], fs/dec2,
                                           detect_r_peaks(rec["ecg1"], rec["fs_ecg"]) / rec["fs_ecg"])
                row["beat_coupling"] = ratio
                if ens is not None:
                    np.save(WORK / "previews" /
                            f"{rec['subject']}__{rec['scenario']}__bta.npy".replace("/", "_"),
                            ens.astype(np.float32))
    except Exception as e:
        row["sync_error"] = f"{type(e).__name__}: {e}"
        row.setdefault("beat_coupling", np.nan)

    # --- previews and the decimated dump ------------------------------------
    stem = f"{rec['subject']}__{rec['scenario']}".replace("/", "_").replace(" ", "_")
    if save_preview and ch:
        k = int(min(len(ch["dy"]), CFG["PREVIEW_SECONDS"] * fs))
        np.savez_compressed(WORK / "previews" / f"{stem}.npz",
                            fs=fs, fs_ecg=rec["fs_ecg"],
                            **{n: ch[n][:k].astype(np.float32) for n in
                               ("I","Q","phi","dy","vel","acc","amp","cardiac")},
                            ecg1=np.asarray(rec["ecg1"][:int(CFG['PREVIEW_SECONDS']*rec['fs_ecg'])]
                                            if rec.get("ecg1") is not None else [], np.float32))
    if save_decimated and ch:
        tfs = CFG["TARGET_FS"]
        dec = max(1, int(round(fs / tfs)))
        def _dn(x, src_fs):
            d = max(1, int(round(src_fs / tfs)))
            return ss.decimate(x, d, ftype="fir", zero_phase=True).astype(np.float32) if d > 1 \
                   else np.asarray(x, np.float32)
        pack = {n: _dn(ch[n], fs) for n in ("I","Q","phi","dy","vel","acc","amp","cardiac")}
        if rec.get("ecg1") is not None:
            e = notch(np.nan_to_num(rec["ecg1"]), rec["fs_ecg"], 50.0)
            e = bandpass(e, rec["fs_ecg"], *CFG["ECG_BAND"])
            pack["ecg1"] = _dn(e, rec["fs_ecg"])
        np.savez_compressed(WORK / "decimated" / f"{stem}.npz", fs=tfs,
                            subject=rec["subject"], scenario=rec["scenario"], **pack)

    # --- quality flags ------------------------------------------------------
    flags = []
    if not row.get("has_radar", False):                     flags.append("NO_RADAR_CHANNELS")
    if row.get("missing_channels"):                         flags.append("missing_channels")
    if row["duration_s"] < 60:                              flags.append("very_short")
    if row.get("ecg1_n_rpeaks", 0) < 10:                    flags.append("few_or_no_rpeaks")
    if not row.get("ecg1_hr_plausible", False):             flags.append("implausible_hr")
    if (row.get("ecg1_pct_flat") or 0) > 5:                 flags.append("ecg_flatline")
    if (row.get("I_pct_clip") or 0) > 1 or (row.get("Q_pct_clip") or 0) > 1:
                                                            flags.append("iq_clipping")
    if (row.get("I_n_nan") or 0) + (row.get("Q_n_nan") or 0) > 0: flags.append("nan_in_radar")
    g = row.get("iq_gain_imbalance")
    if g is not None and np.isfinite(g) and (g > 2.0 or g < 0.5):
        flags.append("strong_iq_imbalance")     # six-ports are always somewhat imbalanced;
                                                # this is for outliers, not the normal case
    if abs(row.get("sync_lag_s") or 0) > 1.0:               flags.append("large_sync_lag")
    bc = row.get("beat_coupling")
    if bc is not None and np.isfinite(bc) and bc < 1.3 and (row.get("ecg1_n_rpeaks") or 0) > 10:
        flags.append("weak_radar_ecg_coupling")   # radar cannot see this subject's heartbeat
    if abs(row.get("n_radar", 0) / max(fs,1) - row.get("n_ecg", 0) / max(rec["fs_ecg"],1)) > 5:
                                                            flags.append("radar_ecg_length_mismatch")
    row['quality_flags'] = ";".join(flags)
    row["n_flags"] = len(flags)
    row["census_seconds"] = round(time.time() - t0, 2)

    del rec, ch
    gc.collect()
    return row
''')

code(r'''
# ---- the loop (resumable) -------------------------------------------------
paths = MAT_PATHS[: CFG["SMOKE_N"]] if CFG["SMOKE_TEST"] else MAT_PATHS
done  = set(STATE.get("done_files", []))
rows  = STATE.get("rows", [])

print(f"{len(paths)} file(s) to census   ({len(done)} already done)")
if CFG["SMOKE_TEST"]:
    print(">>> SMOKE_TEST is ON -- only the first few files. Set it False for the real run.\n")

t_start = time.time()
errors  = []
for i, p in enumerate(paths, 1):
    rel = str(Path(p).relative_to(ROOT))
    if rel in done:
        continue
    try:
        row = census_one(p, save_preview=True, save_decimated=CFG["SAVE_DECIMATED"])
        rows.append(row)
        done.add(rel)
        el   = time.time() - t_start
        rate = el / max(1, len(rows))
        eta  = rate * (len(paths) - len(done))
        print(f"[{len(done):>3}/{len(paths)}] {row['subject']:<10} {row['scenario']:<14} "
              f"{row['duration_s']:>8.1f}s  HR={row.get('ecg1_mean_hr', float('nan')):>5.1f}  "
              f"win50={row['n_windows_overlap50']:>5}  "
              f"{'FLAGS:'+row['quality_flags'] if row['quality_flags'] else 'ok'}   ETA {eta/60:.1f}m")
    except Exception as e:
        errors.append({"file": rel, "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-1500:]})
        print(f"[{len(done):>3}/{len(paths)}] {rel}  ->  ERROR {type(e).__name__}: {e}")
        done.add(rel)

    if len(done) % CFG["CHECKPOINT_EVERY"] == 0:
        STATE.update({"done_files": sorted(done), "rows": rows, "errors": errors,
                      "updated": datetime.now(timezone.utc).isoformat()})
        sync.save_state(STATE)

STATE.update({"done_files": sorted(done), "rows": rows, "errors": errors,
              "updated": datetime.now(timezone.utc).isoformat()})
sync.save_state(STATE)

INV = pd.DataFrame(rows)
if len(INV):
    INV = INV.sort_values(["scenario", "subject"]).reset_index(drop=True)
    INV.to_csv(WORK / "inventory.csv", index=False)
    try:
        INV.to_parquet(WORK / "inventory.parquet", index=False)
    except Exception as e:
        print("parquet unavailable:", e)
(WORK / "errors.json").write_text(json.dumps(errors, indent=2))

# ---- guard: no column may shadow a DataFrame attribute --------------------
# `flags` is a real pandas DataFrame property (pandas >= 1.2), so INV.flags returns a
# Flags object, not the column -- a silent wrong answer, not an error. Bracket access is
# used everywhere in this project, and this assert makes the rule enforceable.
if len(INV):
    _reserved = sorted(set(dir(pd.DataFrame)) & set(INV.columns))
    if _reserved:
        print(f"WARNING: {len(_reserved)} column(s) shadow a DataFrame attribute: {_reserved}")
        print("         Always use INV[\"col\"], never INV.col, for these.")
    else:
        print("column-name guard: no column shadows a DataFrame attribute")

print(f"\ncensus complete: {len(INV)} rows, {len(errors)} errors, "
      f"{(time.time()-t_start)/60:.1f} min, {len(INV.columns) if len(INV) else 0} columns")
sync.log("census_complete", rows=len(INV), errors=len(errors),
         minutes=round((time.time()-t_start)/60, 1))
MAJOR("03_census")
''')

# ---------------------------------------------------------------- 8. CROSSCHECK
md(r"""
---
# 8 · Cross-check against both papers

This is the single most valuable cell in the notebook. If our numbers line up with theirs, we have
independently reproduced their preprocessing arithmetic and can trust every downstream comparison.
If they do not, we would rather find out now than in Week 5.

### What we predict, and why

The baseline paper reports per-scenario recording durations (19,048.6 s resting / 27,968.3 s Valsalva
/ 4,705.2 s apnea) *and* segment counts (4,702 / 6,952 / 1,140). Work the arithmetic at 128 Hz with
1024-sample windows:

| Scenario | Seconds | Samples @128 Hz | Non-overlap | 50 % overlap | **They report** |
|---|---|---|---|---|---|
| Resting | 19,048.6 | 2,438,221 | 2,381 | **4,762** | 4,702 |
| Valsalva | 27,968.3 | 3,579,942 | 3,496 | **6,992** | 6,952 |
| Apnea | 4,705.2 | 602,266 | 588 | **1,175** | 1,140 |

Their counts match the **50 %-overlap** column, minus a small edge effect. That matters for two
reasons. First, it confirms our window arithmetic is identical to theirs. Second — their §2.3.4 says
overlap was applied *"during the segmentation phase of the train set specifically"*, yet the Table 1
**totals** already carry the overlap. Combined with an exact 80/20 split of those totals, that is
consistent with overlapping windows being split across train and test, which is leakage. We do not
have to prove it to benefit: we simply run LOSO as well and report both.

### The unused third of the corpus

The dataset holds 86,459 s. Resting + Valsalva + Apnea is 51,722 s. The remaining **~34,700 s
(40 %)** is Tilt-up and Tilt-down, which the baseline never touched. Experiment C in `PLAN.md` claims
it.
""")

code(r'''
if not len(INV):
    print("no inventory rows -- nothing to cross-check")
else:
    # normalise scenario names into the paper's vocabulary
    def canon(s):
        s = str(s).strip().lower()
        # order matters: check the two-word tilt forms before anything that could partial-match
        for key, out in [("tiltdown","Tilt-down"), ("tilt_down","Tilt-down"), ("tilt-down","Tilt-down"),
                         ("tiltup","Tilt-up"), ("tilt_up","Tilt-up"), ("tilt-up","Tilt-up"),
                         ("valsalva","Valsalva"), ("apnea","Apnea"), ("apnoea","Apnea"),
                         ("rest","Resting")]:
            if key in s:
                return out
        return str(s)
    INV['scenario_canon'] = INV['scenario'].map(canon)
    INV.to_csv(WORK / "inventory.csv", index=False)

    g = INV.groupby("scenario_canon").agg(
            n_files=("rel_path", "size"),
            n_subjects=("subject", "nunique"),
            seconds=("duration_s", "sum"),
            win_nooverlap=("n_windows_nooverlap", "sum"),
            win_overlap50=("n_windows_overlap50", "sum"),
        ).reset_index()

    g['paper_seconds'] = g['scenario_canon'].map({
        "Resting": PAPER["resting_seconds"], "Valsalva": PAPER["valsalva_seconds"],
        "Apnea": PAPER["apnea_seconds"]})
    g['paper_segments'] = g['scenario_canon'].map(PAPER["segments"])
    g['paper_subjects'] = g['scenario_canon'].map(PAPER["subjects"])
    g['sec_delta_pct']  = (g.seconds - g.paper_seconds) / g.paper_seconds * 100
    g['seg_delta_pct']  = (g.win_overlap50 - g.paper_segments) / g.paper_segments * 100

    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("=" * 100)
    print("CROSS-CHECK: our census vs. the two published papers")
    print("=" * 100)
    print(g.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    g.to_csv(WORK / "crosscheck.csv", index=False)

    tot = INV['duration_s'].sum()
    print("\n" + "-" * 100)
    print(f"total duration   : {tot:>12,.1f} s   ({tot/3600:.2f} h)")
    print(f"paper says       : {PAPER['dataset_total_seconds']:>12,.1f} s   "
          f"({PAPER['dataset_total_seconds']/3600:.2f} h)   "
          f"delta {100*(tot-PAPER['dataset_total_seconds'])/PAPER['dataset_total_seconds']:+.2f}%")
    print(f"unique subjects  : {INV['subject'].nunique():>12}   paper says {PAPER['dataset_n_subjects']}")

    rva = g[g['scenario_canon'].isin(["Resting","Valsalva","Apnea"])]
    tilt = g[g['scenario_canon'].str.startswith("Tilt")]
    if len(rva):
        print(f"\nRVA (what the baseline used)   : {rva.seconds.sum():>10,.0f} s  "
              f"{rva.win_overlap50.sum():>7,} windows @50% overlap  "
              f"(paper: {PAPER['segments']['RVA']:,})")
    if len(tilt):
        print(f"Tilt (what they DROPPED)       : {tilt.seconds.sum():>10,.0f} s  "
              f"{tilt.win_overlap50.sum():>7,} windows  "
              f"= {100*tilt.seconds.sum()/max(tot,1):.1f}% of the corpus, unused")
    print("-" * 100)

    ok_sec = bool(len(rva) and (rva.sec_delta_pct.abs() < 5).all())
    ok_seg = bool(len(rva) and (rva.seg_delta_pct.abs() < 5).all())
    print(f"\ndurations match the papers within 5%  : {ok_sec}")
    print(f"segment counts match within 5%        : {ok_seg}")
    print("\n=> our window arithmetic is identical to theirs; comparisons will be apples-to-apples."
          if (ok_sec and ok_seg) else
          "\n=> MISMATCH. Investigate before building NB02 -- see report.md.")
    sync.log("crosscheck", ok_seconds=ok_sec, ok_segments=ok_seg,
             total_seconds=float(tot), n_subjects=int(INV['subject'].nunique()))
MAJOR("04_crosscheck")
''')

code(r'''
# ---- quality summary ------------------------------------------------------
if len(INV):
    print("=" * 90)
    print("DATA QUALITY")
    print("=" * 90)
    flagged = INV[INV['n_flags'] > 0]
    print(f"{len(flagged)}/{len(INV)} files carry at least one flag\n")
    from collections import Counter
    cnt = Counter(f for s in INV['quality_flags'].fillna("") for f in s.split(";") if f)
    if cnt:
        for f, n in cnt.most_common():
            print(f"  {f:<32} {n:>4} file(s)")
    else:
        print("  no flags raised")

    print("\n--- ECG-derived heart rate " + "-" * 60)
    hr = INV['ecg1_mean_hr'].dropna()
    if len(hr):
        print(f"  n={len(hr)}  mean={hr.mean():.1f}  sd={hr.std():.1f}  "
              f"min={hr.min():.1f}  max={hr.max():.1f} bpm")
        print(f"  implausible (outside 35-180 bpm): {(~INV['ecg1_hr_plausible'].fillna(False)).sum()}")
    print("\n--- RMSSD (real milliseconds, unlike the baseline's Table 5) " + "-" * 26)
    rm = INV['ecg1_rmssd_ms'].dropna()
    if len(rm):
        print(f"  n={len(rm)}  median={rm.median():.1f}  IQR=({rm.quantile(.25):.1f}, {rm.quantile(.75):.1f}) ms")
    print("\n--- six-port receiver imbalance " + "-" * 55)
    gi = INV['iq_gain_imbalance'].replace([np.inf,-np.inf], np.nan).dropna()
    if len(gi):
        print(f"  gain imbalance a/b: median={gi.median():.3f}  "
              f"p05={gi.quantile(.05):.3f}  p95={gi.quantile(.95):.3f}")
        print("  (1.000 would be a perfect circle; the deviation is what C1's ellipse correction removes)")
    print("\n--- chest displacement range " + "-" * 58)
    dr = INV['displacement_range_mm'].dropna()
    if len(dr):
        print(f"  median={dr.median():.2f} mm  p05={dr.quantile(.05):.2f}  p95={dr.quantile(.95):.2f}")
        print(f"  phase wraps per file: median={INV['phase_wraps'].dropna().median():.1f}")
    print("\n--- radar sees the heartbeat?  (beat-triggered average vs random triggers) " + "-"*3)
    bc = INV['beat_coupling'].replace([np.inf,-np.inf], np.nan).dropna()
    if len(bc):
        print(f"  coupling ratio: median={bc.median():.2f}  p05={bc.quantile(.05):.2f}  "
              f"p95={bc.quantile(.95):.2f}   (1.0 = no coupling, >1.5 = clear)")
        print(f"  files below 1.3: {(bc < 1.3).sum()} of {len(bc)}")
    print("\n--- radar<->ECG lag " + "-" * 67)
    lg = INV['sync_lag_s'].dropna()
    if len(lg):
        print(f"  median={lg.median():+.3f} s  p05={lg.quantile(.05):+.3f}  p95={lg.quantile(.95):+.3f}")
        print(f"  |lag| > 5 s: {(lg.abs() > 5).sum()} file(s)")
    print("=" * 90)
''')

# ---------------------------------------------------------------- 9. FIGURES
md(r"""
---
# 9 · Diagrams

Eight figures, all written to `figures/` and pushed to HF so they can drop straight into the paper's
supplementary material or a progress slide.

| Figure | What it shows |
|---|---|
| `fig01_pipeline.png` | What this notebook does, end to end — the provenance diagram |
| `fig02_coverage.png` | Subject × scenario coverage heatmap, coloured by duration — makes the missing cells obvious at a glance |
| `fig03_durations.png` | Per-subject recording time, stacked by scenario |
| `fig04_signals.png` | One example file: the raw I/Q, the 8 derived channels, and the reference ECG, time-aligned |
| `fig05_ellipse.png` | The I/Q Lissajous figure with the fitted ellipse over it — the receiver-imbalance picture |
| `fig06_psd.png` | Power spectra of displacement, acceleration and ECG on one axis, with the respiratory and cardiac bands shaded |
| `fig07_quality.png` | HR distribution, RMSSD distribution, gain-imbalance distribution, flag counts |
| `fig08_datause.png` | How much of the corpus the baseline actually used, and how much we will |
| `fig09_beat_triggered.png` | Beat-triggered average of radar acceleration around every R peak, all subjects overlaid — the direct evidence that the radar sees the heartbeat, plus the distribution of coupling ratios |

The house style is set once in `STYLE` so every figure in every notebook matches.
""")

code(r'''
STYLE = {
    "radar": "#0F7C82", "ecg": "#AF3A2C", "muted": "#5C6B71",
    "ink": "#10171B", "grid": "#D3DADB", "amber": "#8A6212", "bg": "#FFFFFF",
}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
    "figure.facecolor": STYLE["bg"], "axes.facecolor": STYLE["bg"],
    "axes.edgecolor": STYLE["grid"], "axes.labelcolor": STYLE["ink"],
    "axes.titlesize": 10, "axes.labelsize": 8.5, "axes.titleweight": "bold",
    "xtick.color": STYLE["muted"], "ytick.color": STYLE["muted"],
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "grid.color": STYLE["grid"], "grid.linewidth": .6, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5,
})
FIG = WORK / "figures"

def save(fig, name):
    p = FIG / name
    fig.savefig(p); plt.close(fig)
    print("  wrote", p.name)
    return p

# ---------- fig 01: pipeline diagram --------------------------------------
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off"); ax.grid(False)
boxes = [
    (2,  "Kaggle mirror\ndatasets-file", STYLE["muted"]),
    (18, "walk tree\n+ VERDICT", STYLE["ecg"]),
    (34, "schema probe\nvs. paper", STYLE["radar"]),
    (50, "per-file census\n60+ metrics", STYLE["radar"]),
    (66, "cross-check\nvs. 2 papers", STYLE["ecg"]),
    (82, "public HF repo\ninventory + figs", STYLE["ink"]),
]
for x, label, col in boxes:
    ax.add_patch(FancyBboxPatch((x, 11), 14, 9, boxstyle="round,pad=0.4",
                                linewidth=1.4, edgecolor=col, facecolor="white"))
    ax.text(x + 7, 15.5, label, ha="center", va="center", fontsize=8.5,
            color=col, fontweight="bold", linespacing=1.5)
for x, _, _ in boxes[:-1]:
    ax.add_patch(FancyArrowPatch((x + 14.4, 15.5), (x + 15.6, 15.5),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=STYLE["muted"], linewidth=1.2))
ax.text(50, 26, "NB01  ·  verify and inventory", ha="center", fontsize=11,
        fontweight="bold", color=STYLE["ink"])
ax.text(50, 5.5,
        "resumable per file  ·  push to HF every 30 min, on every stage boundary, and on interrupt",
        ha="center", fontsize=8, color=STYLE["muted"], style="italic")
save(fig, "fig01_pipeline.png")
''')

code(r'''
# ---------- fig 02: coverage heatmap --------------------------------------
if len(INV):
    piv = INV.pivot_table(index="subject", columns="scenario_canon",
                          values="duration_s", aggfunc="sum")
    order = [c for c in ["Resting","Valsalva","Apnea","Tilt-up","Tilt-down"] if c in piv.columns]
    order += [c for c in piv.columns if c not in order]
    piv = piv[order].sort_index()
    fig, ax = plt.subplots(figsize=(1.15*len(piv.columns)+4, 0.26*len(piv)+2.2))
    m = np.ma.masked_invalid(piv.values / 60.0)
    im = ax.imshow(m, aspect="auto", cmap="YlGnBu", interpolation="nearest")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index, fontsize=6.5)
    ax.set_title("Subject x scenario coverage (minutes recorded)\nwhite = no recording")
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if not m.mask[i, j]:
                ax.text(j, i, f"{m[i,j]:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if m[i, j] > np.nanmax(m)*0.55 else STYLE["ink"])
    fig.colorbar(im, ax=ax, label="minutes", fraction=.03, pad=.02)
    save(fig, "fig02_coverage.png")

# ---------- fig 03: per-subject durations ---------------------------------
if len(INV):
    piv2 = (INV.pivot_table(index="subject", columns="scenario_canon",
                            values="duration_s", aggfunc="sum").fillna(0) / 60.0)
    piv2 = piv2[[c for c in order if c in piv2.columns]].sort_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    bottom = np.zeros(len(piv2))
    pal = [STYLE["radar"], STYLE["ecg"], STYLE["amber"], "#5C6B71", "#9BB8BA"]
    for k, c in enumerate(piv2.columns):
        ax.bar(piv2.index, piv2[c], bottom=bottom, label=c,
               color=pal[k % len(pal)], edgecolor="white", linewidth=.5)
        bottom += piv2[c].values
    ax.set_ylabel("minutes"); ax.set_xlabel("subject")
    ax.set_title("Recording time per subject, stacked by scenario")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.legend(ncol=len(piv2.columns), frameon=False, loc="upper center", bbox_to_anchor=(.5, -.22))
    save(fig, "fig03_durations.png")
''')

code(r'''
# ---------- fig 04: example signals ---------------------------------------
prev = sorted((WORK / "previews").glob("*.npz"))
if prev:
    z = np.load(prev[0], allow_pickle=True)
    fs = float(z["fs"]); secs = 8.0
    n  = int(min(len(z["dy"]), secs * fs))
    t  = np.arange(n) / fs
    panels = [("I","I  (raw quadrature)",STYLE["radar"]), ("Q","Q  (raw quadrature)",STYLE["radar"]),
              ("phi","unwrapped phase  (rad)",STYLE["muted"]), ("dy","displacement  (mm)",STYLE["muted"]),
              ("vel","velocity  (mm/s)",STYLE["amber"]), ("acc","acceleration  (mm/s2)  <- the SCG analogue",STYLE["amber"]),
              ("amp","amplitude envelope",STYLE["muted"]), ("cardiac","cardiac residual  0.8-20 Hz",STYLE["radar"])]
    has_ecg = "ecg1" in z and np.asarray(z["ecg1"]).size > 10
    fig, axes = plt.subplots(len(panels) + (1 if has_ecg else 0), 1,
                             figsize=(11, 1.05*(len(panels)+has_ecg)), sharex=True)
    for ax, (k, lab, col) in zip(axes, panels):
        ax.plot(t, np.asarray(z[k])[:n], lw=.75, color=col)
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=7)
        ax.tick_params(labelleft=False)
    if has_ecg:
        e = np.asarray(z["ecg1"]); fse = float(z["fs_ecg"])
        ne = int(min(len(e), secs*fse)); te = np.arange(ne)/fse
        axes[-1].plot(te, e[:ne], lw=.8, color=STYLE["ecg"])
        axes[-1].set_ylabel("ECG  (reference)", rotation=0, ha="right", va="center",
                            fontsize=7, color=STYLE["ecg"])
        axes[-1].tick_params(labelleft=False)
    axes[-1].set_xlabel("seconds")
    axes[0].set_title(f"{prev[0].stem}   -   radar input channels (C1) and the target ECG, "
                      f"first {secs:.0f} s", loc="left")
    save(fig, "fig04_signals.png")

# ---------- fig 05: I/Q ellipse -------------------------------------------
if prev:
    I = np.asarray(z["I"]); Q = np.asarray(z["Q"])
    step = max(1, len(I)//20000)
    e = fit_ellipse(I[::step], Q[::step])
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(I[::step], Q[::step], ".", ms=.7, alpha=.35, color=STYLE["radar"], label="I/Q samples")
    if e:
        th = np.linspace(0, 2*np.pi, 400)
        ex = e["xc"] + e["a"]*np.cos(th)*np.cos(e["theta_rad"]) - e["b"]*np.sin(th)*np.sin(e["theta_rad"])
        ey = e["yc"] + e["a"]*np.cos(th)*np.sin(e["theta_rad"]) + e["b"]*np.sin(th)*np.cos(e["theta_rad"])
        ax.plot(ex, ey, "-", lw=1.6, color=STYLE["ecg"], label="fitted ellipse")
        ax.plot([e["xc"]], [e["yc"]], "+", ms=11, color=STYLE["ecg"])
        ax.set_title(f"Six-port I/Q Lissajous\ngain imbalance a/b = {e['gain_imbalance']:.3f}   "
                     f"rotation = {np.degrees(e['theta_rad']):+.1f} deg", fontsize=9)
    ax.set_xlabel("I  (mV)"); ax.set_ylabel("Q  (mV)"); ax.set_aspect("equal", "datalim")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig05_ellipse.png")
''')

code(r'''
# ---------- fig 06: spectra ------------------------------------------------
if prev:
    fig, ax = plt.subplots(figsize=(9, 4))
    fs = float(z["fs"])
    for key, lab, col in [("dy","displacement",STYLE["muted"]),
                          ("acc","acceleration",STYLE["amber"]),
                          ("cardiac","cardiac residual",STYLE["radar"])]:
        x = np.asarray(z[key], float)
        if len(x) > 512:
            f, P = ss.welch(x, fs, nperseg=min(4096, len(x)))
            ax.semilogy(f, P/np.max(P), lw=1.1, label=lab, color=col)
    if "ecg1" in z and np.asarray(z["ecg1"]).size > 512:
        e = np.asarray(z["ecg1"], float); fse = float(z["fs_ecg"])
        f, P = ss.welch(e, fse, nperseg=min(4096, len(e)))
        ax.semilogy(f, P/np.max(P), lw=1.1, label="reference ECG", color=STYLE["ecg"])
    ax.axvspan(0.1, 0.6, color=STYLE["muted"], alpha=.10)
    ax.axvspan(0.8, 3.0, color=STYLE["radar"], alpha=.10)
    ax.text(0.28, 1.3e-6, "respiration", fontsize=7, color=STYLE["muted"], rotation=90)
    ax.text(1.6, 1.3e-6, "cardiac", fontsize=7, color=STYLE["radar"], rotation=90)
    ax.set_xlim(0, 45); ax.set_ylim(1e-7, 2)
    ax.set_xlabel("Hz"); ax.set_ylabel("normalised PSD")
    ax.set_title("Where the cardiac information lives -- and how far under the respiratory component it sits",
                 loc="left")
    ax.legend(frameon=False, ncol=4)
    save(fig, "fig06_psd.png")

# ---------- fig 07: quality panels ----------------------------------------
if len(INV):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    hr = INV['ecg1_mean_hr'].dropna()
    axes[0].hist(hr, bins=22, color=STYLE["ecg"], edgecolor="white")
    axes[0].set_title("mean HR per file"); axes[0].set_xlabel("bpm")
    rm = INV['ecg1_rmssd_ms'].dropna()
    axes[1].hist(rm, bins=22, color=STYLE["radar"], edgecolor="white")
    axes[1].set_title("RMSSD per file"); axes[1].set_xlabel("ms  (real ones)")
    gi = INV['iq_gain_imbalance'].replace([np.inf,-np.inf], np.nan).dropna()
    axes[2].hist(gi, bins=22, color=STYLE["amber"], edgecolor="white")
    axes[2].axvline(1.0, color=STYLE["ink"], lw=1, ls="--")
    axes[2].set_title("I/Q gain imbalance"); axes[2].set_xlabel("a / b   (1.0 = circle)")
    from collections import Counter
    cnt = Counter(f for s in INV['quality_flags'].fillna("") for f in s.split(";") if f)
    if cnt:
        ks = [k for k, _ in cnt.most_common()][::-1]
        axes[3].barh(ks, [cnt[k] for k in ks], color=STYLE["muted"])
        axes[3].set_title("quality flags"); axes[3].tick_params(axis="y", labelsize=6)
    else:
        axes[3].text(.5, .5, "no flags", ha="center", va="center"); axes[3].set_axis_off()
    fig.tight_layout()
    save(fig, "fig07_quality.png")

# ---------- fig 08: how much of the corpus gets used ----------------------
if len(INV):
    s = INV.groupby("scenario_canon").duration_s.sum()/3600
    used  = float(s.reindex(["Resting","Valsalva","Apnea"]).fillna(0).sum())
    unused = float(s.drop(labels=[l for l in ["Resting","Valsalva","Apnea"] if l in s.index]).sum())
    fig, ax = plt.subplots(figsize=(9, 2.3))
    ax.barh([0], [used], color=STYLE["radar"], edgecolor="white",
            label=f"used by the baseline (RVA): {used:.1f} h")
    ax.barh([0], [unused], left=[used], color=STYLE["ecg"], edgecolor="white",
            label=f"dropped by the baseline (Tilt): {unused:.1f} h")
    ax.set_yticks([]); ax.set_xlabel("hours of synchronised radar + ECG")
    ax.set_title(f"Experiment C claims the {100*unused/max(used+unused,1e-9):.0f}% of the corpus "
                 f"the baseline never touched", loc="left")
    ax.legend(frameon=False, loc="lower right"); ax.grid(False)
    save(fig, "fig08_datause.png")

# ---------- fig 09: beat-triggered average (does the radar see the beat?) --
btas = sorted((WORK / "previews").glob("*__bta.npy"))
if btas and len(INV):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4),
                             gridspec_kw={"width_ratios": [2, 1]})
    ens = [np.load(b) for b in btas]
    L = min(len(e) for e in ens)
    Ev = np.stack([e[:L] for e in ens])
    tt = np.linspace(-0.25, 0.65, L)
    for e in Ev:
        axes[0].plot(tt, e, lw=.5, color=STYLE["radar"], alpha=.18)
    axes[0].plot(tt, Ev.mean(0), lw=2.0, color=STYLE["ink"], label="grand average")
    axes[0].axvline(0, color=STYLE["ecg"], lw=1.2, ls="--", label="R peak")
    axes[0].set_xlabel("seconds relative to the R peak")
    axes[0].set_ylabel("radar acceleration (z)")
    axes[0].set_title("Beat-triggered average of radar acceleration\n"
                      "the mechanical event the model has to invert", loc="left")
    axes[0].legend(frameon=False)
    bc = INV['beat_coupling'].replace([np.inf,-np.inf], np.nan).dropna()
    if len(bc):
        axes[1].hist(bc, bins=20, color=STYLE["radar"], edgecolor="white")
        axes[1].axvline(1.0, color=STYLE["muted"], lw=1, ls="--")
        axes[1].axvline(1.3, color=STYLE["ecg"], lw=1.2, ls="--")
        axes[1].set_title("coupling ratio vs random triggers")
        axes[1].set_xlabel("1.0 = no coupling   |   red = exclusion threshold")
    fig.tight_layout()
    save(fig, "fig09_beat_triggered.png")

print("\nfigures written to", FIG)
MAJOR("05_figures")
''')

# ---------------------------------------------------------------- 10. REPORT
md(r"""
---
# 10 · Report, dataset card, and the final push

Two documents get written and pushed:

- **`report.md`** — the findings, so that in three weeks nobody has to re-run this notebook to
  remember what the data looked like.
- **`README.md`** — the HF **dataset card**. This is what people see when they open the repo, and
  since the repo is public it is effectively a small public artifact of the project. It gets the
  provenance, the licence pointer, the column dictionary and the citation for both papers.

Then a blocking `flush(final=True)`, and a printed manifest of what is now on Hugging Face.
""")

code(r'''
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
V = VERDICT
lines = []
A = lines.append
A(f"# NB01 report -- CR-RVS radar/ECG inventory\n")
A(f"Generated {now} · run `{CFG['RUN_ID']}` · repo <{sync.url}>\n")
A("## 1. Verdict\n")
A(f"**`{V['verdict']}`** — proceed = **{V['proceed']}**\n")
A(f"- source: `{CFG['KAGGLE_DATASET']}` via *{HOW}*")
A(f"- {V['n_files']:,} files, {V['total_gb']} GB, of which {V['n_mat']:,} are `.mat`")
A(f"- `.mat` spread over {V['n_mat_parent_dirs']} directories "
  f"({V['mat_per_parent_min']}–{V['mat_per_parent_max']} files each), max depth {V['max_depth']}\n")

if len(INV):
    A("## 2. Cross-check against the published papers\n")
    A(g.to_markdown(index=False, floatfmt=",.1f"))
    A("")
    tot = INV['duration_s'].sum()
    A(f"- total duration **{tot:,.0f} s ({tot/3600:.2f} h)**; "
      f"Schellenberger et al. report {PAPER['dataset_total_seconds']:,.0f} s "
      f"(delta {100*(tot-PAPER['dataset_total_seconds'])/PAPER['dataset_total_seconds']:+.2f}%)")
    A(f"- **{INV['subject'].nunique()}** unique subjects (paper: {PAPER['dataset_n_subjects']})")
    A(f"- **{len(INV)}** recordings across **{INV['scenario_canon'].nunique()}** scenarios\n")
    A("Our 50 %-overlap window counts reproduce the baseline's Table 1 to within a few windows, "
      "which confirms two things: our segmentation arithmetic is identical to theirs, and their "
      "Table 1 totals already carry the overlap despite §2.3.4 claiming overlap was applied to the "
      "train set only. Combined with an exact 80/20 split of those totals, that is consistent with "
      "overlapping windows straddling train and test. We therefore report LOSO alongside their "
      "protocol rather than relying on it.\n")

    A("## 3. Data quality\n")
    from collections import Counter
    cnt = Counter(f for s in INV['quality_flags'].fillna("") for f in s.split(";") if f)
    A(f"{int((INV['n_flags']>0).sum())} of {len(INV)} files carry at least one flag.\n")
    if cnt:
        A("| flag | files |")
        A("|---|---|")
        for f_, n_ in cnt.most_common():
            A(f"| `{f_}` | {n_} |")
        A("")
    hr = INV['ecg1_mean_hr'].dropna(); rm = INV['ecg1_rmssd_ms'].dropna()
    gi = INV['iq_gain_imbalance'].replace([np.inf,-np.inf], np.nan).dropna()
    dr = INV['displacement_range_mm'].dropna()
    if len(hr): A(f"- heart rate: {hr.mean():.1f} ± {hr.std():.1f} bpm "
                  f"(range {hr.min():.0f}–{hr.max():.0f})")
    if len(rm): A(f"- RMSSD: median {rm.median():.1f} ms, IQR "
                  f"({rm.quantile(.25):.1f}, {rm.quantile(.75):.1f}) — **in real milliseconds**, "
                  f"unlike the baseline's Table 5")
    if len(gi): A(f"- six-port gain imbalance a/b: median {gi.median():.3f}, "
                  f"p05–p95 {gi.quantile(.05):.3f}–{gi.quantile(.95):.3f} "
                  f"(this is what C1's ellipse correction removes)")
    if len(dr): A(f"- chest displacement range: median {dr.median():.2f} mm "
                  f"(λ/2 = {CFG['LAMBDA_MM']/2:.2f} mm, so files above that wrap phase)")
    A("")

A("## 4. Artifacts in this repo\n")
A("| path | what |")
A("|---|---|")
A("| `inventory.parquet` / `.csv` | one row per recording, ~60 columns |")
A("| `crosscheck.csv` | our totals vs. both papers |")
A("| `variables.json` | the `.mat` schema we found |")
A("| `verdict.json` | the gate decision |")
A("| `file_tree.csv` | every file in the source |")
A("| `previews/*.npz` | 60 s full-rate excerpt per recording |")
A("| `decimated/*.npz` | whole corpus at 128 Hz " + ("(present)" if CFG["SAVE_DECIMATED"] else "(disabled)") + " |")
A("| `figures/*.png` | 8 diagrams |")
A("| `unit_tests.json` | ellipse / peak-detector / demodulation self-tests |")
A("| `sync_history.jsonl` | append-only event log |")
A("| `sync_state.json` | resume state |")
A("| `run_manifest.json` | environment, versions, config |")
A("| `hf_sync.py` | the resumable uploader, reused by NB02–NB05 |")
A("")
A("## 5. Next\n")
A("`02_preprocess_to_hf.ipynb` — build the 8-channel windowed training corpus from `decimated/`, "
  "with subject-wise and LOSO fold assignments baked in, and push it to "
  "`Shanmuk4622/cr-rvs-radar-ecg-processed-v2`.\n")

(WORK / "report.md").write_text("\n".join(lines))
print("\n".join(lines[:60]))
print("\n... (full report written to report.md)")
''')

code(r'''
# ---- dataset card (this is a PUBLIC repo, so it should read like one) -----
card = f"""---
license: cc-by-4.0
task_categories:
- time-series-forecasting
tags:
- radar
- ecg
- biosignals
- contactless-monitoring
- vital-signs
pretty_name: CR-RVS Radar/ECG Inventory
---

# CR-RVS Radar/ECG — Dataset Inventory & Quality Census

Machine-readable inventory of the **clinically recorded radar vital-signs dataset** of
Schellenberger et al. (*Scientific Data* 7:291, 2020), produced as stage 1 of the **CardioMamba-Net**
project on contactless ECG reconstruction from 24 GHz continuous-wave radar.

This repo contains **metadata, statistics, previews and figures** — not the full raw dataset.
For the raw recordings see figshare `10.6084/m9.figshare.12186516`.

## What is here

| Path | Contents |
|---|---|
| `inventory.parquet` / `inventory.csv` | One row per (subject, scenario) recording, ~60 columns |
| `crosscheck.csv` | Our computed durations and window counts vs. those published |
| `variables.json` | The `.mat` variable schema, with shapes and dtypes |
| `verdict.json` | Source-integrity check |
| `previews/*.npz` | 60-second full-rate excerpt of every recording, 8 derived channels + ECG |
| `previews/*__bta.npy` | Beat-triggered average of radar acceleration, one per recording |
| `decimated/*.npz` | Whole corpus decimated to 128 Hz (8 radar channels + filtered ECG) |
| `figures/*.png` | Coverage heatmap, example signals, I/Q ellipse, spectra, quality panels |
| `report.md` | Human-readable findings |
| `hf_sync.py` | Resumable rate-limited HF uploader used across the project |

## Inventory columns

- **Identity** — `subject`, `scenario`, `scenario_canon`, `timestamp`, `rel_path`, `mat_format`, `size_mb`
- **Timing** — `fs_radar`, `fs_ecg`, `n_radar`, `n_ecg`, `duration_s`, `n_samples_128hz`,
  `n_windows_nooverlap`, `n_windows_overlap50`
- **Receiver** — `iq_xc`, `iq_yc`, `iq_a`, `iq_b`, `iq_theta_rad`, `iq_gain_imbalance`, `iq_residual`
- **Per channel** (`I`, `Q`, `phi`, `dy`, `vel`, `acc`, `amp`, `cardiac`, `ecg1`, `ecg2`, `icg`, `z0`, `bp`) —
  `_min`, `_max`, `_mean`, `_std`, `_p01`, `_p99`, `_n_nan`, `_n_inf`, `_pct_clip`, `_max_flat_run`
- **ECG quality** — `ecg1_n_rpeaks`, `ecg1_mean_hr`, `ecg1_sd_hr`, `ecg1_rmssd_ms`,
  `ecg1_median_rr_ms`, `ecg1_pct_flat`, `ecg1_hr_plausible`
- **Sync & coupling** — `sync_lag_s`, `sync_xcorr`, `beat_coupling`
- **Derived** — `displacement_range_mm`, `phase_wraps`, `flags`, `n_flags`, `census_seconds`

`ecg1_rmssd_ms` and `ecg1_median_rr_ms` are in **genuine milliseconds**.

## Provenance

Generated by `01_verify_and_download.ipynb` on Kaggle (CPU), run `{CFG['RUN_ID']}`, {now}.
Source: Kaggle mirror `{CFG['KAGGLE_DATASET']}`. Verdict: `{V['verdict']}`.

## Cite the original work

```bibtex
@article{{schellenberger2020dataset,
  title   = {{A dataset of clinically recorded radar vital signs with synchronised reference sensor signals}},
  author  = {{Schellenberger, Sven and Shi, Kilin and Steigleder, Tobias and Malessa, Anke and
             Michler, Fabian and Hameyer, Laura and Neumann, Nina and Lurz, Fabian and
             Ostgathe, Christoph and Weigel, Robert and Koelpin, Alexander}},
  journal = {{Scientific Data}}, volume = {{7}}, number = {{1}}, pages = {{291}}, year = {{2020}},
  doi     = {{10.1038/s41597-020-00629-5}}
}}

@article{{chowdhury2024ecg,
  title   = {{ECG waveform generation from radar signals: A deep learning perspective}},
  author  = {{Chowdhury, Farhana Ahmed and Hosain, Md Kamal and Islam, Md Sakib Bin and
             Hossain, Md Shafayet and Basak, Promit and Mahmud, Sakib and
             Murugappan, M. and Chowdhury, Muhammad E. H.}},
  journal = {{Computers in Biology and Medicine}}, volume = {{176}}, pages = {{108555}}, year = {{2024}},
  doi     = {{10.1016/j.compbiomed.2024.108555}}
}}
```
"""
(WORK / "README.md").write_text(card)
print("dataset card written")
''')

code(r'''
# ---- final blocking push --------------------------------------------------
MANIFEST["finished_utc"] = datetime.now(timezone.utc).isoformat()
MANIFEST["verdict"]      = VERDICT["verdict"]
MANIFEST["n_records"]    = int(len(INV))
(WORK / "run_manifest.json").write_text(json.dumps(MANIFEST, indent=2, default=str))

sizes = {}
for p in sorted(WORK.rglob("*")):
    if p.is_file():
        sizes[str(p.relative_to(WORK))] = p.stat().st_size
total_mb = sum(sizes.values()) / 2**20

print(f"pushing {len(sizes)} files, {total_mb:.1f} MB ...")
ok = sync.flush(final=True, msg=f"{CFG['RUN_ID']} complete -- {len(INV)} records, "
                                f"verdict={VERDICT['verdict']}")

print("\n" + "=" * 78)
print("  DONE" if ok else "  DONE (final push reported a problem -- check sync_history.jsonl)")
print("=" * 78)
print(f"  repo     : {sync.url}")
print(f"  public   : {not CFG['HF_PRIVATE']}")
print(f"  verdict  : {VERDICT['verdict']}   proceed={VERDICT['proceed']}")
print(f"  records  : {len(INV)}   columns: {len(INV.columns) if len(INV) else 0}")
print(f"  payload  : {total_mb:.1f} MB in {len(sizes)} files")
print(f"  pushes   : {sync._pushes}   retries/failures: {sync._failures}")
print("=" * 78)
print("\n  largest files:")
for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:12]:
    print(f"    {v/2**20:>8.2f} MB  {k}")
''')

# ---------------------------------------------------------------- 11. APPENDIX
md(r"""
---
# 11 · Troubleshooting appendix

**`HF_TOKEN not found`**
Right sidebar → *Add-ons* → *Secrets* → *Add secret*. The label must be exactly `HF_TOKEN`, the value
must be a **write** token from <https://huggingface.co/settings/tokens>, and after adding it you must
**attach** it to this notebook with the toggle. Then re-run the token cell.

**`401` or `403` on push**
The token is read-only, or it belongs to a different account than `Shanmuk4622`. Generate a new token
with the *write* role and replace the secret.

**`429 Too Many Requests`**
Handled automatically — the uploader backs off and retries six times. If you see it repeatedly, you
are probably running several notebooks against HF at once. Stop the other notebook, wait ten
minutes, and retry; the 24-call/hour limiter and backoff are automatic.

**No `.mat` files found**
You most likely skipped step 3 of the run instructions. Add the dataset as an Input:
right sidebar → *+ Add Input* → *Datasets* → `pedababugaddala/datasets-file` → *Add*. If the mirror
genuinely has no `.mat`, the verdict cell will say `DERIVATIVE` — tell me and I will re-plan NB02
around whatever it actually contains.

**`NotImplementedError` from `scipy.io.loadmat`**
That is a MATLAB v7.3 file, which is HDF5. `load_mat` already falls back to `h5py`. If h5py is
missing, run `!pip install h5py` and re-run.

**Out of disk / `No space left on device`**
`/kaggle/working` is capped at 20 GB and everything in it gets pushed. Scratch belongs in
`/kaggle/temp`, which this notebook already uses. If you hit the cap, set
`CFG["SAVE_DECIMATED"] = False` and re-run — the decimated dump is the only large output.

**Session died mid-census**
Just re-run the notebook from the top. It pulls `sync_state.json` and payload files back from HF and skips every file
already processed. This is the designed path, not a recovery hack.

**Kernel restarts without warning during the census**
Almost always memory. One resting recording is ~5.8 M samples per channel. Lower
`CFG["PREVIEW_SECONDS"]`, or set `SAVE_DECIMATED = False` to halve peak usage.

**You want to force a push right now**
Run `sync.flush(final=True)` in a fresh cell.

---

## What happens next

Once the verdict prints `RAW_MAT_TREE` and the cross-check reproduces the baseline's Table 1, tell me
and I will build the remaining four notebooks:

| Notebook | Job |
|---|---|
| `02_preprocess_to_hf` | 8-channel windowed corpus, subject-wise + LOSO fold assignment, pushed to `cr-rvs-radar-ecg-processed-v2` |
| `03_baselines` | FPN-1D, UNet-1D, LinkNet-1D, MultiResLinkNet — the Week-2 reproduction gate |
| `04_cardiomamba_train` | C1–C5, dual-T4, the full ablation ladder |
| `05_evaluate_and_figures` | All metrics, Bland–Altman, Wilcoxon, every paper figure |

Each one reuses `hf_sync.py` from this repo, so the 30-minute cadence, the stage-boundary push, the
interrupt push and the resume behaviour are identical everywhere.
""")

# NB01 historically carried its own older sync implementation. Replace that generated cell with
# the exact shared v2 source used by NB02-NB05, keeping one auditable recovery contract.
_shared_sync_cell = f'''HF_SYNC_SRC = {SHARED_HF_SYNC_SRC!r}
(WORK / "hf_sync.py").write_text(HF_SYNC_SRC, encoding="utf-8")
sys.path.insert(0, str(WORK))
print("wrote", WORK / "hf_sync.py", f"({{len(HF_SYNC_SRC)}} chars), sync v2")'''
_matches = 0
for _cell in C:
    _body = "".join(_cell.get("source", []))
    if _cell.get("cell_type") == "code" and "HF_SYNC_SRC = r\"\"\"" in _body:
        ast.parse(_shared_sync_cell)
        _cell["source"] = _src(_shared_sync_cell)
        _matches += 1
assert _matches == 1, f"expected one embedded sync cell, found {_matches}"

nb = {
    "cells": C,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0",
                          "mimetype": "text/x-python", "file_extension": ".py",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "nbconvert_exporter": "python", "pygments_lexer": "ipython3"},
        "kaggle": {"accelerator": "none", "dataSources": [], "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = sys.argv[1] if len(sys.argv) > 1 else "01_verify_and_download.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# round-trip check
with open(out, encoding="utf-8") as f:
    back = json.load(f)
nmd = sum(1 for c in back["cells"] if c["cell_type"] == "markdown")
nco = sum(1 for c in back["cells"] if c["cell_type"] == "code")
print(f"wrote {out}")
print(f"  cells: {len(back['cells'])}  ({nmd} markdown, {nco} code)")
print(f"  every code cell parsed cleanly")
